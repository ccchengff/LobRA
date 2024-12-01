import os
import numpy as np
import argparse
import bisect
import math
import time
from trainer.dp_bucket import get_buckets_dp
from trainer.utils.wrapper import DatasetWrapper
from data_utils import GPTJsonDataset, Encoder
from model import LLamaConfig
from profiler import CostModel
from types import SimpleNamespace
from trainer.planner import NewStaticBatchPlanner, GroupedStaticBatchPlanner
from trainer.trainer import TrainerConfig, DatasetContext

def dispatch_into_buckets(seq_len_num_distribution, buckets):
    remap_seq_len_num_distribution = {}
    for k, v in seq_len_num_distribution.items():
        if v == 0:
            continue
        bucket = buckets[bisect.bisect_left(buckets, k)]
        remap_seq_len_num_distribution[bucket] = remap_seq_len_num_distribution.get(bucket, 0) + v
    return remap_seq_len_num_distribution

def test_static_planner(args):
    trainer_config = TrainerConfig(args.trainer_config_path)
    cost_model_config = LLamaConfig(
        n_layer=args.num_layers,
        ffn_hidden_size=args.ffn_hidden_size,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads)
    cost_model = CostModel(cost_model_config, args.model_type, trainer_config.train_task_num, \
                           args.trainer_config_path, args.profile_path, \
                           max_tokens_path=args.max_tokens_path, sp=args.sp)
    strategy_candidates = cost_model.get_strategy_candidates(args.num_layers)
    global_batch_size_list = trainer_config.get_global_batch_size_list()
    data_dispatch_pattern = os.environ.get('HETU_DATA_DISPATCH')
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)
    if data_dispatch_pattern == 'GROUP':
        static_batch_planner = GroupedStaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                         global_batch_size_list, args.num_gpus, strategy_candidates)
    elif data_dispatch_pattern == 'BALANCE':
        static_batch_planner = NewStaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                     global_batch_size_list, args.num_gpus, strategy_candidates, args.strategy_proposal)
    else:
        print("Please set HETU_DATA_DISPATCH to GROUP or BALANCE")
        exit(-1)
    dataset_ctxs = []
    seq_len_distribution_list = []
    if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
        seq_len_distribution_list.append({256: 7, 512: 18, 1024: 33, 2048: 9, 4096: 1, 8192: 1})
        num = sum(seq_len_distribution_list[0].values())
        seq_len_distribution_list[0] = {key: value / num for key, value in seq_len_distribution_list[0].items()}
        static_batch_planner.global_batch_size_list = [num]
    else:
        encoder_args = {
            'key': 'text',
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': args.vocab_file,
            'merge_file': args.merge_file,
        }
        encoder_args = SimpleNamespace(**encoder_args)
        encoder = Encoder(encoder_args)
        train_dataset_pool = {}
        fine_grained_seq_len_num_distribution_list = []
        fine_grained_buckets_of_all_tasks = set()
        bucket_limit = args.bucket_num
        alignment = 16
        for i in range(trainer_config.train_task_num):
            task_config = trainer_config.task_configs[i]
            if train_dataset_pool.get((task_config.dataset_name, task_config.context_length)) is not None:
                train_dataset = train_dataset_pool[(task_config.dataset_name, task_config.context_length)]
            else:
                train_dataset = dataset_wrapper.create_dataset(
                    dataset_name=task_config.dataset_name,
                    key=task_config.json_key,
                    max_seq_len=task_config.context_length,
                    vocab_file=args.vocab_file,
                    merge_file=args.merge_file,
                    encoder=encoder)
                train_dataset_pool[(task_config.dataset_name, task_config.context_length)] = train_dataset
            dataset_ctx = DatasetContext(
                dataset=train_dataset,
                steps=task_config.steps,
                epochs=task_config.epochs)
            dataset_ctxs.append(dataset_ctx)
        if os.environ.get("BUCKET_PLAN") == "DYNAMIC":
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_buckets = train_dataset.get_aligned_buckets(alignment=alignment)
                fine_grained_buckets_of_all_tasks = fine_grained_buckets_of_all_tasks.union(fine_grained_buckets)
            fine_grained_buckets_of_all_tasks = sorted(list(fine_grained_buckets_of_all_tasks))

            max_seq_len = 0
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, fine_grained_buckets_of_all_tasks)
                # print(f"distribution of task {i} = {fine_grained_seq_len_distribution}")
                max_seq_len = max(max_seq_len, max(fine_grained_seq_len_distribution.keys()))
                # sample_gbs = min(static_batch_planner.global_batch_size_list[i] * 1000, len(train_dataset))
                sample_gbs = static_batch_planner.global_batch_size_list[i] * 100
                # seq_len_num_distribution = {k: math.ceil(p * sample_gbs) if p * sample_gbs < 1 else round(p * sample_gbs) for k, p in fine_grained_seq_len_distribution.items()}
                seq_len_num_distribution = {k: round(p * sample_gbs) for k, p in fine_grained_seq_len_distribution.items()}
                # print(f"gbs of task {i}: {sample_gbs}, {new_gbs}")
                # static_batch_planner.set_global_batch_size(new_gbs, i)
                fine_grained_seq_len_num_distribution_list.append(seq_len_num_distribution)
            bucket_candidates = fine_grained_buckets_of_all_tasks
            merge_global_batch_seqlen_list = []
            has_max_seq_len = False
            for i in range(trainer_config.train_task_num):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                for k, v in seq_len_num_distribution.items():
                    if k == max_seq_len and v > 0:
                        has_max_seq_len = True
                    merge_global_batch_seqlen_list.extend([k] * v)
            if not has_max_seq_len:
                merge_global_batch_seqlen_list.append(max_seq_len)
            global_batch_seqlen_list = sorted(merge_global_batch_seqlen_list)
            dp_buckets = get_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(fine_grained_seq_len_distribution)
        elif os.environ.get("BUCKET_PLAN") == "STATIC":
            if args.bucket_num == 7:
                dp_buckets = [256, 512, 1024, 2048, 4096, 8192, 16384] # 7 bucket
            elif args.bucket_num == 16:
                dp_buckets = [144, 256, 304, 512, 640, 800, 1024, 1216, 1504, 1888, 2656, 4096, 4256, 5840, 8192, 16384]
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(seq_len_distribution)
        else:
            print("Please set BUCKET_PLAN to DYNAMIC or STATIC")
            exit(-1)

    # print(seq_len_distribution_list)
    s_time = time.time()
    static_batch_planner.schedule(seq_len_distribution_list)
    e_time = time.time()
    print(f"schedule time = {e_time - s_time:.3f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="llama", help="finetune base model type"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="max seq length of samples"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=256, help="min seq length of samples"
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=768, help="FFN hidden size of llama model",
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "--sp", type=int, default=0, help="sp option"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="gpu num"
    )
    parser.add_argument(
        "--bucket_num", type=int, default=16, help="bucket num"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--trainer_config_path", type=str, default='', help="Trainer config path of multi-task training."
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    parser.add_argument(
        "--max_tokens_path", type=str, default='', help="max tokens path of profiler."
    )
    parser.add_argument(
        "--strategy_proposal", type=int, default=1, help="use optimized strategy pool"
    )
    args = parser.parse_args()
    test_static_planner(args)