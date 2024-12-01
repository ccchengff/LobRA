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
from trainer.planner import StaticBatchPlanner, NewStaticBatchPlanner, GroupedStaticBatchPlanner
from trainer.trainer import TrainerConfig, DatasetContext

def dispatch_into_buckets(seq_len_num_distribution, buckets):
    remap_seq_len_num_distribution = {}
    for k, v in seq_len_num_distribution.items():
        if v == 0:
            continue
        # 用bisect_left找到k在buckets中的位置
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
                                                     global_batch_size_list, args.num_gpus, strategy_candidates)
    else:
        static_batch_planner = StaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                  global_batch_size_list, args.num_gpus)
    dataset_ctxs = []
    seq_len_distribution_list = []
    if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
        # seq_len_distribution_list.append({2048: 88, 4096: 24, 8192: 12, 16384: 4})
        # seq_len_distribution_list.append({2048: 96, 4096: 32, 8192: 8, 16384: 2})
        # seq_len_distribution_list.append({2048: 94, 4096: 32, 8192: 8, 16384: 2})
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
            # if train_dataset_pool.get((task_config.dataset_name, task_config.context_length)) is not None:
            if False:
                pass
                # train_dataset = train_dataset_pool[(task_config.dataset_name, task_config.context_length)]
            else:
                train_dataset = dataset_wrapper.create_dataset(
                    dataset_name=task_config.dataset_name,
                    key=task_config.json_key,
                    max_seq_len=task_config.context_length,
                    vocab_file=args.vocab_file,
                    merge_file=args.merge_file,
                    encoder=encoder)
                # train_dataset_pool[(task_config.dataset_name, task_config.context_length)] = train_dataset
            dataset_ctx = DatasetContext(
                dataset=train_dataset,
                steps=task_config.steps,
                epochs=task_config.epochs)
            dataset_ctxs.append(dataset_ctx)
        if os.environ.get("BUCKET_PLAN") == "PLAN_A":
            # Plan A
            # 1. 首先给一个比较粗的粒度，获取数据集在各个bucket的占比分布，乘上gbs得到相应的数量
            # 2. 对粗粒度进行进一步划分（搜索空间），得到一个分布，乘上每个粗粒度bucket的数量得到细粒度bucket内的数量
            # 3. 为保证细粒度bucket的总数与粗粒度相同，如果在乘上gbs后有误差，则给占比最高的细粒度bucket加上seq
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_buckets = train_dataset.get_aligned_buckets(alignment=alignment)
                fine_grained_buckets_of_all_tasks = fine_grained_buckets_of_all_tasks.union(fine_grained_buckets)
            fine_grained_buckets_of_all_tasks = sorted(list(fine_grained_buckets_of_all_tasks))
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length)
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, fine_grained_buckets_of_all_tasks)
                remap_fine_grained_seq_len_distribution = train_dataset.get_distribution_of_fine_grained_buckets(global_batch_size_list[i], seq_len_distribution, fine_grained_seq_len_distribution)
                new_gbs = sum(remap_fine_grained_seq_len_distribution.values())
                static_batch_planner.set_global_batch_size(new_gbs, i)
                fine_grained_seq_len_num_distribution_list.append(remap_fine_grained_seq_len_distribution)
            bucket_candidates = fine_grained_buckets_of_all_tasks
            merge_global_batch_seqlen_list = []
            for i in range(len(fine_grained_seq_len_num_distribution_list)):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                for k, v in seq_len_num_distribution.items():
                    merge_global_batch_seqlen_list.extend([k] * v)
            global_batch_seqlen_list = sorted(merge_global_batch_seqlen_list)
            dp_buckets = get_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
            print(f"dp_buckets = {dp_buckets}")
            for i in range(len(fine_grained_seq_len_num_distribution_list)):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                remap_seq_len_num_distribution = dispatch_into_buckets(seq_len_num_distribution, dp_buckets)
                for k, v in remap_seq_len_num_distribution.items():
                    remap_seq_len_num_distribution[k] = v / static_batch_planner.global_batch_size_list[i]
                seq_len_distribution_list.append(remap_seq_len_num_distribution)
        elif os.environ.get("BUCKET_PLAN") == "PLAN_B":
            # Plan B
            # 1. 给一个粗粒度，乘上gbs得到每个粗粒度bucket的数量
            # 2. 直接从每个bucket里面取实际的seq，组成一个batch
            global_batch_seqlen_list_of_all_tasks = []
            for i in range(trainer_config.train_task_num):
                seq_len_num_bucket = {}
                global_batch_seqlen_list = []
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length)
                for doc_ids in train_dataset.data:
                    sample_len = len(doc_ids) - doc_ids.count(train_dataset.encoder.pad_id())
                    padded_len = max(min(2 ** (sample_len.bit_length()), args.max_seq_length), args.min_seq_length)
                    if padded_len not in seq_len_num_bucket:
                        seq_len_num_bucket[padded_len] = []
                    seq_len_num_bucket[padded_len].append(sample_len)
                new_gbs = 0
                for seq_len, p in seq_len_distribution.items():
                    seq_num = p * static_batch_planner.global_batch_size_list[i]
                    seq_num = math.ceil(seq_num) if seq_num < 1 else round(seq_num)
                    new_gbs += seq_num
                    global_batch_seqlen_list.extend(seq_len_num_bucket[seq_len][:seq_num])
                static_batch_planner.set_global_batch_size(new_gbs, i)
                global_batch_seqlen_list_of_all_tasks.append(global_batch_seqlen_list)
            merge_global_batch_seqlen_list = []
            for i in range(trainer_config.train_task_num):
                merge_global_batch_seqlen_list.extend(global_batch_seqlen_list_of_all_tasks[i])
            merge_global_batch_seqlen_list = sorted(merge_global_batch_seqlen_list)
            bucket_candidates = set()
            for seq_len in merge_global_batch_seqlen_list:
                bucket_candidates.add(int(np.ceil(seq_len / alignment) * alignment))
            bucket_candidates = sorted(list(bucket_candidates))
            dp_buckets = get_buckets_dp(np.array(merge_global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
            print(f"dp_buckets = {dp_buckets}")
            for i in range(trainer_config.train_task_num):
                remap_seq_len_distribution = {}
                global_batch = global_batch_seqlen_list_of_all_tasks[i]
                for seq_len in global_batch:
                    bucket = dp_buckets[bisect.bisect_left(dp_buckets, seq_len)]
                    remap_seq_len_distribution[bucket] = remap_seq_len_distribution.get(bucket, 0) + 1
                for seq_len, num in remap_seq_len_distribution.items():
                    remap_seq_len_distribution[seq_len] = num / static_batch_planner.global_batch_size_list[i]
                seq_len_distribution_list.append(remap_seq_len_distribution)
        elif os.environ.get("BUCKET_PLAN") == "PLAN_C":
            # Plan C
            # 1. 细粒度bucket，获取整个数据集的分布
            # 2. 取gbs的倍数，例如100 x gbs，乘上分布得到每个bucket内的数量
            # 3. 拼成batch，用dp bucket求得buckets
            # 4. 每个任务重新映射到新的bucket，求解
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_buckets = train_dataset.get_aligned_buckets(alignment=alignment)
                fine_grained_buckets_of_all_tasks = fine_grained_buckets_of_all_tasks.union(fine_grained_buckets)
            fine_grained_buckets_of_all_tasks = sorted(list(fine_grained_buckets_of_all_tasks))
            gbs_num = 1e6
            # 确定取多少个global batch
            for i in range(trainer_config.train_task_num):
                gbs_num = min(gbs_num, len(dataset_ctxs[i].dataset) // static_batch_planner.global_batch_size_list[i])
            print(f"gbs_num = {gbs_num}")

            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, fine_grained_buckets_of_all_tasks)
                print(f"distribution of task {i} = {fine_grained_seq_len_distribution}")
                # sample_gbs = min(static_batch_planner.global_batch_size_list[i] * 1000, len(train_dataset))
                sample_gbs = static_batch_planner.global_batch_size_list[i] * gbs_num
                # seq_len_num_distribution = {k: math.ceil(p * sample_gbs) if p * sample_gbs < 1 else round(p * sample_gbs) for k, p in fine_grained_seq_len_distribution.items()}
                seq_len_num_distribution = {k: round(p * sample_gbs) for k, p in fine_grained_seq_len_distribution.items()}
                new_gbs = sum(seq_len_num_distribution.values())
                print(f"gbs of task {i}: {sample_gbs}, {new_gbs}")
                static_batch_planner.set_global_batch_size(new_gbs, i)
                fine_grained_seq_len_num_distribution_list.append(seq_len_num_distribution)
            bucket_candidates = fine_grained_buckets_of_all_tasks
            merge_global_batch_seqlen_list = []
            for i in range(len(fine_grained_seq_len_num_distribution_list)):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                for k, v in seq_len_num_distribution.items():
                    merge_global_batch_seqlen_list.extend([k] * v)
            global_batch_seqlen_list = sorted(merge_global_batch_seqlen_list)
            dp_buckets = get_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
            print(f"dp_buckets = {dp_buckets}")
            for i in range(len(fine_grained_seq_len_num_distribution_list)):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                remap_seq_len_num_distribution = dispatch_into_buckets(seq_len_num_distribution, dp_buckets)
                for k, v in remap_seq_len_num_distribution.items():
                    remap_seq_len_num_distribution[k] = v / static_batch_planner.global_batch_size_list[i]
                seq_len_distribution_list.append(remap_seq_len_num_distribution)
        elif os.environ.get("BUCKET_PLAN") == "PLAN_D":
            # Plan D
            # 1. 细粒度bucket，获取整个数据集的分布
            # 2. 取gbs的倍数，例如100 x gbs，乘上分布得到每个bucket内的数量
            # 3. 拼成batch，用dp bucket求得buckets
            # 4. 每个任务用新的buckets求分布，乘上各自的gbs，再进行schedule
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_buckets = train_dataset.get_aligned_buckets(alignment=alignment)
                fine_grained_buckets_of_all_tasks = fine_grained_buckets_of_all_tasks.union(fine_grained_buckets)
            fine_grained_buckets_of_all_tasks = sorted(list(fine_grained_buckets_of_all_tasks))
            gbs_num = 1e6
            # 确定取多少个global batch
            for i in range(trainer_config.train_task_num):
                gbs_num = min(gbs_num, len(dataset_ctxs[i].dataset) // static_batch_planner.global_batch_size_list[i])
            print(f"gbs_num = {gbs_num}")

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
                new_gbs = sum(seq_len_num_distribution.values())
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
            print(f"dp_buckets = {dp_buckets}")
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                print(f"{i}: {fine_grained_seq_len_distribution}")
                seq_len_distribution_list.append(fine_grained_seq_len_distribution)
        else:
            if args.bucket_num == 7:
                dp_buckets = [256, 512, 1024, 2048, 4096, 8192, 16384] # 7 bucket
            elif args.bucket_num == 16:
                dp_buckets = [144, 256, 304, 512, 640, 800, 1024, 1216, 1504, 1888, 2656, 4096, 4256, 5840, 8192, 16384]
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(seq_len_distribution)

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
    args = parser.parse_args()
    test_static_planner(args)