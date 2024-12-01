import os
import json
import subprocess
import argparse
import hetu as ht
from utils import read_from_csv
from model import LLamaConfig
from data_utils import GPTJsonDataset
from profiler import CostModel
from trainer import DatasetWrapper
from trainer.trainer import TrainerConfig, DatasetContext
from trainer.planner import StaticBatchPlanner, NewStaticBatchPlanner, GroupedStaticBatchPlanner

def profile_throughput(args, train_task_num):
    if os.path.exists(args.throughput_path):
        return read_from_csv(args.throughput_path)
    else:
        cmd = f"bash scripts/new_profile_throughput.sh \
                {args.num_layers} {args.hidden_size} {args.num_attention_heads} {train_task_num} \
                {args.throughput_path} {args.max_tokens_path} {args.trainer_config_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            return read_from_csv(args.throughput_path)
        except:
            return None

def run_finetune(args, train_task_num, ds_parallel_config):
    # generate ds config
    ds_config_list = []
    dps = ds_parallel_config['dps']
    tps = ds_parallel_config['tps']
    pps = ds_parallel_config['pps']
    max_tokens_list = ds_parallel_config['max_tokens']
    num_gpus = 0
    for dp, tp, pp, max_tokens in zip(dps, tps, pps, max_tokens_list):
        num_gpus += dp * tp * pp
        ds_config_list.append(f"({dp}, {tp}, {pp}, {max_tokens})")
    dps_str = ",".join([str(dp) for dp in dps])
    tps_str = ",".join([str(tp) for tp in tps])
    pps_str = ",".join([str(pp) for pp in pps])
    max_tokens_str = ",".join([str(max_tokens) for max_tokens in max_tokens_list])
    ds_config_str = ", ".join(ds_config_list)
    # print strategy from planner
    print(f"strategy from static planner (dp, tp, pp, max_tokens): {ds_config_str}")
    # print("generate ds parallel config from static planner...")
    # ds_parallel_configs = generate_ds_parallel_config(args.num_layers, num_gpus=num_gpus, dps=dps, tps=tps, pps=pps, sps=sps)
    # save_folder = args.save_folder
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # for i, ds_parallel_config in enumerate(ds_parallel_configs):
    #     file_name = f'dp{dps[i]}_tp{tps[i]}_pp{pps[i]}_{int(sps[i])}_lora_{i}.json'
    #     with open(f'{save_folder}/{file_name}', 'w') as f:
    #         json.dump(ds_parallel_config, f, indent=4)
    # ds_parallel_config_path = "@".join([f"{save_folder}/dp{dps[i]}_tp{tps[i]}_pp{pps[i]}_{int(sps[i])}_lora_{i}.json" for i in range(len(ds_parallel_configs))])

    # run finetune
    
    # cmd = f"bash scripts/{args.model_type}_lora_multi_task.sh \
    #         {args.num_layers} {args.hidden_size} {args.num_attention_heads} {train_task_num} \
    #         {max_tokens_str} {dps_str} {tps_str} {pps_str} {args.sp} \
    #         {args.max_seq_length} {args.min_seq_length} \
    #         {args.trainer_config_path} {args.profile_fw_path} {args.profile_bw_path} \
    #         {args.vocab_file} {args.merge_file}"
    
    # cmd = f"mpirun --allow-run-as-root -np ${num_gpus} \
    #         --output-filename logs/multi_task/ds_parallel_gpt_${dps_str}_${tps_str}_${pps_str}_${args.sp}_${max_tokens_str} --merge-stderr-to-stdout \
    #         python3 scripts/${args.model_type}_lora_multi_task.py \
    #         --ds_parallel_config ${ds_parallel_config_path} \
    #         --trainer_config_path ${args.trainer_config_path} \
    #         --profile_fw_path ${args.profile_fw_path} \
    #         --profile_bw_path ${args.profile_bw_path} \
    #         --max_tokens ${max_tokens_str} \
    #         --vocab_file ${args.vocab_file} \
    #         --merge_file ${args.merge_file} \
    #         --vocab_size 30592 \
    #         --hidden_size ${args.hidden_size} \
    #         --num_layers ${args.num_layers} \
    #         --num_attention_heads ${args.num_attention_heads} \
    #         --max_seq_length ${args.max_seq_length} \
    #         --min_seq_length ${args.min_seq_length} \
    #         --lr 1e-4 \
    #         --adam_weight_decay 0.01 \
    #         --hidden_act relu \
    #         --dropout_prob 0.1 \
    #         --bf16 \
    #         --use_flash_attn"
    
    # try:
    #     subprocess.run(cmd, shell=True, check=True)
    #     return 0
    # except:
    #     return -1

def multi_task_workflow(args):
    trainer_config = TrainerConfig(args.trainer_config_path)
    cost_model_config = LLamaConfig(
        ffn_hidden_size=args.ffn_hidden_size,
        n_layer=args.num_layers,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads)
    cost_model = CostModel(cost_model_config, args.model_type, trainer_config.train_task_num, \
                           args.trainer_config_path, args.profile_path, \
                           max_tokens_path=args.max_tokens_path, sp=args.sp)
    if os.environ.get('HETU_STRATEGY_FILTER') == "PROFILE":
        # 基于profile
        strategy_candidates = profile_throughput(args, trainer_config.train_task_num)
        # print(f"cadidates: {strategy_candidates}")
    else:
        # 基于cost model
        strategy_candidates = cost_model.get_strategy_candidates(args.num_layers, num_micro_batches=64, throughput_path=args.throughput_path)
    global_batch_size_list = trainer_config.get_global_batch_size_list()
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)
    data_dispatch_pattern = os.environ.get('HETU_DATA_DISPATCH')
    print(f"create static {data_dispatch_pattern} batch planner...")
    if data_dispatch_pattern == 'GROUP':
        static_batch_planner = GroupedStaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                         global_batch_size_list, args.num_gpus, strategy_candidates)
    elif data_dispatch_pattern == 'BALANCE':
        static_batch_planner = NewStaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                     global_batch_size_list, args.num_gpus, strategy_candidates)
    else:
        static_batch_planner = StaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                  global_batch_size_list, args.num_gpus)
    # 分析数据集分布
    dataset_ctxs = []
    seq_len_distribution_list = []
    for i in range(trainer_config.train_task_num):
        task_config = trainer_config.task_configs[i]
        train_dataset = dataset_wrapper.create_dataset(
            dataset_name=task_config.dataset_name,
            key=task_config.json_key,
            max_seq_len=args.max_seq_length,
            vocab_file=args.vocab_file,
            merge_file=args.merge_file)
        dataset_ctx = DatasetContext(
            dataset=train_dataset,
            steps=task_config.steps,
            epochs=task_config.epochs)
        dataset_ctxs.append(dataset_ctx)
        seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length)
        seq_len_distribution_list.append(seq_len_distribution)
    ds_parallel_config = static_batch_planner.schedule(seq_len_distribution_list)
    run_finetune(args, trainer_config.train_task_num, ds_parallel_config)

if __name__ == "__main__":
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
        "--num_gpus", type=int, default=8, help="Number of gpus"
    )
    parser.add_argument(
        "--sp", type=int, default=0, help="sp option"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--trainer_config_path", type=str, default='', help="Trainer config path of multi-task training."
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    # parser.add_argument(
    #     "--profile_fw_path", type=str, default='', help="profile fw path of profiler."
    # )
    # parser.add_argument(
    #     "--profile_bw_path", type=str, default='', help="profile bw path of profiler."
    # )
    parser.add_argument(
        "--max_tokens_path", type=str, default='', help="max tokens path of profiler."
    )
    parser.add_argument(
        "--throughput_path", type=str, default='', help="throughput path of profiler."
    )
    args = parser.parse_args()
    multi_task_workflow(args)
