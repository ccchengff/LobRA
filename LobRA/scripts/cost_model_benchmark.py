import os
import argparse
import hetu as ht
import numpy as np
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from model import GPTConfig, GPTLMHeadModel, QKVFusedGPTLMHeadModel, LLamaConfig, LLamaLMHeadModel, QKVFusedLLamaLMHeadModel
from utils import distributed_init, generate_ds_parallel_config, assign_global_to_all_variables, write_to_csv, read_from_csv
from peft.lora import MultiLoraModel
from profiler import Profiler

def run_benchmark_gpt(profile_args, seq_len_range, profile_mbs, profile_path):
    local_device = ht.local_device()
    num_gpus = profile_args.dp * profile_args.tp * profile_args.pp
    ds_parallel_configs = generate_ds_parallel_config(profile_args.num_layers, num_gpus, [profile_args.dp], [profile_args.tp], [profile_args.pp], [bool(profile_args.sp)], False)
    ds_parallel_configs = [assign_global_to_all_variables(ds_parallel_config) for ds_parallel_config in ds_parallel_configs]
    model_config = GPTConfig(
        vocab_size=profile_args.vocab_size,
        n_embd=profile_args.hidden_size,
        n_head=profile_args.num_attention_heads,
        n_layer=profile_args.num_layers,
        resid_pdrop=profile_args.dropout_prob,
        embd_pdrop=profile_args.dropout_prob,
        attn_pdrop=profile_args.dropout_prob,
        use_flash_attn=profile_args.use_flash_attn,
    )
    model_config.dp_symbol = ht.IntSymbol(1)
    
    # simple check for blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == model_config.num_hidden_layers - 1, \
        f"blocks range: {ranges} is conflict with num_hidden_layers: {model_config.num_hidden_layers}!"
    
    # wrapper
    if trainer_config.variant == 'fused':
        model_wrapper = ModelWrapper(QKVFusedGPTLMHeadModel, model_config)
    else:
        model_wrapper = ModelWrapper(GPTLMHeadModel, model_config)
    finetune_model_wrapper = ModelWrapper(MultiLoraModel, model_config)
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    trainer_config = TrainerConfig(profile_args.trainer_config_path)
    assert profile_args.train_task_num == trainer_config.train_task_num
    
    # profiler
    profile_args.default_seq_len = seq_len_range[0]
    profile_args.default_mbs = profile_mbs[0]
    profiler = Profiler(profile_args, model_wrapper, finetune_model_wrapper, optimizer_wrapper, trainer_config, ds_parallel_configs)
    
    # build graph
    profiler.build_model(profile_args, ds_parallel_configs)
    
    # profile
    for seq_len in seq_len_range:
        for mbs in profile_mbs:
            print(f"profiling: (tp, seq_len, mbs) = ({profile_args.tp}, {seq_len}, {mbs})")
            profiler.profile(mbs, seq_len)
            if local_device.index == 0:
                total_stream_time = profiler.total_stream_time
                block_time = profiler.block_stream_time
                total_time_entry = {
                    'tp': profile_args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(total_stream_time)
                }
                block_time_entry = {
                    'tp': profile_args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(block_time)    
                }
                if profile_args.num_layers <= 3:
                    write_to_csv(block_time_entry, profile_path)
                else:
                    write_to_csv(total_time_entry, profile_args.validation_path)

def run_benchmark_llama(profile_args, seq_len_range, profile_mbs, profile_path):
    local_device = ht.local_device()
    num_gpus = profile_args.dp * profile_args.tp * profile_args.pp
    ds_parallel_configs = generate_ds_parallel_config(profile_args.num_layers, num_gpus, [profile_args.dp], [profile_args.tp], [profile_args.pp], [profile_args.sp], False)
    ds_parallel_configs = [assign_global_to_all_variables(ds_parallel_config) for ds_parallel_config in ds_parallel_configs]
    model_config = LLamaConfig(
        vocab_size=profile_args.vocab_size,
        ffn_hidden_size=profile_args.ffn_hidden_size,
        n_embd=profile_args.hidden_size,
        n_head=profile_args.num_attention_heads,
        n_layer=profile_args.num_layers,
        resid_pdrop=profile_args.dropout_prob,
        embd_pdrop=profile_args.dropout_prob,
        attn_pdrop=profile_args.dropout_prob,
        use_flash_attn=profile_args.use_flash_attn,
    )
    model_config.dp_symbol = ht.IntSymbol(1)
    
    # simple check for blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == model_config.num_hidden_layers - 1, \
        f"blocks range: {ranges} is conflict with num_hidden_layers: {model_config.num_hidden_layers}!"
    
    # wrapper
    trainer_config = TrainerConfig(profile_args.trainer_config_path)
    if trainer_config.variant == 'fused':
        model_wrapper = ModelWrapper(QKVFusedLLamaLMHeadModel, model_config)
    else:
        model_wrapper = ModelWrapper(LLamaLMHeadModel, model_config)
    finetune_model_wrapper = ModelWrapper(MultiLoraModel, model_config)
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    assert profile_args.train_task_num == trainer_config.train_task_num
    
    # profiler
    profile_args.default_seq_len = seq_len_range[0]
    profile_args.default_mbs = profile_mbs[0]
    profiler = Profiler(profile_args, model_wrapper, finetune_model_wrapper, optimizer_wrapper, trainer_config, ds_parallel_configs)
    
    # build graph
    profiler.build_model(profile_args, ds_parallel_configs)
    
    # read from cache
    cache_dict = None
    if profile_args.num_layers <= 3:
        rows = read_from_csv(profile_path)
        cache_dict = {(row['tp'], row['seq_len'], row['mbs']) : row['time'] for row in rows}
    else:
        rows = read_from_csv(profile_args.validation_path)
        cache_dict = {(row['tp'], row['seq_len'], row['mbs']) : row['time'] for row in rows}
    # profile
    for seq_len in seq_len_range:
        for mbs in profile_mbs:
            if cache_dict is not None and (profile_args.tp, seq_len, mbs) in cache_dict:
                continue
            print(f"profiling: (tp, seq_len, mbs) = ({profile_args.tp}, {seq_len}, {mbs})")
            profiler.profile(mbs, seq_len)
            if local_device.index == 0:
                total_stream_time = profiler.total_stream_time
                block_time = profiler.block_time
                total_time_entry = {
                    'tp': profile_args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(total_stream_time)
                }
                block_time_entry = {
                    'tp': profile_args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(block_time)    
                }
                if profile_args.num_layers <= 3:
                    write_to_csv(block_time_entry, profile_path)
                else:
                    write_to_csv(total_time_entry, profile_args.validation_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_two_node", action="store_true", help="use 2x8 gpus to run script."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )
    parser.add_argument(
        '--tp', type=int, default=1, help='tp degree'
    )
    parser.add_argument(
        '--sp', type=int, default=0, help='sp option'
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
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
        "--profile_steps", type=int, default=100, help="profile steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="warmup steps"
    )
    parser.add_argument(
        "--train_task_num", type=int, default=1, help="Number of layers"
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
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--model_type", type=str, default='gpt', help="profile fw path of profiler."
    )
    parser.add_argument(
        "--validation_path", type=str, default='', help="validation path of profiler."
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    parser.add_argument(
        "--profile_memory_path", type=str, default='', help="profile memory path of profiler."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--seq_len_range", type=str, default='', help="profile seq len range"
    )
    parser.add_argument(
        "--profile_mbs", type=str, default='', help="profile micro batch size"
    )
    profile_args = parser.parse_args()
    profile_args_dict = vars(profile_args)
    profile_args_dict['dp'] = 1
    profile_args_dict['pp'] = 1
    profile_args = argparse.Namespace(**profile_args_dict)
    if profile_args.profile_mbs == '':
        if profile_args.num_layers <= 3:
            profile_mbs = [1, 2, 4, 8, 16]
        else:
            profile_mbs = [1, 2, 4, 8, 16]
    else:
        profile_mbs = list(map(int, profile_args.profile_mbs.split(',')))
    if profile_args.seq_len_range == '':
        seq_len_range = [256, 512, 1024, 2048, 4096, 8192, 16384]
    else:
        seq_len_range = list(map(int, profile_args.seq_len_range.split(',')))
    if os.path.exists(profile_args.profile_memory_path):
        rows = read_from_csv(profile_args.profile_memory_path)
        memory_dict = {(row['tp'], row['pp'], row['sp']) : row['max_tokens'] for row in rows}
        max_tokens = 0
        for ds_config, tokens in memory_dict.items():
            if ds_config[0] == profile_args.tp:
                max_tokens = max(max_tokens, tokens)
        seq_len_range = [seq_len for seq_len in seq_len_range if seq_len <= max_tokens]
    seq_len_range = [256, 512, 1024, 2048, 4096, 8192, 16384]
    distributed_init(profile_args.use_two_node)
    if profile_args.model_type == 'gpt':
        run_benchmark_gpt(profile_args, seq_len_range, profile_mbs, profile_args.profile_path)
    elif profile_args.model_type == 'llama':
        run_benchmark_llama(profile_args, seq_len_range, profile_mbs, profile_args.profile_path)
        
