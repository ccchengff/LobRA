import os
import argparse
import socket
import hetu as ht
from utils import read_ds_parallel_config, distributed_init
from model import LLamaConfig, LLamaLMHeadModel, QKVFusedLLamaLMHeadModel, PackedLLamaLMHeadModel
from profiler.cost_model import CostModel
from data_utils import GPTJsonDataset
from peft.lora import MultiLoraModel
from trainer import Trainer, TrainerConfig, DatasetWrapper, ModelWrapper, OptimizerWrapper

def finetune(args, ds_parallel_configs=None, max_tokens_list=None):
    trainer_config = TrainerConfig(args.trainer_config_path)
    model_config = LLamaConfig(
        vocab_size=args.vocab_size, 
        ffn_hidden_size=args.ffn_hidden_size,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads, 
        n_layer=args.num_layers,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        use_flash_attn=args.use_flash_attn)
    cost_model = CostModel(model_config, 'llama', trainer_config.train_task_num, \
                           args.trainer_config_path, args.profile_path, \
                           sp=args.sp)
    if ds_parallel_configs is None:
        ds_parallel_configs = read_ds_parallel_config(args)
    # build symbolic shape
    model_config.dp_symbol = ht.IntSymbol(1)
    
    # simple check for blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == model_config.num_hidden_layers - 1, \
        f"blocks range: {ranges} is conflict with num_hidden_layers: {model_config.num_hidden_layers}!"

    # wrapper
    is_pack = False
    if trainer_config.variant == 'fused':
        model_wrapper = ModelWrapper(QKVFusedLLamaLMHeadModel, model_config)
    elif trainer_config.variant == 'packed':
        model_wrapper = ModelWrapper(PackedLLamaLMHeadModel, model_config)
        is_pack = True
    else:
        model_wrapper = ModelWrapper(LLamaLMHeadModel, model_config)
    finetune_model_wrapper = ModelWrapper(MultiLoraModel, model_config)
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)
    
    # trainer
    trainer = Trainer(args, dataset_wrapper, model_wrapper, finetune_model_wrapper, optimizer_wrapper, \
                      trainer_config, cost_model, ds_parallel_configs, max_tokens_list, is_pack=is_pack)
    
    # build graph
    trainer.build_model(args, ds_parallel_configs)
    # train
    if is_pack:
        trainer.packed_run(args)
    else:
        trainer.run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_two_node", action="store_true", help="use 2x8 gpus to run script."
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
    )
    parser.add_argument(
        "--sp", type=int, default=1, help="sp option"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="max seq length of samples"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=256, help="min seq length of samples"
    )
    parser.add_argument(
        "--max_tokens", type=str, default="", help="max tokens of each strategy"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
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
        "--bucket_num", type=int, default=16, help="Number of dp bucket"
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
    args = parser.parse_args()
    distributed_init(args.use_two_node)
    finetune(args)
    print(f'train hetu ds parallel end...')