import os
import time
import signal
import argparse
import hetu as ht
import numpy as np
from tqdm import tqdm
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from model import LLamaConfig, LLamaLMHeadModel, QKVFusedLLamaLMHeadModel
from utils import parse_multi_ds_parallel_config, distributed_init, generate_ds_parallel_config, assign_global_to_all_variables, write_to_csv
from data_utils import get_position_ids
from trainer.batch_scheduler import make_micro_batches
from profiler.cost_model import CostModel
from peft.lora import MultiLoraModel

def generate_micro_batches(seq_len_range, num_micro_batches, max_tokens, train_task_num):
    seq_len_range = sorted(seq_len_range, reverse=True)
    seq_len_num = len(seq_len_range)
    num_fragment = num_micro_batches % seq_len_num
    num_complete = num_micro_batches // seq_len_num
    if num_fragment == 0:
        if num_complete > 1:
            num_complete -= 1
            num_fragment += 1
    micro_batches = []
    if num_complete > 0:
        for seq_len in seq_len_range:
            micro_batches.extend(make_micro_batches(max_tokens // seq_len, num_complete, seq_len, train_task_num))
    if num_fragment > 0:
        possible_fragment_seq_len_range = sorted([seq_len for seq_len in seq_len_range if max_tokens // seq_len > 1], reverse=True)
        num_fragment_of_seq_len = num_fragment // len(possible_fragment_seq_len_range)
        rest_fragment_num = num_fragment % len(possible_fragment_seq_len_range)
        for seq_len in possible_fragment_seq_len_range:
            micro_batch_num = num_fragment_of_seq_len
            if rest_fragment_num > 0:
                micro_batch_num += 1
                rest_fragment_num -= 1
            if micro_batch_num > 0:
                micro_batches.extend(make_micro_batches(1, micro_batch_num, seq_len, train_task_num))
    micro_batches = sorted(micro_batches, key=lambda x: (x.seq_length * x.batch_size, x.seq_length), reverse=True)  
    print(f"======generated micro batches======")
    for i, micro_batch in enumerate(micro_batches):
        print(f"batch {i}: {micro_batch.batch_size} x {micro_batch.seq_length}")
    print(f"======generated micro batches======")
    return micro_batches

def fit_time(X, c1, c2, c3, c4, c5, c6):
    mbs, seq_len = X
    return (c1 * mbs + c2) * seq_len * seq_len + (c3 * mbs + c4) * seq_len + (c5 * mbs + c6)

def get_estimated_time(cost_model, tp, pp, num_layers, micro_batches):
    max_batch_time = 0
    total_time = 0
    for micro_batch in micro_batches:
        mbs = micro_batch.batch_size
        seq_len = micro_batch.seq_length
        # t = (mbs * cost_model.slopes[(seq_len, tp)] + cost_model.biases[(seq_len, tp)]) * num_layers / pp
        t = fit_time((mbs, seq_len), *cost_model.popt[tp]) * num_layers / pp if mbs != 0 else 0
        total_time += t
        max_batch_time = max(max_batch_time, t)
    total_time += (pp - 1) * max_batch_time
    return total_time

def test_cost_model(args, seq_len_range):
    if args.bf16:
        precision = "ht.bfloat16"
    else:
        precision = "ht.float32"
    precision = eval(precision)

    local_device = ht.local_device()
    num_gpus = args.dp * args.tp * args.pp
    ds_parallel_configs = generate_ds_parallel_config(args.num_layers, num_gpus, [args.dp], [args.tp], [args.pp], [args.sp], False)
    ds_parallel_configs = [assign_global_to_all_variables(ds_parallel_config) for ds_parallel_config in ds_parallel_configs]
    model_config = LLamaConfig(
        vocab_size=args.vocab_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads,
        n_layer=args.num_layers,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        use_flash_attn=args.use_flash_attn,
    )
    
    # simple check for blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == model_config.num_hidden_layers - 1, \
        f"blocks range: {ranges} is conflict with num_hidden_layers: {model_config.num_hidden_layers}!"

    trainer_config = TrainerConfig(args.trainer_config_path)
    cost_model = CostModel(model_config, 'llama', trainer_config.train_task_num, \
                           args.trainer_config_path, args.profile_path, sp=args.sp)

    # wrapper
    if trainer_config.variant == 'fused':
        pretrained_model_wrapper = ModelWrapper(QKVFusedLLamaLMHeadModel, model_config)
    else:
        pretrained_model_wrapper = ModelWrapper(LLamaLMHeadModel, model_config)
    finetune_model_wrapper = ModelWrapper(MultiLoraModel, model_config)
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    assert args.train_task_num == trainer_config.train_task_num
    
    # profiler
    args.default_seq_len = args.max_tokens
    args.default_mbs = 1
    
    # build model
    with ht.graph("define_and_run", num_strategy=1):
        with ht.autocast(precision):
            input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
            label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
            task_multi_ds, task_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
            
            default_seq_len = args.max_tokens
            default_mbs_times_dp = 1 * input_multi_ds[0].get_dim(0)
            default_batch_offset = 0
            default_batch_size = 1
            pretrained_model_wrapper.model_config.dp_symbol = ht.IntSymbol(args.dp)
    
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
            position_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')
            task_batch_idxs = []
            for i in range(args.train_task_num):
                task_batch_idxs.append(ht.parallel_placeholder(ht.int64, global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                                                            multi_ds=task_multi_ds, device_groups=task_device_groups, name='task_batch_idxs_task{}'.format(i), is_cpu=True))
            pretrained_model_wrapper.model_config.train_task_num = trainer_config.train_task_num
            
            pretrained_model = pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
            peft_configs = [task_config.lora_config for task_config in trainer_config.task_configs]
            model = finetune_model_wrapper.create_model(model=pretrained_model, peft_configs=peft_configs)
            loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=None,
                labels=masked_lm_labels,
                task_batch_idxs=task_batch_idxs
            )
            opt = optimizer_wrapper.create_optimizer(lr=args.lr)
            train_op = opt.minimize(loss)
    
    # micro_batch_num = args.num_micro_batches
    max_tokens = args.max_tokens
    pretrained_model.config.dp_symbol.set_data(args.dp)
    e2e_time = []
    # total_stream_time = []
    # print(f"run llama benchmark strategy config: (dp, tp, pp, sp) = ({args.dp}, {args.tp}, {args.pp}, {args.sp})")
    # print(f"run llama benchmark data config: (mbs, seq_len, num_micro_batches) = ({mbs}, {args.seq_length}, {args.num_micro_batches})")
    _profile_option = False if args.save_path == "null" else True
    pbar = None
    if local_device.index == 0:
        pbar = tqdm(total=args.profile_steps) if _profile_option else None
    if args.warmup_steps > 0:
        print(f"Warmup takes {args.warmup_steps} steps...")
    input_ids_list = []
    position_ids_list = []
    masked_lm_labels_list = []
    task_batch_idxs_list = [[] for _ in range(args.train_task_num)]
    # run
    # test_micro_batches = make_micro_batches(mbs, micro_batch_num, args.seq_length, args.train_task_num)
    test_micro_batches = generate_micro_batches(seq_len_range, args.num_micro_batches, args.max_tokens, args.train_task_num)
    estimated_time = get_estimated_time(cost_model, args.tp, args.pp, args.num_layers, test_micro_batches)
    print(f"estimated time = {estimated_time}")
    micro_batch_num = len(test_micro_batches)
    mbs_list = []
    seq_len_list = []
    for test_micro_batch in test_micro_batches:
        mbs_list.append(test_micro_batch.batch_size)
        seq_len_list.append(test_micro_batch.seq_length)
        mbs = test_micro_batch.batch_size
        batch_data = np.array(test_micro_batch.batch_data).reshape(mbs, -1)
        labels = batch_data[:, 1:]
        tokens = batch_data[:, :-1]
        _position_ids = get_position_ids(mbs, test_micro_batch.seq_length)
        # _token_type_ids = np.zeros([mbs, args.seq_length])
        task_batch_idxs_list = []
        for train_task_idx in range(args.train_task_num):
            task_batch_idxs_i = np.zeros([test_micro_batch.batch_offset_list[train_task_idx], test_micro_batch.batch_size_list[train_task_idx], test_micro_batch.batch_size], dtype=np.int64)
            task_batch_idxs_list.append(task_batch_idxs_i)
        input_ids_list.append(tokens.astype(np.int64))
        position_ids_list.append(_position_ids.astype(np.int64))
        # token_type_ids_list.append(_token_type_ids.astype(np.int64))
        masked_lm_labels_list.append(labels.astype(np.int64))
    feed_dict = {
        input_ids: input_ids_list,
        position_ids: position_ids_list,
        # token_type_ids: token_type_ids_list,
        masked_lm_labels: masked_lm_labels_list,
    }
    for i in range(args.train_task_num):
        feed_dict[task_batch_idxs[i]] = task_batch_idxs_list[i]
    for step in range(args.profile_steps + args.warmup_steps):
        print(f"step: {step}")
        if args.warmup_steps > 0 and step == args.warmup_steps:
            print("Warmup done! Start to profile...")
        try:
            with ht.autocast(precision):
                with ht.profiler(enabled=True, record_shapes=False) as profiler:
                    # start_time = time.time()
                    results = train_op.graph.run(
                                loss,
                                [loss, train_op],
                                feed_dict=feed_dict,
                                num_micro_batches=micro_batch_num,
                                cur_strategy_id=0,
                                run_level=ht.run_level("update"),
                                grad_scale=1.0)
                # end_time = time.time()
                # print(f"e2e_time = {end_time - start_time:.3f}s")
            print(f"stream time = {float(profiler.summary()['graph_view'][11][1])}")
        except RuntimeError as e:
            print(e)
            os.killpg(0, signal.SIGTERM)
        if _profile_option and local_device.index == 0 and step >= args.warmup_steps:
            # e2e_time.append(end_time - start_time)
            e2e_time.append(float(profiler.summary()['graph_view'][11][1])) # stream_total
            pbar.update(1)
    # record
    if _profile_option and local_device.index == 0:
        print("prepare to record")
        pbar.close()
        record_entry = {
            'dp': args.dp,
            'tp': args.tp,
            'pp': args.pp,
            'sp': args.sp,
            'max_tokens': max_tokens,
            'mbs': mbs_list,
            'seq_len': seq_len_list,
            'num_micro_batches': args.num_micro_batches,
            'e2e_time': np.mean(e2e_time),
            # 'total_stream_time': np.mean(total_stream_time) if len(total_stream_time) > 0 else 0,
            'estimate_time': estimated_time
        }
        print("ready to write")
        write_to_csv(record_entry, args.save_path)

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
        '--dp', type=int, default=1, help='dp degree'
    )
    parser.add_argument(
        '--tp', type=int, default=1, help='tp degree'
    )
    parser.add_argument(
        '--pp', type=int, default=1, help='pp degree'
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
        "--profile_path", type=str, default='', help="Profile path of multi-task training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--save_path", type=str, default='', help="save path of max tokens."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="profile seq length"
    )
    parser.add_argument(
        "--num_micro_batches", type=int, default=16, help="num micro batches"
    )
    parser.add_argument(
        "--profile_steps", type=int, default=100, help="profile steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="warmup steps"
    )
    seq_len_range = [256, 512, 1024, 2048, 4096, 8192, 16384]
    args = parser.parse_args()
    seq_len_range = [seq_len for seq_len in seq_len_range if seq_len <= args.max_tokens]
    distributed_init(args.use_two_node)
    test_cost_model(args, seq_len_range)