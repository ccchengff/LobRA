import os
import argparse
import subprocess
from tqdm import tqdm
from utils import write_to_csv, read_from_csv

def run_profile(
    args,
    tp,
    pp,
    sp,
    seq_len
):
    num_micro_batches = pp * 2
    cmd = f"bash scripts/run_benchmark.sh \
            {args.num_layers} {args.hidden_size} {args.num_attention_heads} {args.train_task_num} \
            {seq_len} 1 {num_micro_batches} \
            1 {tp} {pp} {sp} \
            null {args.trainer_config_path} profile_memory"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return 0
    except Exception as e:
        print(e)
        return -1

def profile_max_tokens(args, seq_len_range):
    # for specific strategy
    if args.num_gpus_limit == -1:
        print("profile max tokens for specific strategy")
        max_tokens = 0
        for seq_len in seq_len_range:
            if run_profile(args, args.tp, args.pp, args.sp, seq_len) == 0:
                max_tokens = seq_len
            else:
                break
        print(f"strategy (tp, pp, sp) = ({args.tp}, {args.pp}, {args.sp}) found max tokens = {max_tokens}")
        max_tokens_entry = {
            'tp': args.tp,
            'pp': args.pp,
            'sp': args.sp,
            'max_tokens': max_tokens
        }
        write_to_csv(max_tokens_entry, args.save_path)
    else:
        cache_dict = None
        if os.path.exists(args.save_path):
            rows = read_from_csv(args.save_path)
            cache_dict = {(row['tp'], row['pp'], row['sp']) : row['max_tokens'] for row in rows}
        # pp_candidates 为 args.num_layers 的因数
        pp_candidates = [i for i in range(1, args.num_layers + 1) if args.num_layers % i == 0]
        tp_candidates = [1, 2, 4, 8]
        sp = args.sp
        pbar = tqdm(total=len(pp_candidates) * len(tp_candidates))
        for pp in pp_candidates:
            for tp in tp_candidates:
                if tp * pp > args.num_gpus_limit or (cache_dict is not None and (tp, pp, sp) in cache_dict.keys()):
                    pbar.update(1)
                    continue
                max_tokens = 0
                for seq_len in seq_len_range:
                    if run_profile(args, tp, pp, sp, seq_len) == 0:
                        max_tokens = seq_len
                    else:
                        break
                print(f"strategy (tp, pp, sp) = ({tp}, {pp}, {sp}) found max tokens = {max_tokens}")
                max_tokens_entry = {
                    'tp': tp,
                    'pp': pp,
                    'sp': sp,
                    'max_tokens': max_tokens
                }
                write_to_csv(max_tokens_entry, args.save_path)
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp", type=int, default=1, help="tp degree"
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="pp degree"
    )
    parser.add_argument(
        "--sp", type=int, default=0, help="sp option"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
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
        "--save_path", type=str, default='', help="save path of max tokens."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--seq_len_range", type=str, default='', help="seq length range"
    )
    parser.add_argument(
        "--num_micro_batches", type=int, default=16, help="num micro batches"
    )
    parser.add_argument(
        "--num_gpus_limit", type=int, default=-1, help="num gpus limit"
    )
    args = parser.parse_args()
    # seq_len_range = list(map(int, args.seq_len_range.split(',')))
    # seq_len_range = [2048, 4096, 8192, 16384]
    seq_len_range = [8192, 16384]
    profile_max_tokens(args, seq_len_range)
