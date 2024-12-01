import os
import csv
import time
import subprocess
from tqdm import tqdm
from scipy.optimize import curve_fit
from utils.logger import read_from_csv, write_to_csv

class CostModel:
    '''
    Profile-based cost model for multi-task lora fine-tuning.
    '''
    def __init__(
        self,
        model_config,
        model_type,
        train_task_num,
        trainer_config_path,
        profile_path,
        max_tokens_path=None,
        sp=1,
    ):
        self.popt = {}
        self.seq_len_range = None
        self.max_tokens = None
        self.tps = [1, 2, 4, 8, 16]
        self.profile_mbs = [1, 4]
        self.profile_n_layer = 2
        self.sp = sp
        self.model_config = model_config
        self.train_task_num = train_task_num
        self.model_type = model_type
        self.trainer_config_path = trainer_config_path
        self.max_tokens_path = max_tokens_path
        self.profile_path = profile_path
        print(f"Building cost model...")
        # memory profile
        if max_tokens_path is not None and os.path.exists(max_tokens_path):
            self._read_from_mem_profile(max_tokens_path)
        else:
            print(f"[WARN] Profile file not found: {max_tokens_path}, it will be used in static planner. Please run `profile_max_tokens.sh` to generate it if needed.")
            self.max_tokens = None
        # time profile
        if os.path.exists(profile_path):
            self._read_from_time_profile(profile_path)
        else:
            print(f"[WARN] Cannot find profile_path: {profile_path}")
            if 'OMPI_COMM_WORLD_SIZE' in os.environ:
                raise ValueError('Cannot profile on multi-gpu environment.')
            else:
                self.seq_len_range = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
                print(f"Profile files not found, profiling with seq_len_range = {self.seq_len_range} and profile_mbs = {self.profile_mbs}...")
                start_time = time.time()
                self.run_benchmark(profile_path)
                end_time = time.time()
                print(f"Profile time: {end_time - start_time:.4f}s")

    def _run_benchmark(self, tp):
        seq_len_range = ",".join(self.seq_len_range)
        profile_mbs = ",".join(self.profile_mbs)
        model_size = ""
        if self.model_config.ffn_hidden_size == 11008:
            model_size = "7B"
        elif self.model_config.ffn_hidden_size == 13824:
            model_size = "13B"
        elif self.model_config.ffn_hidden_size == 28672:
            model_size = "70B"
        cmd = f"mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np {tp} \
                --output-filename logs/cost_model_{self.model_type}_{model_size}/ds_parallel_{tp}_tp{tp}_sp{self.sp} --merge-stderr-to-stdout \
                python3 scripts/cost_model_benchmark.py \
                --trainer_config_path {self.trainer_config_path} \
                --profile_path {self.profile_path} \
                --profile_memory_path {self.max_tokens_path} \
                --vocab_size 30592 \
                --hidden_size {self.model_config.n_embd} \
                --ffn_hidden_size {self.model_config.ffn_hidden_size} \
                --num_attention_heads {self.model_config.n_head} \
                --seq_len_range {seq_len_range} \
                --profile_mbs {profile_mbs} \
                --model_type {self.model_type} \
                --tp {tp} \
                --sp {self.sp} \
                --train_task_num {self.train_task_num} \
                --num_layers {self.profile_n_layer} \
                --lr 1e-4 \
                --profile_steps 100 \
                --warmup_steps 10 \
                --dropout_prob 0 \
                --bf16 \
                --use_flash_attn"
        try:
            subprocess.run(cmd, shell=True, check=True)
            return 0
        except:
            return -1
    
    def run_benchmark(self, profile_path):
        with tqdm(total=len(self.tps)) as pbar:
            for tp in self.tps:
                self._run_benchmark(tp)
                pbar.update(1)
        self._read_from_time_profile(profile_path)
    
    def _read_from_mem_profile(self, profile_path):
        print(f"Read profiled max tokens from {profile_path}")
        max_tokens = {}
        rows = read_from_csv(profile_path)
        for row in rows:
            tp = row['tp']
            pp = row['pp']
            sp = row['sp']
            if sp != self.sp:
                continue
            max_tokens[(tp, pp)] = row['max_tokens']
        self.max_tokens = max_tokens

    def _read_from_time_profile(self, profile_path):
        print(f"Read profiled fw/bw time from {profile_path}")
        mbs_dict = {}
        seq_len_dict = {}
        estimate_time_dict = {}
        self.seq_len_range = set()
        
        for tp in self.tps:
            mbs_dict[tp] = []
            seq_len_dict[tp] = []
            estimate_time_dict[tp] = []

        # 读取单层 Transformer Layer 的时间
        with open(profile_path, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)
            
            for row in reader:
                tp = int(row[0])
                seq_len = int(row[1])
                mbs = int(row[2])
                block_time = float(row[-1])
                mbs_dict[tp].append(mbs)
                seq_len_dict[tp].append(seq_len)
                estimate_time_dict[tp].append(block_time)
                self.seq_len_range.add(seq_len)
        
        for tp in self.tps:
            if len(mbs_dict[tp]) == 0:
                self.popt[tp] = None
            else:
                self.popt[tp], _ = curve_fit(self.curve_fit_func, (mbs_dict[tp], seq_len_dict[tp]), estimate_time_dict[tp])
    
    def curve_fit_func(self, X, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2) * seq_len * seq_len + (c3 * mbs + c4) * seq_len + (c5 * mbs + c6)

    def estimate_time(self, mbs, seq_len, tp, pp, num_micro_batches, num_layers):
        if mbs == 0 or num_micro_batches == 0:
            return 0
        return self.curve_fit_func((mbs, seq_len), *self.popt[tp]) * num_layers * (num_micro_batches + pp - 1) / pp
    
    def get_strategy_candidates(self, num_layers, num_micro_batches=16, throughput_path="throughput_from_cost_model.csv"):
        # TODO: read from existed strategy candidate file
        strategy_candidates = []
        for (tp, pp), max_tokens in self.max_tokens.items():
            num_gpus = tp * pp
            latency = self.estimate_time(1, max_tokens, tp, pp, num_micro_batches, num_layers) / 1000
            strategy_candidate = {
                'tp': tp,
                'pp': pp,
                'max_tokens': max_tokens,
                'latency': latency,
                'throughput_per_gpu': (max_tokens * num_micro_batches) / (latency * num_gpus)
            }
            write_to_csv(strategy_candidate, throughput_path)
            strategy_candidates.append(strategy_candidate)
        return strategy_candidates
