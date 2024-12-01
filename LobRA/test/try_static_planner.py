import os
import numpy as np
import argparse
import bisect
import time
import copy
from trainer.build_strategy_planner import build_strategy_planner_cython
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from trainer.dp_bucket import get_buckets_dp
from trainer.utils.wrapper import DatasetWrapper
from data_utils import GPTJsonDataset, Encoder
from model import LLamaConfig
from profiler import CostModel
from types import SimpleNamespace
from trainer.planner import NewStaticBatchPlanner, GroupedStaticBatchPlanner
from trainer.trainer import TrainerConfig, DatasetContext
from pyscipopt import Model, quicksum

def compute_estimated_time(args):
    optimizer, mbs, seq_len, tp, pp = args
    return optimizer.get_estimated_time(mbs, seq_len, tp, pp)

class FusedStaticBatchPlanner:
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        gpu_num,
        strategy_candidates,
        lp_threads=64
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.global_batch_size_across_tasks = 0
        self.strategy_num = 0
        self.max_tokens_list = []
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.gpu_num = gpu_num
        self.cache_estimate_times = {}
        self.cache_fit_times = {}
        self.aux_fragment = []
        self.aux_complete = []
        self.dps = []
        self.tps = []
        self.pps = []
        self.lp_threads = lp_threads
        self.popt, self.profile_seq_lens = cost_model.popt, cost_model.seq_len_range
        self.strategy_pool = self.get_optimized_strategy_pool(strategy_candidates)
        self.strategy_pool_size = len(self.strategy_pool)
        
        self.multi_task_seq_num_distribution = None
        self.seq_distribution_across_tasks = None
        # tokens
        self.token_num = 0
        self.valid_token_num = 0
    
    def get_optimized_strategy_pool(self, strategy_candidates):
        use_optimized_strategy_pool = True
        goat_strategy_of_max_tokens = {}
        min_gpu_num_strategy_of_max_tokens = {}
        if not use_optimized_strategy_pool:
            strategy_pool = [(strategy['tp'] * strategy['pp'], (strategy['tp'], strategy['pp']), strategy['max_tokens']) for strategy in strategy_candidates]
            return strategy_pool
        for strategy in strategy_candidates:
            tp = strategy['tp']
            pp = strategy['pp']
            num_gpus = tp * pp
            if 'max_tokens' not in strategy.keys():
                assert 'mbs' in strategy.keys() and 'seq_len' in strategy.keys()
                max_tokens = strategy['mbs'] * strategy['seq_len']
            else:
                max_tokens = strategy['max_tokens']
            if max_tokens == 0:
                continue
            throughput = strategy['throughput_per_gpu']
            if max_tokens not in goat_strategy_of_max_tokens.keys():
                goat_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
            elif throughput > goat_strategy_of_max_tokens[max_tokens][0]:
                goat_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
            elif throughput == goat_strategy_of_max_tokens[max_tokens][0] and \
                    num_gpus < goat_strategy_of_max_tokens[max_tokens][1]:
                goat_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
            
            if max_tokens not in min_gpu_num_strategy_of_max_tokens.keys():
                min_gpu_num_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
            elif num_gpus < min_gpu_num_strategy_of_max_tokens[max_tokens][1]:
                min_gpu_num_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
            elif num_gpus == min_gpu_num_strategy_of_max_tokens[max_tokens][1] and \
                    throughput > min_gpu_num_strategy_of_max_tokens[max_tokens][0]:
                min_gpu_num_strategy_of_max_tokens[max_tokens] = (throughput, num_gpus, (tp, pp))
        strategy_pool = set([(goat_strategy[1], goat_strategy[2], max_tokens) for max_tokens, goat_strategy in goat_strategy_of_max_tokens.items()])
        strategy_pool = list(strategy_pool.union(set([(min_gpu_num_strategy[1], min_gpu_num_strategy[2], max_tokens) for max_tokens, min_gpu_num_strategy in min_gpu_num_strategy_of_max_tokens.items()])))
        print(f"strategy_pool = {strategy_pool}")
        return strategy_pool
    
    def fit_time(self, X, aux_fragment, tp, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + (c3 * mbs + c4 * aux_fragment) * seq_len + (c5 * mbs + c6 * aux_fragment)
    
    def estimate_time(self, mbs, s, tp, pp, aux_fragment=1):
        return self.fit_time((mbs, s), aux_fragment, tp, *self.popt[tp]) * self.num_layers / pp

    def estimate_total_time(self, strategy_idx, m, r, aux_bool_fragment, max_batch_time):
        tp = self.tps[strategy_idx]
        pp = self.pps[strategy_idx]
        # return quicksum(self.estimate_time(self.mbs_map[i][strategy_idx], seq_len, tp, pp) * m[i][strategy_idx] + \
        # return quicksum(self.get_estimated_time(self.mbs_map[i][strategy_idx], seq_len, tp, pp) * m[i][strategy_idx] + \
        return quicksum(self.cache_estimate_times.get((self.mbs_map[i][strategy_idx], seq_len, tp, pp), 0) * m[i][strategy_idx] + \
                        self.estimate_time(r[i][strategy_idx], seq_len, tp, pp, aux_bool_fragment[i][strategy_idx]) \
                        for i, seq_len in enumerate(self.task_seq_lens)) + (pp - 1) * max_batch_time
    
    def get_estimated_time(self, mbs, s, tp, pp):
        if mbs == 0:
            return 0
        return self.fit_time((mbs, s), 1, tp, *self.popt[tp]) * self.num_layers / pp
    
    def print_estimate_total_time(self, tp, pp, multi_task_batch_dispatch_map, strategy_id):
        estimate_time = 0
        max_tokens = self.max_tokens_list[strategy_id]
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        for task in range(self.train_task_num):
            print(f"task {task}: {multi_task_batch_dispatch_map[task][strategy_id]}")
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        seq_len_map = {seq_len : np.sum([multi_task_batch_dispatch_map[task][strategy_id][seq_len] for task in range(self.train_task_num)]) \
                       for seq_len in self.task_seq_lens}
        estimate_time = 0
        max_batch_time = 0
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            if mbs == 0:
                m = 0
            else:
                m = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] // mbs
            rest_sample_num = seq_len_map[seq_len] // self.dps[strategy_id] - mbs * m
            full_time = self.get_estimated_time(mbs, seq_len, tp, pp)
            piece_time = self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            if m > 0:
                print(f"|---(mbs, seq_len, m) = ({mbs}, {seq_len}, {m}): {full_time * m / 1000}s")
            if rest_sample_num > 0:
                print(f"|---(mbs, seq_len, m) = ({rest_sample_num}, {seq_len}, 1): {piece_time / 1000}s")
            estimate_time += self.get_estimated_time(mbs, seq_len, tp, pp) * m + self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            cur_max_time = full_time if m > 0 else piece_time
            max_batch_time = np.max([max_batch_time, cur_max_time])
        estimate_time += (pp - 1) * max_batch_time
        print(f"strategy - {strategy_id}: max_batch_time = {max_batch_time / 1000}s, total_estimate_time = {estimate_time / 1000} s")
        return estimate_time
    
    def get_estimate_total_time(self, tp, pp, multi_task_batch_dispatch_map, strategy_id):
        # print(f"multi = {multi_task_batch_dispatch_map}")
        max_tokens = self.max_tokens_list[strategy_id]
        seq_len_map = {seq_len : np.sum([multi_task_batch_dispatch_map[task][strategy_id][seq_len] for task in range(self.train_task_num)]) \
                       for seq_len in self.task_seq_lens}
        estimate_time = 0
        max_batch_time = 0
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            if mbs == 0:
                m = 0
            else:
                # m = seq_len_map[seq_len] // self.dps[strategy_id] // mbs
                m = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] // mbs
            rest_sample_num = seq_len_map[seq_len] // self.dps[strategy_id] - mbs * m
            # print(f"mbs = {mbs}, seq_len = {seq_len}, tp = {tp}, pp = {pp}, m = {m}, rest = {rest_sample_num}")
            full_time = self.get_estimated_time(mbs, seq_len, tp, pp)
            piece_time = self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            estimate_time += full_time * m + piece_time
            cur_max_time = full_time if m > 0 else piece_time
            max_batch_time = max(max_batch_time, cur_max_time)
        estimate_time += (pp - 1) * max_batch_time
        return estimate_time
    
    def build_strategy_planner(self):
        target = self.gpu_num
        n = self.strategy_pool_size
        dp = [0] * (target + 1)
        dp[0] = 1
        
        paths = [[] for _ in range(target + 1)]
        paths[0] = [[]]
        
        for i in range(1, n + 1):
            strategy_gpu = self.strategy_pool[i - 1][0]
            for j in range(strategy_gpu, target + 1):
                if dp[j - strategy_gpu] > 0:
                    dp[j] += dp[j - strategy_gpu]
                    paths[j].extend([path + [i - 1] for path in paths[j - strategy_gpu]])
        return dp[target], paths[target]
    
    def build_planner(self, seq_distribution):
        # get cumulative_seq_distribution
        # cumulative_seq_distribution = seq_distribution.copy()
        # cumulative_seq_num = 0
        # for seq_len in self.task_seq_lens:
        #     cumulative_seq_num += int(seq_distribution[seq_len])
        #     cumulative_seq_distribution[seq_len] = cumulative_seq_num

        model = Model("dynamic_batch_planner")
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, strategy%s)" % (seq_len, j)) \
                             for j in range(self.strategy_num)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, strategy%s)" % (seq_len, j)) \
                             for j in range(self.strategy_num)] for seq_len in self.task_seq_lens] 
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time(strategy%s)" % i) if self.pps[i] > 1 else 0 \
                          for i in range(self.strategy_num)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(strategy%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.pps[i] > 1 else [] for i in range(self.strategy_num)]

        # model.addCons(quicksum(quicksum(self.dps[j] * n[k][j] for j in range(self.strategy_num)) for k, _ in enumerate(self.task_seq_lens)) == cumulative_seq_num, name="eq_seq_num_sum")
        # for i, seq_len in enumerate(self.task_seq_lens):
        #     model.addCons(quicksum(quicksum(self.dps[j] * n[k][j] for j in range(self.strategy_num)) for k in range(i + 1)) <= cumulative_seq_distribution[seq_len], name="dispatch_seq_num_le_cdf_%s" % seq_len)
        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(n[i][j] for j in range(self.strategy_num)) == seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            # model.addCons(quicksum(n[i][j] for j in range(self.strategy_num)) <= seq_distribution[seq_len] + np.sum([self.dps[k] - 1 for k in range(self.strategy_num)]), name="le_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        for j in range(self.strategy_num):
            model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(strategy%s)" % j)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.strategy_num):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                # auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, strategy%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, strategy%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, strategy%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, strategy%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, strategy%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, strategy%s)" % (seq_len, j))
        # max batch time
        for i in range(self.strategy_num):
            if self.pps[i] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(max_batch_time[i] >= self.cache_estimate_times.get((self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]), 0) * aux_bool_complete[j][i], name="max_batch_time_ge_complete(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.cache_estimate_times.get((self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]), 0) * aux_bool_complete[j][i] + self.max_batch_time_list[i] * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                # fragment
                model.addCons(max_batch_time[i] >= self.estimate_time(r[j][i], seq_len, self.tps[i], self.pps[i], aux_bool_fragment[j][i]), name="max_batch_time_ge_fragment(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(r[j][i], seq_len, self.tps[i], self.pps[i], aux_bool_fragment[j][i]) + self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]), name="max_batch_time_le_fragment_plus_bound(strategy%s, %s)" % (i, j))
        # 设置目标函数
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for j in range(self.strategy_num):
            model.addCons(objvar >= self.estimate_total_time(j, m, r, aux_bool_fragment, max_batch_time[j]))
        return model
    
    def schedule(self, multi_task_seq_distribution):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        # assert task_seq_len_range.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        max_seq_len = max(self.task_seq_lens)
        max_seq_len_to_2 = 2 ** int(np.ceil(np.log2(max_seq_len)))
        
        combine_num, combine_dps = self.build_strategy_planner()
        print(f"max_seq_len = {max_seq_len}, max_seq_len_to_2 = {max_seq_len_to_2}")
        print(f"combine_num = {combine_num}")
        strategy_combination = []
        for i in range(len(combine_dps)):
            strategy_combination.append({})
            for j in range(len(combine_dps[i])):
                if combine_dps[i][j] not in strategy_combination[i].keys():
                    strategy_combination[i][combine_dps[i][j]] = combine_dps[i].count(combine_dps[i][j])
            max_tokens = 0
            for sid in strategy_combination[i].keys():
                max_tokens = max(max_tokens, self.strategy_pool[sid][2])
            if max_tokens != max_seq_len_to_2:
                strategy_combination[i] = {}
        # 删除空的strategy_combination
        strategy_combination = [strategy for strategy in strategy_combination if len(strategy) > 0]
        print(f"strategy_combination = {len(strategy_combination)}")
        
    	# for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        global_batch_size_across_tasks = 0
        multi_task_seq_num_distribution = {task_id: {seq_len : 0 for seq_len in self.task_seq_lens} for task_id in range(self.train_task_num)}
        max_seq_len = max(self.task_seq_lens)
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            # print(f"task {i}: {task_seq_distribution}")
            for seq_len, p in task_seq_distribution.items():
                # seq_num = math.ceil(p * self.global_batch_size_list[i])
                # seq_num = math.ceil(seq_num) if seq_num < 1 and seq_num > 0 else round(seq_num)
                seq_num = round(p * self.global_batch_size_list[i])
                multi_task_seq_num_distribution[i][seq_len] = seq_num
                seq_distribution_across_tasks[seq_len] += seq_num
                global_batch_size_across_tasks += seq_num
        seq_distribution_across_tasks = {s : num for s, num in seq_distribution_across_tasks.items()}
        if seq_distribution_across_tasks[max_seq_len] == 0:
            seq_distribution_across_tasks[max_seq_len] = 1
            global_batch_size_across_tasks += 1
            for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
                if max_seq_len in task_seq_distribution.keys():
                    # print(f"{i}: {task_seq_distribution[max_seq_len]}")
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break
        # print(f"multi_task_seq_num_distribution = {multi_task_seq_num_distribution}")
        cost_list = []
        # strategy_combination.reverse()
        # strategy_combination = strategy_combination[237:]
        s_time = time.time()
        for idx, strategy in enumerate(strategy_combination):
            multi_task_seq_num_distribution_copy = copy.deepcopy(multi_task_seq_num_distribution)
            print(f"strategy: {idx}")
            strategy_idxs = strategy.keys()
            max_tokens_list = [self.strategy_pool[i][2] for i in strategy_idxs]
            tps_list = [self.strategy_pool[i][1][0] for i in strategy_idxs]
            pps_list = [self.strategy_pool[i][1][1] for i in strategy_idxs]
            dps_list = [strategy[i] for i in strategy_idxs]
            self.max_tokens_list = max_tokens_list
            self.tps = tps_list
            self.pps = pps_list
            self.dps = dps_list
            self.strategy_num = len(strategy_idxs)
            self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
            self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                        for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_num)]
            self.cache_estimate_times = {(mbs, seq_len, tp, pp): self.fit_time((mbs, seq_len), 1, tp, *self.popt[tp]) * self.num_layers / pp
                                        for mbs in set(np.array(self.mbs_map).flatten()) if mbs > 0
                                        for seq_len in self.task_seq_lens
                                        for tp, pp in zip(self.tps, self.pps)}
            # print(f"Dynamic planner start to dispatch batch for running tasks...")
            # print(f"seq distribution across tasks: {seq_distribution_across_tasks}")
            start_time = time.time()
            model = self.build_planner(seq_distribution_across_tasks)
            # Set model param
            model.setIntParam("lp/threads", self.lp_threads)
            model.setIntParam("parallel/maxnthreads", self.lp_threads)
            # model.setIntParam("limits/bestsol", 5)
            # model.setLongintParam("limits/nodes", 100000)
            # model.setLongintParam("limits/stallnodes", 500)
            # model.setRealParam("limits/gap", 1e-3)

            # model.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF) 
            # model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.FAST)
            model.hideOutput()
            model.optimize()
            try:
                model.getObjVal()
            except:
                print("No solution found")
                end_time = time.time()
                cost_list.append(1e8)
            print(f"model status = {model.getStatus()}, sol num = {model.getNSols()}")
            end_time = time.time()
            # print(f"Dynamic batch planner takes {end_time - start_time:.4f}s to dispatch running batch")
            # get the dispatch result for all strategies
            seq_len_num_map_list = []
            for i in range(self.strategy_num):
                seq_len_num_map = {s : 0 for s in self.task_seq_lens}
                for v in model.getVars():
                    for seq_len in self.task_seq_lens:
                        n_name = "seq_num(%s, strategy%s)" % (seq_len, i)
                        if n_name in v.name:
                            n_val = round(model.getVal(v))
                            if n_val < 1:
                                continue
                            # seq_len_num_map[seq_len] = n_val * self.dps[i]
                            seq_len_num_map[seq_len] = n_val
                seq_len_num_map_list.append(seq_len_num_map)
            # print(f"[DEBUG] seq_len_num_map_list = {seq_len_num_map_list}")
            # tune dispatched seq num
            for seq_len in self.task_seq_lens:
                total_seq_num = seq_distribution_across_tasks[seq_len]
                dispatched_seq_num = 0
                for i in range(self.strategy_num):
                    dispatched_seq_num += seq_len_num_map_list[i][seq_len]
                if dispatched_seq_num > total_seq_num:
                    print(f"seq_len {seq_len} with {dispatched_seq_num} v.s. {total_seq_num}")
                    dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] > 1 and seq_len_num_map_list[i][seq_len] > 0],
                                        key=lambda x: self.dps[x], reverse=True)
                    overflow_seq_num = dispatched_seq_num - total_seq_num
                    max_overflow_seq_num = np.sum([self.dps[i] - 1 for i in dp_strategy])
                    if overflow_seq_num > max_overflow_seq_num:
                        total_overflow_num = overflow_seq_num - max_overflow_seq_num
                        non_dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] == 1 and seq_len_num_map_list[i][seq_len] > 0],
                                                key=lambda x: seq_len_num_map_list[x][seq_len], reverse=True)
                        # avg_overflow_num = total_overflow_num // len(non_dp_strategy)
                        for i in non_dp_strategy:
                            if total_overflow_num == 0:
                                break
                            tune_num = min(seq_len_num_map_list[i][seq_len], total_overflow_num)
                            seq_len_num_map_list[i][seq_len] -= tune_num
                            total_overflow_num -= tune_num
                        # if total_overflow_num > 0:
                        #     for i in non_dp_strategy:
                        #         if total_overflow_num == 0:
                        #             break
                        #         tune_num = min(seq_len_num_map_list[i][seq_len], total_overflow_num)
                        #         seq_len_num_map_list[i][seq_len] -= tune_num
                        #         total_overflow_num -= tune_num
                        assert total_overflow_num == 0
                        overflow_seq_num = max_overflow_seq_num
                    # assert overflow_seq_num <= max_overflow_seq_num, \
                    #     f"overflow seq num is larger than the limit for seq_len = {seq_len} where limit = {max_overflow_seq_num}"
                    for i in dp_strategy:
                        if overflow_seq_num == 0:
                            break
                        seq_len_num_map_list[i][seq_len] -= min(self.dps[i] - 1, overflow_seq_num)
                        overflow_seq_num -= min(self.dps[i] - 1, overflow_seq_num)
                    assert overflow_seq_num == 0
            # print(f"[DEBUG] after tune: seq_len_num_map_list = {seq_len_num_map_list}")
            # print(f"multi_task_seq_num_distribution = {multi_task_seq_num_distribution_copy}")

            # dispatch task-specific samples
            multi_task_batch_dispatch_map = {task_id : [{s : 0 for s in self.task_seq_lens} for _ in range(self.strategy_num)] for task_id in range(self.train_task_num)}
            for seq_idx, seq_len in enumerate(self.task_seq_lens):
                for task_id in range(self.train_task_num):
                    for i in range(self.strategy_num):
                        dispatch_num = min(seq_len_num_map_list[i].get(seq_len, 0), multi_task_seq_num_distribution_copy[task_id].get(seq_len, 0))
                        if dispatch_num == 0:
                            continue
                        multi_task_batch_dispatch_map[task_id][i][seq_len] += int(dispatch_num)
                        seq_len_num_map_list[i][seq_len] -= dispatch_num
                        multi_task_seq_num_distribution_copy[task_id][seq_len] -= dispatch_num
                    if multi_task_seq_num_distribution_copy[task_id].get(seq_len, 0) > 0:
                        assert seq_idx < len(self.task_seq_lens) - 1
                        if self.task_seq_lens[seq_idx + 1] not in multi_task_seq_num_distribution_copy[task_id].keys():
                            multi_task_seq_num_distribution_copy[task_id][self.task_seq_lens[seq_idx + 1]] = 0
                        multi_task_seq_num_distribution_copy[task_id][self.task_seq_lens[seq_idx + 1]] += multi_task_seq_num_distribution_copy[task_id][seq_len]
            # print(f"multi_task_batch_dispatch_map = {multi_task_batch_dispatch_map}")
            
            # profile
            cost_time = 0
            # print(f"strategy num = {self.strategy_num}")
            for i in range(self.strategy_num):
                cost_time = max(cost_time, self.get_estimate_total_time(self.tps[i], self.pps[i], multi_task_batch_dispatch_map, i))
            dps, tps, pps, max_tokens_list = [], [], [], []
            for strategy_idx in strategy.keys():
                dp = strategy[strategy_idx]
                tp, pp = self.strategy_pool[strategy_idx][1]
                max_tokens = self.strategy_pool[strategy_idx][2]
                dps.append(dp)
                tps.append(tp)
                pps.append(pp)
                max_tokens_list.append(max_tokens)
            print(f"{dps}, {tps}, {pps}, {max_tokens_list} : {cost_time}")
            cost_list.append(cost_time)
        e_time = time.time()
        print(f"search time = {(e_time - s_time):.3f}s")
        # 对cost_list排序，找到前k个最小cost对应的策略组合
        cost_idxs = np.argsort(cost_list)
        k = 5
        for i in range(k):
            print(f"====================")
            print(f"cost - {i}: {cost_list[cost_idxs[i]]}")
            min_cost_idx = cost_idxs[i]
            multi_strategy = strategy_combination[min_cost_idx]
            for strategy_idx in multi_strategy.keys():
                strategy = self.strategy_pool[strategy_idx]
                dp = multi_strategy[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                print(f"strategy - {strategy_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            print(f"====================")
        
        # 找到最小cost的策略组合
        # min_cost_idx = np.argmin(cost_list)
        # multi_strategy = strategy_combination[min_cost_idx]
        # for strategy_idx in multi_strategy.keys():
        #     strategy = self.strategy_pool[strategy_idx]
        #     dp = multi_strategy[strategy_idx]
        #     tp, pp = strategy[1]
        #     max_tokens = strategy[2]
        #     print(f"strategy - {strategy_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            
        return multi_strategy, end_time - start_time

    def parallel_schedule(self, multi_task_seq_distribution):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        # assert task_seq_len_range.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        # for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        global_batch_size_across_tasks = 0
        multi_task_seq_num_distribution = {task_id: {seq_len : 0 for seq_len in self.task_seq_lens} for task_id in range(self.train_task_num)}
        max_seq_len = max(self.task_seq_lens)
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            # print(f"task {i}: {task_seq_distribution}")
            for seq_len, p in task_seq_distribution.items():
                # seq_num = math.ceil(p * self.global_batch_size_list[i])
                # seq_num = math.ceil(seq_num) if seq_num < 1 and seq_num > 0 else round(seq_num)
                seq_num = round(p * self.global_batch_size_list[i])
                multi_task_seq_num_distribution[i][seq_len] = seq_num
                seq_distribution_across_tasks[seq_len] += seq_num
                global_batch_size_across_tasks += seq_num
        seq_distribution_across_tasks = {s : num for s, num in seq_distribution_across_tasks.items()}
        if seq_distribution_across_tasks[max_seq_len] == 0:
            seq_distribution_across_tasks[max_seq_len] = 1
            global_batch_size_across_tasks += 1
            for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
                if max_seq_len in task_seq_distribution.keys():
                    # print(f"{i}: {task_seq_distribution[max_seq_len]}")
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break
        self.multi_task_seq_num_distribution = multi_task_seq_num_distribution
        # task_seq_len_range = set()
        # for i in range(self.train_task_num):
        #     task_seq_len_range = task_seq_len_range.union(set(self.multi_task_seq_num_distribution[i].keys()))
        # self.task_seq_lens = sorted(list(task_seq_len_range))
        max_seq_len = max(self.task_seq_lens)
        max_seq_len_to_2 = 2 ** int(np.ceil(np.log2(max_seq_len)))
        print(f"max_seq_len = {max_seq_len}, max_seq_len_to_2 = {max_seq_len_to_2}")
        
        print("build strategy planner begin...")
        s_time = time.time()
        # ss_time = time.time()
        # combine_num, combine_dps = self.build_strategy_planner()
        gpu_num_of_strategy_pool = np.array([strategy[0] for strategy in self.strategy_pool], dtype=np.int32)
        combine_num, combine_dps = build_strategy_planner_cython(self.gpu_num, self.strategy_pool_size, gpu_num_of_strategy_pool)
        # ee_time = time.time()
        # print(f"build strategy planner takes {ee_time - ss_time:.2f}s", flush=True)
        # print(f"combine_num = {combine_num}")
        strategy_combination = []
        for i in range(len(combine_dps)):
            strategy_combination.append({})
            for j in range(len(combine_dps[i])):
                if combine_dps[i][j] not in strategy_combination[i].keys():
                    strategy_combination[i][combine_dps[i][j]] = combine_dps[i].count(combine_dps[i][j])
            # if len(strategy_combination[i].keys()) > 1:
            #     strategy_combination[i] = {}
            #     continue
            max_tokens = 0
            for sid in strategy_combination[i].keys():
                max_tokens = max(max_tokens, self.strategy_pool[sid][2])
            if max_tokens != max_seq_len_to_2:
                strategy_combination[i] = {}
        # 删除空的strategy_combination
        strategy_combination = [strategy for strategy in strategy_combination if len(strategy) > 0]
        # print(f"strategy_combination = {len(strategy_combination)}")
        # '''
        def get_lower_bound_from_group_planner(strategy):
            # 策略配置
            strategy_idxs = strategy.keys()
            max_tokens_list = [self.strategy_pool[i][2] for i in strategy_idxs]
            tps_list = [self.strategy_pool[i][1][0] for i in strategy_idxs]
            pps_list = [self.strategy_pool[i][1][1] for i in strategy_idxs]
            dps_list = [strategy[i] for i in strategy_idxs]
            self.max_tokens_list = max_tokens_list
            self.dps = dps_list
            # sorted_max_tokens_idxs = np.argsort(max_tokens_list)
            # seq2strategy_1 = {s: sorted_max_tokens_idxs[np.searchsorted(max_tokens_list, s, side='left')] 
            #                   for s in self.task_seq_lens}
            # seq2strategy = {}
            # strategy_idx = 0
            # for s in self.task_seq_lens:
            #     while s > max_tokens_list[sorted_max_tokens_idxs[strategy_idx]]:
            #         strategy_idx += 1
            #     seq2strategy[s] = sorted_max_tokens_idxs[strategy_idx]
            # print(f"seq2strategy = {seq2strategy}")
            sorted_max_tokens_idxs = np.argsort(max_tokens_list)
            sorted_max_tokens_list = sorted(max_tokens_list)
            seq2strategy = {}
            for s in self.task_seq_lens:
                strategy_idx = np.searchsorted(sorted_max_tokens_list, s, side='left')
                seq2strategy[s] = sorted_max_tokens_idxs[strategy_idx]
            
            # print(f"seq2strategy = {seq2strategy}")
            multi_task_batch_dispatch_map = {task_id : [{s : self.multi_task_seq_num_distribution[task_id][s] if seq2strategy[s] == i else 0 for s in self.task_seq_lens} for i in range(len(strategy_idxs))] for task_id in range(self.train_task_num)}
            # print(f"multi_task_batch_dispatch_map = {multi_task_batch_dispatch_map}")
            cu_estimate_time = 0
            gpu_num = 0
            for strategy_idx in range(len(strategy_idxs)):
                estimate_time = self.get_estimate_total_time(tps_list[strategy_idx], pps_list[strategy_idx], multi_task_batch_dispatch_map, strategy_idx)
                cu_estimate_time += estimate_time * tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
                gpu_num += tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
            return cu_estimate_time / gpu_num / 1000
        # '''
        estimate_cost = [None] * len(strategy_combination)
        min_candidate_num = 8
        # ss_time = time.time()
        # estimate_cost = Parallel(n_jobs=min(max(1, len(strategy_combination) // 32), os.cpu_count()), prefer="processes", backend="multiprocessing")(
        #     delayed(get_lower_bound_from_group_planner)(strategy) for strategy in strategy_combination
        # )
        estimate_cost = []
        no_repetitive_strategy_combination_cost = []
        for strategy in strategy_combination:
            strategy_cost = get_lower_bound_from_group_planner(strategy)
            if len(set([self.strategy_pool[i][2] for i in strategy.keys()])) == len(strategy.keys()):
                no_repetitive_strategy_combination_cost.append(strategy_cost)
            estimate_cost.append(strategy_cost)
        # ee_time = time.time()
        # print(f"estimate cost takes {ee_time - ss_time:.2f}s")
        # estimate_cost = Parallel(n_jobs=1, prefer="processes")(
        #     delayed(get_lower_bound_from_group_planner)(strategy) for strategy in strategy_combination
        # )

        # for i, strategy in enumerate(strategy_combination):
        #     estimate_cost[i] = self.get_lower_bound_from_group_planner(i, strategy)[1]
        
        # e_time = time.time()
        # print(f"estimate time = {e_time - s_time:.3f}s")
        # min_cost = min(estimate_cost)
        min_cost = min(no_repetitive_strategy_combination_cost)
        pruned_strategy_combination = []
        for cost, strategy in zip(estimate_cost, strategy_combination):
            # if abs(cost - min_cost) > 0.1 * min_cost and len(strategy.keys()) > 1:
            if cost - min_cost > 0.15 * min_cost:
                continue
            pruned_strategy_combination.append(strategy)
        # print(f"before: {len(estimate_cost)}, after: {len(pruned_strategy_combination)}")
        if len(pruned_strategy_combination) < min_candidate_num:
            sorted_estimate_cost_idx = np.argsort(estimate_cost)
            sorted_estimate_cost_idx = sorted_estimate_cost_idx[:min_candidate_num]
            pruned_strategy_combination = [strategy_combination[i] for i in sorted_estimate_cost_idx]
            # print(f"tune to min candidate num: {len(pruned_strategy_combination)}")
            
        
        # print(f"multi_task_seq_num_distribution = {multi_task_seq_num_distribution}")
        cost_list = []
        # s_time = time.time()
        def parallel_strategy_solve(strategy):
            # multi_task_seq_num_distribution_copy = copy.deepcopy(multi_task_seq_num_distribution)
            strategy_idxs = strategy.keys()
            max_tokens_list = [self.strategy_pool[i][2] for i in strategy_idxs]
            tps_list = [self.strategy_pool[i][1][0] for i in strategy_idxs]
            pps_list = [self.strategy_pool[i][1][1] for i in strategy_idxs]
            dps_list = [strategy[i] for i in strategy_idxs]
            self.max_tokens_list = max_tokens_list
            self.tps = tps_list
            self.pps = pps_list
            self.dps = dps_list
            self.strategy_num = len(strategy_idxs)
            self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
            self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                        for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_num)]
            self.cache_estimate_times = {(mbs, seq_len, tp, pp): self.fit_time((mbs, seq_len), 1, tp, *self.popt[tp]) * self.num_layers / pp
                                        for mbs in set(np.array(self.mbs_map).flatten()) if mbs > 0
                                        for seq_len in self.task_seq_lens
                                        for tp, pp in zip(self.tps, self.pps)}
            model = self.build_planner(seq_distribution_across_tasks)
            # Set model param
            # model.setIntParam("lp/threads", self.lp_threads)
            # model.setIntParam("parallel/maxnthreads", self.lp_threads)
            # model.setIntParam("lp/threads", 64)
            # model.setIntParam("parallel/maxnthreads", 64)
            # model.setLongintParam("limits/nodes", 100000)
            model.setLongintParam("limits/stallnodes", 5000)
            model.setRealParam("limits/gap", 5e-3)
            model.hideOutput()
            model.optimize()
            try:
                cost_time = model.getObjVal() / 1000
            except:
                print("No solution found")
                cost_list.append(1e8)
                return 1e8, None
            model_status = model.getStatus()
            dps, tps, pps, max_tokens_list = [], [], [], []
            for strategy_idx in strategy.keys():
                dp = strategy[strategy_idx]
                tp, pp = self.strategy_pool[strategy_idx][1]
                max_tokens = self.strategy_pool[strategy_idx][2]
                dps.append(dp)
                tps.append(tp)
                pps.append(pp)
                max_tokens_list.append(max_tokens)
            # print(f"{dps}, {tps}, {pps}, {max_tokens_list} : {cost_time}")
            return cost_time, strategy, model_status
        # results = Parallel(n_jobs=1, prefer="processes")(
        results = Parallel(n_jobs=min(max(1, len(pruned_strategy_combination) // 64), os.cpu_count()), prefer="processes")(
            delayed(parallel_strategy_solve)(strategy) for strategy in pruned_strategy_combination
        )
        e_time = time.time()
        print(f"search time = {(e_time - s_time):.3f}s")
        cost_idxs = np.argsort([result[0] for result in results])
        for cost_idx in cost_idxs:
            multi_strategy = results[cost_idx][1]
            if len(multi_strategy.keys()) > 1:
                continue
            _, lower_bound_of_best_strategy = self.get_lower_bound_from_group_planner(i, multi_strategy)
            print(f"====================")
            # print(f"cost - {i}: {cost_list[cost_idxs[i]]}")
            print(f"cost - single: {results[cost_idx][0]}, lower_bound = {lower_bound_of_best_strategy}, status = {results[cost_idx][2]}")
            for strategy_idx in multi_strategy.keys():
                strategy = self.strategy_pool[strategy_idx]
                dp = multi_strategy[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                print(f"strategy - {strategy_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            print(f"====================")
        '''
        # 对cost_list排序，找到前k个最小cost对应的策略组合
        k = min(5, len(results))
        for i in range(k):
            min_cost_idx = cost_idxs[i]
            multi_strategy = results[min_cost_idx][1]
            if multi_strategy is None:
                continue
            _, lower_bound_of_best_strategy = self.get_lower_bound_from_group_planner(i, multi_strategy)
            print(f"====================")
            # print(f"cost - {i}: {cost_list[cost_idxs[i]]}")
            print(f"cost - {i}: {results[min_cost_idx][0]}, lower_bound = {lower_bound_of_best_strategy}, status = {results[min_cost_idx][2]}")
            for strategy_idx in multi_strategy.keys():
                strategy = self.strategy_pool[strategy_idx]
                dp = multi_strategy[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                print(f"strategy - {strategy_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            print(f"====================")
        '''

        min_cost_idx = cost_idxs[0]
        multi_strategy = results[min_cost_idx][1]
        if os.environ.get("BUCKET_PLAN") == "PROFILE":
            with open('effectiveness_static.txt', 'a') as f:
                for strategy_idx in multi_strategy.keys():
                    strategy = self.strategy_pool[strategy_idx]
                    dp = multi_strategy[strategy_idx]
                    tp, pp = strategy[1]
                    max_tokens = strategy[2]
                    f.write(f"dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}\n")
                f.write(f"{results[min_cost_idx][0]}\n")
                f.write(f"{e_time - s_time:.3f}\n")
                f.write("\n")
        return multi_strategy, e_time - s_time

    def get_lower_bound_from_group_planner(self, strategy_id, strategy):
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(self.multi_task_seq_num_distribution[i].keys()))
        # assert task_seq_len_range.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        # 策略配置
        strategy_idxs = strategy.keys()
        max_tokens_list = [self.strategy_pool[i][2] for i in strategy_idxs]
        tps_list = [self.strategy_pool[i][1][0] for i in strategy_idxs]
        pps_list = [self.strategy_pool[i][1][1] for i in strategy_idxs]
        dps_list = [strategy[i] for i in strategy_idxs]
        self.max_tokens_list = max_tokens_list
        self.dps = dps_list
        sorted_max_tokens_idxs = np.argsort(max_tokens_list)
        seq2strategy = {}
        strategy_idx = 0
        for s in self.task_seq_lens:
            while s > max_tokens_list[sorted_max_tokens_idxs[strategy_idx]]:
                strategy_idx += 1
            seq2strategy[s] = sorted_max_tokens_idxs[strategy_idx]
        # print(f"seq2strategy = {seq2strategy}")
        multi_task_batch_dispatch_map = {task_id : [{s : self.multi_task_seq_num_distribution[task_id][s] if seq2strategy[s] == i else 0 for s in self.task_seq_lens} for i in range(len(strategy_idxs))] for task_id in range(self.train_task_num)}
        # print(f"multi_task_batch_dispatch_map = {multi_task_batch_dispatch_map}")
        cu_estimate_time = 0
        gpu_num = 0
        for strategy_idx in range(len(strategy_idxs)):
            estimate_time = self.get_estimate_total_time(tps_list[strategy_idx], pps_list[strategy_idx], multi_task_batch_dispatch_map, strategy_idx)
            cu_estimate_time += estimate_time * tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
            gpu_num += tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
        return strategy_id, cu_estimate_time / gpu_num / 1000
    
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
                                                     global_batch_size_list, args.num_gpus, strategy_candidates)
    elif data_dispatch_pattern == 'TEST':
        static_batch_planner = FusedStaticBatchPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                       global_batch_size_list, args.num_gpus, strategy_candidates)
    else:
        print(f"Invalid data dispatch pattern: {data_dispatch_pattern}")
        return
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
            gbs_num = 1e6
            for i in range(trainer_config.train_task_num):
                gbs_num = min(gbs_num, len(dataset_ctxs[i].dataset) // static_batch_planner.global_batch_size_list[i])
            # print(f"gbs_num = {gbs_num}")

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
            # print(f"dp_buckets = {dp_buckets}")
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                # print(f"{i}: {fine_grained_seq_len_distribution}")
                seq_len_distribution_list.append(fine_grained_seq_len_distribution)
        elif os.environ.get("BUCKET_PLAN") == "PROFILE":
            pass
        elif os.environ.get("BUCKET_PLAN") == "STATIC":
            if bucket_limit == 7:
                dp_buckets = [256, 512, 1024, 2048, 4096, 8192, 16384] # 7 bucket
            elif bucket_limit == 16:
                dp_buckets = [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 12288, 16384] # 16 buckets
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(seq_len_distribution)
        else:
            print("Invalid bucket plan")
            return

    if os.environ.get("BUCKET_PLAN") == "PROFILE":
        with open("effectiveness.txt", 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                seq_len_distribution_list = []
                seq_len_distribution = eval(lines[i])
                for v in seq_len_distribution.values():
                    seq_len_distribution_list.append(v)
                with open('effectiveness_static.txt', 'a') as ff:
                    ff.write(f'step: {i // 4}\n')
                static_batch_planner.parallel_schedule(seq_len_distribution_list)
    else:
        static_batch_planner.parallel_schedule(seq_len_distribution_list)

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