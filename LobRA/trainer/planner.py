import re
import math
import os
import time
import numpy as np
import pyscipopt
from pyscipopt import Model, quicksum

class GroupedStaticBatchPlanner:
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        gpu_num,
        strategy_candidates,
        lp_threads=32
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.gpu_num = gpu_num
        self.lp_threads = lp_threads
        self.cost_model = cost_model
        self.M_bound = 1e6
        self.popt, self.profile_seq_lens = cost_model.popt, cost_model.seq_len_range
        self.strategy_pool = sorted(self.get_optimized_strategy_pool(strategy_candidates), key=lambda strategy: strategy[2])
        self.strategy_pool_size = len(self.strategy_pool)

    def get_optimized_strategy_pool(self, strategy_candidates):
        goat_strategy_of_max_tokens = {}
        min_gpu_num_strategy_of_max_tokens = {}
        for strategy in strategy_candidates:
            tp = strategy['tp']
            pp = strategy['pp']
            num_gpus = tp * pp
            if 'max_tokens' not in strategy.keys():
                assert 'mbs' in strategy.keys() and 'seq_len' in strategy.keys()
                max_tokens = strategy['mbs'] * strategy['seq_len']
            else:
                max_tokens = strategy['max_tokens']
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

    def get_mbs_map(self, strategy_pool, task_seq_len_range):
        mbs_map = []
        for seq_len in task_seq_len_range:
            mbs_of_seq_len = [self.cost_model.max_tokens[strategy[1]] // seq_len for strategy in strategy_pool]
            mbs_map.append(mbs_of_seq_len)
        return mbs_map

    def fit_time(self, X, aux_fragment, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + (c3 * mbs + c4 * aux_fragment) * seq_len + (c5 * mbs + c6 * aux_fragment)

    def estimate_time(self, mbs, s, strategy, aux_fragment=1):
        tp = strategy[1][0]
        pp = strategy[1][1]
        return self.fit_time((mbs, s), aux_fragment, *self.popt[tp]) * self.num_layers / pp

    def get_estimated_time(self, mbs, s, strategy):
        if mbs == 0:
            return 0
        tp = strategy[1][0]
        pp = strategy[1][1]
        return self.fit_time((mbs, s), 1, *self.popt[tp]) * self.num_layers / pp
    
    def estimate_total_time(self, strategy_idx, m, r, aux_bool_fragment, max_batch_time):
        strategy = self.strategy_pool[strategy_idx]
        pp = strategy[1][1]
        return quicksum(self.estimate_time(self.mbs_map[i][strategy_idx], seq_len, strategy) * m[i][strategy_idx] + \
    					self.estimate_time(r[i][strategy_idx], seq_len, strategy, aux_bool_fragment[i][strategy_idx]) \
    					for i, seq_len in enumerate(self.task_seq_lens)) + (pp - 1) * max_batch_time

    def build_planner(self, seq_distribution):
        strategy_idx = 0
        max_tokens_list = [strategy[2] for strategy in self.strategy_pool]
        sorted_max_tokens_idx = np.argsort(max_tokens_list)
        for i, seq_len in enumerate(self.task_seq_lens):
            while seq_len > self.strategy_pool[sorted_max_tokens_idx[strategy_idx]][2]:
                strategy_idx += 1
            max_tokens_of_seq_len = self.strategy_pool[sorted_max_tokens_idx[strategy_idx]][2]
            for j in range(self.strategy_pool_size):
                if self.strategy_pool[sorted_max_tokens_idx[j]][2] != max_tokens_of_seq_len:
                    self.mbs_map[i][sorted_max_tokens_idx[j]] = 0
        print(f"mbs_map = {self.mbs_map}")
        model = Model("grouped_static_batch_planner")
        dp = [model.addVar(lb=0, ub=self.gpu_num // self.strategy_pool[i][0], vtype="I", name="dp_strategy%s" % i) \
              for i in range(self.strategy_pool_size)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, strategy%s)" % (seq_len, i)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_pool_size)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_pool_size)] for seq_len in self.task_seq_lens]
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time_strategy%s" % i) if self.strategy_pool[i][1][1] > 1 else 0 \
                          for i in range(self.strategy_pool_size)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(strategy%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.strategy_pool[i][1][1] > 1 else [] for i in range(self.strategy_pool_size)]

        # gpu num
        model.addCons(quicksum(dp[i] * self.strategy_pool[i][0] for i in range(self.strategy_pool_size)) == self.gpu_num, name="eq_gpus")
        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.strategy_pool_size)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.strategy_pool_size)) <= seq_distribution[seq_len] + np.sum([self.gpu_num // self.strategy_pool[k][0] - 1 for k in range(self.strategy_pool_size)]), name="le_dispatch_seq_num_%s" % seq_len)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.strategy_pool_size):
                model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
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
        for i in range(self.strategy_pool_size):
            if self.strategy_pool[i][1][1] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(max_batch_time[i] >= self.estimate_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) * aux_bool_complete[j][i], name="max_batch_time_ge_complete(strategy%s, %s)" % (i, j))
                # model.addCons(max_batch_time[i] <= self.estimate_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) * aux_bool_complete[j][i] + self.max_batch_time_list[i] * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) * aux_bool_complete[j][i] + self.M_bound * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                # fragment
                model.addCons(max_batch_time[i] >= self.estimate_time(r[j][i], seq_len, self.strategy_pool[i], aux_bool_fragment[j][i]), name="max_batch_time_ge_fragment(strategy%s, %s)" % (i, j))
                # model.addCons(max_batch_time[i] <= self.estimate_time(r[j][i], seq_len, self.strategy_pool[i], aux_bool_fragment[j][i]) + self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]), name="max_batch_time_le_fragment_plus_bound(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(r[j][i], seq_len, self.strategy_pool[i], aux_bool_fragment[j][i]) + self.M_bound * (1 - aux_max[i][j + len(self.task_seq_lens)]), name="max_batch_time_le_fragment_plus_bound(strategy%s, %s)" % (i, j))

        # Set objective function
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for i in range(self.strategy_pool_size):
            model.addCons(objvar >= self.estimate_total_time(i, m, r, aux_bool_fragment, max_batch_time[i]))
        return model

    def schedule(self, multi_task_seq_distribution):
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        # assert task_seq_len_range.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        self.mbs_map = self.get_mbs_map(self.strategy_pool, self.task_seq_lens)
        self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_pool_size)]
    	# for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        global_batch_size_across_tasks = 0
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            for seq_len, p in task_seq_distribution.items():
                # seq_num = math.ceil(p * self.global_batch_size_list[i])
                seq_num = p * self.global_batch_size_list[i]
                seq_distribution_across_tasks[seq_len] += seq_num
                global_batch_size_across_tasks += seq_num
        seq_distribution_across_tasks = {s : math.ceil(num) for s, num in seq_distribution_across_tasks.items()}
        print(f"Static planner start to optimize the strategy config, it may take a few minutes...")
        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        # model.setIntParam("timing/clocktype", 2) # wall clock
        model.setIntParam("lp/threads", self.lp_threads)
        # model.setRealParam('limits/time', 180)
        # model.setRealParam("limits/gap", 0.1)
        model.hideOutput()
        model.optimize()
        try:
            model.getObjVal()
        except:
            print("No solution found")
            exit(-1)
        end_time = time.time()
        # 获取ds config, max tokens, num_strategy
        ds_parallel_config = {
            'dps': [],
            'tps': [],
            'pps': [],
            'max_tokens': [],
            'num_strategy': 0
        }
        for v in model.getVars():
            if "dp" in v.name and round(model.getVal(v)) > 0:
                dp = round(model.getVal(v))
                strategy_idx = int(re.search(r'\d+', v.name).group())
                strategy = self.strategy_pool[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                ds_parallel_config['dps'].append(dp)
                ds_parallel_config['tps'].append(tp)
                ds_parallel_config['pps'].append(pp)
                ds_parallel_config['max_tokens'].append(max_tokens)
                ds_parallel_config['num_strategy'] += 1

        print(f"Static batch planner takes {end_time - start_time:.4f}s to get strategy config, with {ds_parallel_config['num_strategy']} strategies as follows:")
        for i in range(ds_parallel_config['num_strategy']):
            print(f"strategy {i}: dp = {ds_parallel_config['dps'][i]}, tp = {ds_parallel_config['tps'][i]}, pp = {ds_parallel_config['pps'][i]}, max_tokens = {ds_parallel_config['max_tokens'][i]}")
        return ds_parallel_config

class NewStaticBatchPlanner:
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        gpu_num,
        strategy_candidates,
        use_optimized_strategy_pool=True,
        lp_threads=32
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.gpu_num = gpu_num
        self.lp_threads = lp_threads
        self.cost_model = cost_model
        self.cache_estimate_times = {}
        self.popt, self.profile_seq_lens = cost_model.popt, cost_model.seq_len_range
        self.strategy_pool = self.get_optimized_strategy_pool(strategy_candidates)
        self.strategy_pool_size = len(self.strategy_pool)
        self.use_optimized_strategy_pool = use_optimized_strategy_pool
    
    def set_global_batch_size(self, global_batch_size, i):
        self.global_batch_size_list[i] = global_batch_size

    def get_optimized_strategy_pool(self, strategy_candidates):
        goat_strategy_of_max_tokens = {}
        min_gpu_num_strategy_of_max_tokens = {}
        if not self.use_optimized_strategy_pool:
            strategy_pool = [(strategy['tp'] * strategy['pp'], (strategy['tp'], strategy['pp']), strategy['max_tokens']) for strategy in strategy_candidates if strategy['max_tokens'] > 0]
            print(f"strategy_pool = {strategy_pool}")
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

    def get_mbs_map(self, strategy_pool, task_seq_len_range):
        mbs_map = []
        for seq_len in task_seq_len_range:
            mbs_of_seq_len = [self.cost_model.max_tokens[strategy[1]] // seq_len for strategy in strategy_pool]
            mbs_map.append(mbs_of_seq_len)
        return mbs_map

    def fit_time(self, X, aux_fragment, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + (c3 * mbs + c4 * aux_fragment) * seq_len + (c5 * mbs + c6 * aux_fragment)

    def estimate_time(self, mbs, s, strategy, aux_fragment=1):
        tp = strategy[1][0]
        pp = strategy[1][1]
        return self.fit_time((mbs, s), aux_fragment, *self.popt[tp]) * self.num_layers / pp

    def get_estimated_time(self, mbs, s, tp, pp):
        if mbs == 0:
            return 0
        return self.fit_time((mbs, s), 1, *self.popt[tp]) * self.num_layers / pp
    
    def print_estimate_total_time(self, dp, tp, pp, multi_task_batch_dispatch_map, strategy_id, max_tokens):
        estimate_time = 0
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
                m = seq_len_map[seq_len] // dp // mbs
            rest_sample_num = seq_len_map[seq_len] // dp - mbs * m
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
    
    def estimate_total_time(self, strategy_idx, m, r, aux_bool_fragment, max_batch_time):
        strategy = self.strategy_pool[strategy_idx]
        pp = strategy[1][1]
        return quicksum(self.cache_estimate_times.get((self.mbs_map[i][strategy_idx], seq_len, strategy[1][0], strategy[1][1]), 0) * m[i][strategy_idx] + \
                        self.estimate_time(r[i][strategy_idx], seq_len, strategy, aux_bool_fragment[i][strategy_idx]) \
                        for i, seq_len in enumerate(self.task_seq_lens)) + (pp - 1) * max_batch_time

    def build_planner(self, seq_distribution):
        max_seq_len = max(seq_distribution.keys())
        max_seq_len_to_2 = 2 ** int(np.ceil(np.log2(max_seq_len)))
        max_seq_len_strategy_num = 0
        for strategy in self.strategy_pool:
            if strategy[2] == max_seq_len_to_2:
                max_seq_len_strategy_num += 1
        model = Model("static_batch_planner")
        dp = [model.addVar(lb=1 if max_seq_len_strategy_num == 1 and self.strategy_pool[i][2] == max_seq_len_to_2 else 0, ub=self.gpu_num // self.strategy_pool[i][0] if self.strategy_pool[i][2] <= max_seq_len_to_2 else 0, vtype="I", name="dp_strategy%s" % i) \
              for i in range(self.strategy_pool_size)]
        # dp = [model.addVar(lb=0, ub=self.gpu_num // self.strategy_pool[i][0], vtype="I", name="dp_strategy%s" % i) \
        #       for i in range(self.strategy_pool_size)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_pool_size)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_pool_size)] for seq_len in self.task_seq_lens]
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time_strategy%s" % i) if self.strategy_pool[i][1][1] > 1 else 0 \
                          for i in range(self.strategy_pool_size)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(strategy%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.strategy_pool[i][1][1] > 1 else [] for i in range(self.strategy_pool_size)]

        # gpu num
        model.addCons(quicksum(dp[i] * self.strategy_pool[i][0] for i in range(self.strategy_pool_size)) == self.gpu_num, name="eq_gpus")
        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.strategy_pool_size)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.strategy_pool_size)) <= seq_distribution[seq_len] + np.sum([self.gpu_num // self.strategy_pool[k][0] - 1 for k in range(self.strategy_pool_size)]), name="le_dispatch_seq_num_%s" % seq_len)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.strategy_pool_size):
                model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
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
        for i in range(self.strategy_pool_size):
            if self.strategy_pool[i][1][1] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(max_batch_time[i] >= self.cache_estimate_times.get((self.mbs_map[j][i], seq_len, self.strategy_pool[i][1][0], self.strategy_pool[i][1][1]), 0) * aux_bool_complete[j][i], name="max_batch_time_ge_complete(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.cache_estimate_times.get((self.mbs_map[j][i], seq_len, self.strategy_pool[i][1][0], self.strategy_pool[i][1][1]), 0) * aux_bool_complete[j][i] + self.max_batch_time_list[i] * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                # model.addCons(max_batch_time[i] >= self.estimate_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) * aux_bool_complete[j][i], name="max_batch_time_ge_complete(strategy%s, %s)" % (i, j))
                # model.addCons(max_batch_time[i] <= self.estimate_time(self.mbs_map[j][i], seq_len, self.strategy_pool[i]) * aux_bool_complete[j][i] + self.max_batch_time_list[i] * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                # fragment
                model.addCons(max_batch_time[i] >= self.estimate_time(r[j][i], seq_len, self.strategy_pool[i], aux_bool_fragment[j][i]), name="max_batch_time_ge_fragment(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(r[j][i], seq_len, self.strategy_pool[i], aux_bool_fragment[j][i]) + self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]), name="max_batch_time_le_fragment_plus_bound(strategy%s, %s)" % (i, j))

        # Set objective function
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for i in range(self.strategy_pool_size):
            model.addCons(objvar >= self.estimate_total_time(i, m, r, aux_bool_fragment, max_batch_time[i]))
        return model

    def schedule(self, multi_task_seq_distribution):
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        # assert task_seq_len_range.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        self.mbs_map = self.get_mbs_map(self.strategy_pool, self.task_seq_lens)
        tp_list = [strategy[1][0] for strategy in self.strategy_pool]
        pp_list = [strategy[1][1] for strategy in self.strategy_pool]
        self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, tp=self.strategy_pool[i][1][0], pp=self.strategy_pool[i][1][1]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_pool_size)]
        self.cache_estimate_times = {(mbs, seq_len, tp, pp): self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
                                     for mbs in set(np.array(self.mbs_map).flatten()) if mbs > 0
                                     for seq_len in self.task_seq_lens
                                     for tp, pp in zip(tp_list, pp_list)}
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
                    print(f"{i}: {task_seq_distribution[max_seq_len]}")
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break
        # print(f"sequence distribution = {seq_distribution_across_tasks}")
        # print(f"Static planner start to optimize the strategy config, it may take a few minutes...")
        # start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        # model.setIntParam("timing/clocktype", 2) # wall clock
        model.setIntParam("lp/threads", 64)
        model.setIntParam("parallel/maxnthreads", 64)
        # model.setRealParam("limits/gap", 1e-3)
        # model.setRealParam('limits/time', 180)
        # model.hideOutput()
        model.optimize()
        try:
            model.getObjVal()
        except:
            print("No solution found")
            exit(-1)
        # end_time = time.time()
        # 获取ds config, max tokens, num_strategy
        ds_parallel_config = {
            'strategy_idxs': [],
            'dps': [],
            'tps': [],
            'pps': [],
            'max_tokens': [],
            'num_strategy': 0
        }
        for v in model.getVars():
            if "dp" in v.name and round(model.getVal(v)) > 0:
                dp = round(model.getVal(v))
                strategy_idx = int(re.search(r'\d+', v.name).group())
                strategy = self.strategy_pool[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                ds_parallel_config['strategy_idxs'].append(strategy_idx)
                ds_parallel_config['dps'].append(dp)
                ds_parallel_config['tps'].append(tp)
                ds_parallel_config['pps'].append(pp)
                ds_parallel_config['max_tokens'].append(max_tokens)
                ds_parallel_config['num_strategy'] += 1

        # print(f"Static batch planner takes {end_time - start_time:.4f}s to get strategy config, with {ds_parallel_config['num_strategy']} strategies as follows:")
        # for i in range(ds_parallel_config['num_strategy']):
        #     print(f"strategy {i}: dp = {ds_parallel_config['dps'][i]}, tp = {ds_parallel_config['tps'][i]}, pp = {ds_parallel_config['pps'][i]}, max_tokens = {ds_parallel_config['max_tokens'][i]}")
        
        # profile: 写入文件
        # with open("sensitivity_exp.txt", "a") as f:
        #     f.write(f"bucket num: {len(self.task_seq_lens)}\n")
        #     for i in range(ds_parallel_config['num_strategy']):
        #         f.write(f"strategy {i}: dp = {ds_parallel_config['dps'][i]}, tp = {ds_parallel_config['tps'][i]}, pp = {ds_parallel_config['pps'][i]}, max_tokens = {ds_parallel_config['max_tokens'][i]}\n")
        #     f.write("\n")
        
        # debug: 获取pipeline时间
        # get the dispatch result for all strategies
        seq_len_num_map_list = []
        for i in range(ds_parallel_config['num_strategy']):
            seq_len_num_map = {s : 0 for s in self.task_seq_lens}
            for v in model.getVars():
                for seq_len in self.task_seq_lens:
                    n_name = "seq_num(%s, strategy%s)" % (seq_len, ds_parallel_config['strategy_idxs'][i])
                    if n_name in v.name:
                        n_val = round(model.getVal(v))
                        if n_val < 1:
                            continue
                        seq_len_num_map[seq_len] = n_val * ds_parallel_config['dps'][i]
            seq_len_num_map_list.append(seq_len_num_map)
        # print(f"multi_task_seq_distribution = {multi_task_seq_num_distribution}")
        # print(f"[DEBUG] seq_len_num_map_list = {seq_len_num_map_list}")
        # tune dispatched seq num
        for seq_len in self.task_seq_lens:
            total_seq_num = seq_distribution_across_tasks[seq_len]
            dispatched_seq_num = 0
            for i in range(ds_parallel_config['num_strategy']):
                dispatched_seq_num += seq_len_num_map_list[i][seq_len]
            if dispatched_seq_num > total_seq_num:
                dp_strategy = sorted([i for i in range(ds_parallel_config['num_strategy']) if ds_parallel_config['dps'][i] > 1 and seq_len_num_map_list[i][seq_len] > 0],
                                     key=lambda x: ds_parallel_config['dps'][x], reverse=True)
                overflow_seq_num = dispatched_seq_num - total_seq_num
                max_overflow_seq_num = np.sum([ds_parallel_config['dps'][i] - 1 for i in dp_strategy])
                if overflow_seq_num > max_overflow_seq_num:
                    non_dp_strategy = sorted([i for i in range(ds_parallel_config['num_strategy']) if ds_parallel_config['dps'][i] == 1 and seq_len_num_map_list[i][seq_len] > 0],
                                              key=lambda x: seq_len_num_map_list[x][seq_len], reverse=True)
                    total_overflow_num = overflow_seq_num - max_overflow_seq_num
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
                    # assert total_overflow_num == 0
                    overflow_seq_num = max_overflow_seq_num
                # assert overflow_seq_num <= max_overflow_seq_num, \
                #     f"overflow seq num is larger than the limit for seq_len = {seq_len} where limit = {max_overflow_seq_num}"
                for i in dp_strategy:
                    if overflow_seq_num == 0:
                        break
                    seq_len_num_map_list[i][seq_len] -= min(ds_parallel_config['dps'][i] - 1, overflow_seq_num)
                    overflow_seq_num -= min(ds_parallel_config['dps'][i] - 1, overflow_seq_num)
        # print(f"[DEBUG] after tune: seq_len_num_map_list = {seq_len_num_map_list}")

        # dispatch task-specific samples
        multi_task_batch_dispatch_map = {task_id : [{s : 0 for s in self.task_seq_lens} for _ in range(ds_parallel_config['num_strategy'])] for task_id in range(self.train_task_num)}
        for seq_idx, seq_len in enumerate(self.task_seq_lens):
            for task_id in range(self.train_task_num):
                for i in range(ds_parallel_config['num_strategy']):
                    dispatch_num = min(seq_len_num_map_list[i].get(seq_len, 0), multi_task_seq_num_distribution[task_id].get(seq_len, 0))
                    if dispatch_num == 0:
                        continue
                    multi_task_batch_dispatch_map[task_id][i][seq_len] += int(dispatch_num)
                    seq_len_num_map_list[i][seq_len] -= dispatch_num
                    multi_task_seq_num_distribution[task_id][seq_len] -= dispatch_num
                if multi_task_seq_num_distribution[task_id].get(seq_len, 0) > 0:
                    assert seq_idx < len(self.task_seq_lens) - 1, f"seq_len = {seq_len}"
                    if self.task_seq_lens[seq_idx + 1] not in multi_task_seq_num_distribution[task_id].keys():
                        multi_task_seq_num_distribution[task_id][self.task_seq_lens[seq_idx + 1]] = 0
                    multi_task_seq_num_distribution[task_id][self.task_seq_lens[seq_idx + 1]] += multi_task_seq_num_distribution[task_id][seq_len]

        # profile
        for i in range(ds_parallel_config['num_strategy']):
            self.print_estimate_total_time(ds_parallel_config['dps'][i], ds_parallel_config['tps'][i], ds_parallel_config['pps'][i], multi_task_batch_dispatch_map, i, ds_parallel_config['max_tokens'][i])
        return ds_parallel_config

class GroupedDynamicBatchPlanner:
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        strategy_num,
        max_tokens_list,
        dps,
        tps,
        pps,
        cur_strategy_id=0,
        local_device=0,
        lp_threads=32,
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.global_batch_size_across_tasks = 0
        self.strategy_num = strategy_num
        self.max_tokens_list = max_tokens_list
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.running_task_ids = None
        self.dps = dps
        self.tps = tps
        self.pps = pps
        self.lp_threads = lp_threads
        self.popt, self.profile_seq_lens = cost_model.popt, cost_model.seq_len_range
        # tokens
        self.token_num = 0
        self.valid_token_num = 0
        # debug
        self.local_device = local_device
        self.cur_strategy_id = cur_strategy_id
    
    def fit_time(self, X, aux_fragment, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + (c3 * mbs + c4 * aux_fragment) * seq_len + (c5 * mbs + c6 * aux_fragment)
    
    def estimate_time(self, mbs, s, tp, pp, aux_fragment=1):
        return self.fit_time((mbs, s), aux_fragment, *self.popt[tp]) * self.num_layers / pp

    def estimate_total_time(self, strategy_idx, m, r, aux_bool_fragment, max_batch_time):
        tp = self.tps[strategy_idx]
        pp = self.pps[strategy_idx]
        return quicksum(self.estimate_time(self.mbs_map[i][strategy_idx], seq_len, tp, pp) * m[i][strategy_idx] + \
                        self.estimate_time(r[i][strategy_idx], seq_len, tp, pp, aux_bool_fragment[i][strategy_idx]) \
                        for i, seq_len in enumerate(self.task_seq_lens)) + (pp - 1) * max_batch_time
    
    def get_estimated_time(self, mbs, s, tp, pp):
        if mbs == 0:
            return 0
        return self.fit_time((mbs, s), 1, *self.popt[tp]) * self.num_layers / pp
    
    def print_estimate_total_time(self, tp, pp, multi_task_batch_dispatch_map, strategy_id):
        estimate_time = 0
        max_tokens = self.max_tokens_list[strategy_id]
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        for task in self.running_task_ids:
            print(f"task {task}: {multi_task_batch_dispatch_map[task][strategy_id]}")
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        seq_len_map = {seq_len : np.sum([multi_task_batch_dispatch_map[task][strategy_id][seq_len] for task in self.running_task_ids]) \
                       for seq_len in self.task_seq_lens}
        estimate_time = 0
        max_batch_time = 0
        seq_to_num = {}
        seq_to_time = {}
        seq_num = 0
        def get_bucket(seq_len):
            if seq_len <= 2048:
                return 2048
            elif seq_len <= 4096:
                return 4096
            elif seq_len <= 8192:
                return 8192
            else:
                return 16384
        for seq_len in self.task_seq_lens:
            bucket = get_bucket(seq_len)
            mbs = max_tokens // seq_len
            if mbs == 0:
                m = 0
            else:
                m = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] // mbs
            rest_sample_num = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] - mbs * m
            full_time = self.get_estimated_time(mbs, seq_len, tp, pp)
            piece_time = self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            seq_to_num[bucket] = seq_to_num.get(bucket, 0) + (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id]
            seq_num += (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id]
            seq_to_time[bucket] = seq_to_time.get(bucket, 0) + (full_time * m + piece_time)
            if m > 0:
                print(f"|---(mbs, seq_len, m) = ({mbs}, {seq_len}, {m}): {full_time * m / 1000}s")
            if rest_sample_num > 0:
                print(f"|---(mbs, seq_len, m) = ({rest_sample_num}, {seq_len}, 1): {piece_time / 1000}s")
            estimate_time += self.get_estimated_time(mbs, seq_len, tp, pp) * m + self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            cur_max_time = full_time if m > 0 else piece_time
            max_batch_time = np.max([max_batch_time, cur_max_time])
        estimate_time += (pp - 1) * max_batch_time
        print(f"strategy - {strategy_id}: max_batch_time = {max_batch_time / 1000}s, total_estimate_time = {estimate_time / 1000} s")
        seq_to_percent = {k: v / seq_num for (k, v) in seq_to_num.items()}
        seq_to_time = {k: v / estimate_time for (k, v) in seq_to_time.items()}
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
        if strategy_id == self.cur_strategy_id:
            with open(f"case_study/GROUP/{local_host_name}-{self.local_device}.txt", 'a') as f:
                f.write(f"{seq_to_percent}\n")
                f.write(f"{seq_to_time}\n")
        print(f"seq_to_percent = {seq_to_percent}")
        print(f"seq_to_time = {seq_to_time}")
        return estimate_time
    
    def build_planner(self, seq_distribution):
        # get cumulative_seq_distribution
        # cumulative_seq_distribution = seq_distribution.copy()
        # cumulative_seq_num = 0
        # for seq_len in sorted(self.task_seq_lens):
        #     cumulative_seq_num += seq_distribution[seq_len]
        #     cumulative_seq_distribution[seq_len] = cumulative_seq_num
        # seq len range group for different strategies
        strategy_idx = 0
        sorted_max_tokens_idx = np.argsort(self.max_tokens_list)
        for i, seq_len in enumerate(self.task_seq_lens):
            while seq_len > self.max_tokens_list[sorted_max_tokens_idx[strategy_idx]]:
                strategy_idx += 1
            max_tokens_of_seq_len = self.max_tokens_list[sorted_max_tokens_idx[strategy_idx]]
            for j in range(self.strategy_num):
                if self.max_tokens_list[sorted_max_tokens_idx[j]] != max_tokens_of_seq_len:
                    self.mbs_map[i][sorted_max_tokens_idx[j]] = 0
        # print(f"mbs_map = {self.mbs_map}")
        model = Model("grouped_dynamic_batch_planner")
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, strategy%s)" % (seq_len, j)) \
             for j in range(self.strategy_num)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_num)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, strategy%s)" % (seq_len, i)) \
                             for i in range(self.strategy_num)] for seq_len in self.task_seq_lens]
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
            # model.addCons(quicksum(self.dps[j] * n[i][j] for j in range(self.strategy_num)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            # model.addCons(quicksum(self.dps[j] * n[i][j] for j in range(self.strategy_num)) <= seq_distribution[seq_len] + np.sum([self.dps[k] - 1 for k in range(self.strategy_num)]), name="le_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        # for j in range(self.strategy_num):
        #     model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(strategy%s)" % j)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.strategy_num):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                # model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
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
                model.addCons(max_batch_time[i] >= self.estimate_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) * aux_bool_complete[j][i], name="max_batch_time_ge_complete(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) * aux_bool_complete[j][i] + self.max_batch_time_list[i] * (1 - aux_max[i][j]), name="max_batch_time_le_complete_plus_bound(strategy%s, %s)" % (i, j))
                # fragment
                model.addCons(max_batch_time[i] >= self.estimate_time(r[j][i], seq_len, self.tps[i], self.pps[i], aux_bool_fragment[j][i]), name="max_batch_time_ge_fragment(strategy%s, %s)" % (i, j))
                model.addCons(max_batch_time[i] <= self.estimate_time(r[j][i], seq_len, self.tps[i], self.pps[i], aux_bool_fragment[j][i]) + self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]), name="max_batch_time_le_fragment_plus_bound(strategy%s, %s)" % (i, j))
        # 设置目标函数
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for j in range(self.strategy_num):
            model.addCons(objvar >= self.estimate_total_time(j, m, r, aux_bool_fragment, max_batch_time[j]))
        return model
    
    def schedule(self, seq_distribution_map):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        # print(f"new dynamic planner: {seq_distribution_map}")
        self.running_task_ids = sorted(list(seq_distribution_map.keys()))
        for task_id in self.running_task_ids:
            for seq_len in sorted(seq_distribution_map[task_id].keys()):
                seq_distribution_map[task_id][seq_len] = int(seq_distribution_map[task_id][seq_len] * self.global_batch_size_list[task_id])
        task_seq_lens = set()
        for i in self.running_task_ids:
            task_seq_lens = task_seq_lens.union(set(seq_distribution_map[i].keys()))
        # assert task_seq_lens.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_lens))
        self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
        self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_num)]
        
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        self.global_batch_size_across_tasks = 0
        for i in self.running_task_ids:
            for seq_len in seq_distribution_map[i].keys():
                seq_distribution_across_tasks[seq_len] += seq_distribution_map[i][seq_len]
            self.global_batch_size_across_tasks += self.global_batch_size_list[i]
        # print(f"Dynamic planner start to dispatch batch for running tasks...")
        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        # model.setLongintParam("limits/nodes", 100000)
        # model.setLongintParam("limits/stallnodes", 500)
        model.setLongintParam("limits/stallnodes", 5000)
        # model.setRealParam("limits/gap", 1e-3)
        model.hideOutput()
        model.optimize()
        try:
            model.getObjVal()
        except:
            print("No solution found")
            exit(-1)
        end_time = time.time()
        print(f"Dynamic batch planner takes {end_time - start_time:.4f}s to dispatch running batch")
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
                dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] > 1 and seq_len_num_map_list[i][seq_len] > 0],
                                     key=lambda x: self.dps[x], reverse=True)
                overflow_seq_num = dispatched_seq_num - total_seq_num
                max_overflow_seq_num = np.sum([self.dps[i] - 1 for i in dp_strategy])
                assert overflow_seq_num <= max_overflow_seq_num, \
                    f"overflow seq num is larger than the limit for seq_len = {seq_len} where limit = {max_overflow_seq_num}"
                for i in dp_strategy:
                    if overflow_seq_num == 0:
                        break
                    seq_len_num_map_list[i][seq_len] -= min(self.dps[i] - 1, overflow_seq_num)
                    overflow_seq_num -= min(self.dps[i] - 1, overflow_seq_num)
        # print(f"[DEBUG] after tune: seq_len_num_map_list = {seq_len_num_map_list}")

        # dispatch task-specific samples
        multi_task_batch_dispatch_map = {task_id : [{s : 0 for s in self.task_seq_lens} for _ in range(self.strategy_num)] for task_id in self.running_task_ids}
        for seq_idx, seq_len in enumerate(self.task_seq_lens):
            for task_id in self.running_task_ids:
                for i in range(self.strategy_num):
                    dispatch_num = min(seq_len_num_map_list[i].get(seq_len, 0), seq_distribution_map[task_id].get(seq_len, 0))
                    if dispatch_num == 0:
                        continue
                    multi_task_batch_dispatch_map[task_id][i][seq_len] += dispatch_num
                    seq_len_num_map_list[i][seq_len] -= dispatch_num
                    seq_distribution_map[task_id][seq_len] -= dispatch_num
                if seq_distribution_map[task_id].get(seq_len, 0) > 0:
                    assert seq_idx < len(self.task_seq_lens) - 1
                    if self.task_seq_lens[seq_idx + 1] not in seq_distribution_map[task_id].keys():
                        seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] = 0
                    seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] += seq_distribution_map[task_id][seq_len]

        # profile
        for i in range(self.strategy_num):
            self.print_estimate_total_time(self.tps[i], self.pps[i], multi_task_batch_dispatch_map, i)
        
        return multi_task_batch_dispatch_map, end_time - start_time

class NewDynamicBatchPlanner:
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        strategy_num,
        max_tokens_list,
        dps,
        tps,
        pps,
        cur_strategy_id=0,
        local_device=0,
        lp_threads=64,
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.global_batch_size_across_tasks = 0
        self.strategy_num = strategy_num
        self.max_tokens_list = max_tokens_list
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.running_task_ids = None
        self.cache_estimate_times = {}
        self.cache_fit_times = {}
        self.aux_fragment = []
        self.aux_complete = []
        self.dps = dps
        self.tps = tps
        self.pps = pps
        self.lp_threads = lp_threads
        self.popt, self.profile_seq_lens = cost_model.popt, cost_model.seq_len_range
        # tokens
        self.token_num = 0
        self.valid_token_num = 0
        # debug
        self.local_device = local_device
        self.cur_strategy_id = cur_strategy_id

    def fit_time(self, X, aux_fragment, tp, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        # if (seq_len, tp) not in self.cache_fit_times.keys():
        #     self.cache_fit_times[(seq_len, tp)] = (
        #         c1 * seq_len * seq_len + c3 * seq_len + c5,
        #         c2 * seq_len * seq_len + c4 * seq_len + c6
        #     )
        #     return self.cache_fit_times[(seq_len, tp)][0] * mbs + self.cache_fit_times[(seq_len, tp)][1] * aux_fragment
        # return self.cache_fit_times[(seq_len, tp)][0] * mbs + self.cache_fit_times[(seq_len, tp)][1] * aux_fragment
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + (c3 * mbs + c4 * aux_fragment) * seq_len + (c5 * mbs + c6 * aux_fragment)
    
    def estimate_time(self, mbs, s, tp, pp, aux_fragment=1):
        # 打印mbs的类型
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
        # assert self.fit_time((mbs, s), 1, *self.popt[tp]) * self.num_layers / pp == self.cache_estimate_times.get((mbs, s, tp, pp), 0), \
        #     f'different of {mbs, s, tp, pp} = {self.fit_time((mbs, s), 1, *self.popt[tp]) * self.num_layers / pp} and {self.cache_estimate_times.get((mbs, s, tp, pp), 0)}'
        return self.fit_time((mbs, s), 1, tp, *self.popt[tp]) * self.num_layers / pp
    
    def print_estimate_total_time(self, tp, pp, multi_task_batch_dispatch_map, strategy_id):
        estimate_time = 0
        max_tokens = self.max_tokens_list[strategy_id]
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        for task in self.running_task_ids:
            print(f"task {task}: {multi_task_batch_dispatch_map[task][strategy_id]}")
        print(f"-----strategy - {strategy_id}: multi task seq_len map-----")
        seq_len_map = {seq_len : np.sum([multi_task_batch_dispatch_map[task][strategy_id][seq_len] for task in self.running_task_ids]) \
                       for seq_len in self.task_seq_lens}
        estimate_time = 0
        max_batch_time = 0
        seq_to_num = {}
        seq_to_time = {}
        seq_num = 0
        def get_bucket(seq_len):
            if seq_len <= 2048:
                return 2048
            elif seq_len <= 4096:
                return 4096
            elif seq_len <= 8192:
                return 8192
            else:
                return 16384
        for seq_len in self.task_seq_lens:
            bucket = get_bucket(seq_len)
            mbs = max_tokens // seq_len
            if mbs == 0:
                m = 0
            else:
                m = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] // mbs
                # m = seq_len_map[seq_len] // self.dps[strategy_id] // mbs
            rest_sample_num = (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id] - mbs * m
            full_time = self.get_estimated_time(mbs, seq_len, tp, pp)
            piece_time = self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            seq_to_num[bucket] = seq_to_num.get(bucket, 0) + (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id]
            seq_num += (seq_len_map[seq_len] + self.dps[strategy_id] - 1) // self.dps[strategy_id]
            seq_to_time[bucket] = seq_to_time.get(bucket, 0) + (full_time * m + piece_time)
            if m > 0:
                print(f"|---(mbs, seq_len, m) = ({mbs}, {seq_len}, {m}): {full_time * m / 1000}s")
            if rest_sample_num > 0:
                print(f"|---(mbs, seq_len, m) = ({rest_sample_num}, {seq_len}, 1): {piece_time / 1000}s")
            estimate_time += self.get_estimated_time(mbs, seq_len, tp, pp) * m + self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            cur_max_time = full_time if m > 0 else piece_time
            max_batch_time = np.max([max_batch_time, cur_max_time])
        estimate_time += (pp - 1) * max_batch_time
        print(f"strategy - {strategy_id}: max_batch_time = {max_batch_time / 1000}s, total_estimate_time = {estimate_time / 1000} s")
        seq_to_percent = {k: v / seq_num for (k, v) in seq_to_num.items()}
        seq_to_time = {k: v / estimate_time for (k, v) in seq_to_time.items()}
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
        if strategy_id == self.cur_strategy_id:
            with open(f"case_study/BALANCE/{local_host_name}-{self.local_device}.txt", 'a') as f:
                f.write(f"{seq_to_percent}\n")
                f.write(f"{seq_to_time}\n")
        print(f"seq_to_percent = {seq_to_percent}")
        print(f"seq_to_time = {seq_to_time}")
        return estimate_time
    
    def get_estimate_total_time(self, tp, pp, multi_task_batch_dispatch_map, strategy_id):
        estimate_time = 0
        max_tokens = self.max_tokens_list[strategy_id]
        seq_len_map = {seq_len : np.sum([multi_task_batch_dispatch_map[task][strategy_id][seq_len] for task in self.running_task_ids]) \
                       for seq_len in self.task_seq_lens}
        estimate_time = 0
        max_batch_time = 0
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            if mbs == 0:
                m = 0
            else:
                m = seq_len_map[seq_len] // self.dps[strategy_id] // mbs
            rest_sample_num = seq_len_map[seq_len] // self.dps[strategy_id] - mbs * m
            full_time = self.get_estimated_time(mbs, seq_len, tp, pp)
            piece_time = self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            estimate_time += self.get_estimated_time(mbs, seq_len, tp, pp) * m + self.get_estimated_time(rest_sample_num, seq_len, tp, pp)
            cur_max_time = full_time if m > 0 else piece_time
            max_batch_time = np.max([max_batch_time, cur_max_time])
        estimate_time += (pp - 1) * max_batch_time
        return estimate_time
    
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
            # model.addCons(quicksum(self.dps[j] * n[i][j] for j in range(self.strategy_num)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            # model.addCons(quicksum(self.dps[j] * n[i][j] for j in range(self.strategy_num)) <= seq_distribution[seq_len] + np.sum([self.dps[k] - 1 for k in range(self.strategy_num)]), name="le_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        for j in range(self.strategy_num):
            model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(strategy%s)" % j)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.strategy_num):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
                # model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, strategy%s)" % (seq_len, j))
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
    
    def schedule(self, seq_distribution_map):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        # self.split_strategy_dps()
        # print(f"seq_distribution_map = {seq_distribution_map}")
        self.running_task_ids = sorted(list(seq_distribution_map.keys()))
        for task_id in self.running_task_ids:
            for seq_len in sorted(seq_distribution_map[task_id].keys()):
                seq_distribution_map[task_id][seq_len] = int(seq_distribution_map[task_id][seq_len] * self.global_batch_size_list[task_id])
        task_seq_lens = set()
        for i in self.running_task_ids:
            task_seq_lens = task_seq_lens.union(set(seq_distribution_map[i].keys()))
        # assert task_seq_lens.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_lens))
        self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
        self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_num)]
        self.cache_estimate_times = {(mbs, seq_len, tp, pp): self.fit_time((mbs, seq_len), 1, tp, *self.popt[tp]) * self.num_layers / pp
                                     for mbs in set(np.array(self.mbs_map).flatten()) if mbs > 0
                                     for seq_len in self.task_seq_lens
                                     for tp, pp in zip(self.tps, self.pps)}
        
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        self.global_batch_size_across_tasks = 0
        for i in self.running_task_ids:
            for seq_len in seq_distribution_map[i].keys():
                seq_distribution_across_tasks[seq_len] += seq_distribution_map[i][seq_len]
            self.global_batch_size_across_tasks += self.global_batch_size_list[i]
        # print(f"Dynamic planner start to dispatch batch for running tasks...")
        # print(f"seq distribution across tasks: {seq_distribution_across_tasks}")
        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        model.setIntParam("parallel/maxnthreads", self.lp_threads)
        # model.setLongintParam("limits/nodes", 100000)
        model.setLongintParam("limits/stallnodes", 6000)
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
            return {}, end_time - start_time
        end_time = time.time()
        print(f"Dynamic batch planner takes {end_time - start_time:.4f}s to dispatch running batch")
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
                dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] > 1 and seq_len_num_map_list[i][seq_len] > 0],
                                     key=lambda x: self.dps[x], reverse=True)
                overflow_seq_num = dispatched_seq_num - total_seq_num
                max_overflow_seq_num = np.sum([self.dps[i] - 1 for i in dp_strategy])
                if overflow_seq_num > max_overflow_seq_num:
                    non_dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] == 1 and seq_len_num_map_list[i][seq_len] > 0],
                                              key=lambda x: seq_len_num_map_list[x][seq_len], reverse=True)
                    total_overflow_num = overflow_seq_num - max_overflow_seq_num
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
                    # assert total_overflow_num == 0
                    overflow_seq_num = max_overflow_seq_num
                # assert overflow_seq_num <= max_overflow_seq_num, \
                #     f"overflow seq num is larger than the limit for seq_len = {seq_len} where limit = {max_overflow_seq_num}"
                for i in dp_strategy:
                    if overflow_seq_num == 0:
                        break
                    seq_len_num_map_list[i][seq_len] -= min(self.dps[i] - 1, overflow_seq_num)
                    overflow_seq_num -= min(self.dps[i] - 1, overflow_seq_num)
        # print(f"[DEBUG] after tune: seq_len_num_map_list = {seq_len_num_map_list}")

        # dispatch task-specific samples
        multi_task_batch_dispatch_map = {task_id : [{s : 0 for s in self.task_seq_lens} for _ in range(self.strategy_num)] for task_id in self.running_task_ids}
        for seq_idx, seq_len in enumerate(self.task_seq_lens):
            for task_id in self.running_task_ids:
                for i in range(self.strategy_num):
                    dispatch_num = min(seq_len_num_map_list[i].get(seq_len, 0), seq_distribution_map[task_id].get(seq_len, 0))
                    if dispatch_num == 0:
                        continue
                    multi_task_batch_dispatch_map[task_id][i][seq_len] += int(dispatch_num)
                    seq_len_num_map_list[i][seq_len] -= dispatch_num
                    seq_distribution_map[task_id][seq_len] -= dispatch_num
                if seq_distribution_map[task_id].get(seq_len, 0) > 0:
                    assert seq_idx < len(self.task_seq_lens) - 1
                    if self.task_seq_lens[seq_idx + 1] not in seq_distribution_map[task_id].keys():
                        seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] = 0
                    seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] += seq_distribution_map[task_id][seq_len]

        # profile
        for i in range(self.strategy_num):
            self.print_estimate_total_time(self.tps[i], self.pps[i], multi_task_batch_dispatch_map, i)
        
        return multi_task_batch_dispatch_map, end_time - start_time
    
    def schedule_and_profile(self, seq_distribution_map, node_limit=100000):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        self.running_task_ids = sorted(list(seq_distribution_map.keys()))
        for task_id in self.running_task_ids:
            for seq_len in sorted(seq_distribution_map[task_id].keys()):
                seq_distribution_map[task_id][seq_len] = int(seq_distribution_map[task_id][seq_len] * self.global_batch_size_list[task_id])
        task_seq_lens = set()
        for i in self.running_task_ids:
            task_seq_lens = task_seq_lens.union(set(seq_distribution_map[i].keys()))
        # assert task_seq_lens.issubset(set(self.profile_seq_lens))
        self.task_seq_lens = sorted(list(task_seq_lens))
        self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
        for i, seq_len in enumerate(self.task_seq_lens):
            self.aux_fragment.append({})
            self.aux_complete.append({})
            for j in range(self.strategy_num):
                if self.mbs_map[i][j] <= 1:
                    self.aux_fragment[i][j] = 0
                if self.mbs_map[i][j] == 0:
                    self.aux_complete[i][j] = 0
        self.cache_estimate_times = {(mbs, seq_len, tp, pp): self.fit_time((mbs, seq_len), 1, tp, *self.popt[tp]) * self.num_layers / pp
                                     for mbs in set(np.array(self.mbs_map).flatten()) if mbs > 0
                                     for seq_len in self.task_seq_lens
                                     for tp, pp in zip(self.tps, self.pps)}
        self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.strategy_num)]
        
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        self.global_batch_size_across_tasks = 0
        for i in self.running_task_ids:
            for seq_len in seq_distribution_map[i].keys():
                seq_distribution_across_tasks[seq_len] += seq_distribution_map[i][seq_len]
            self.global_batch_size_across_tasks += self.global_batch_size_list[i]
        # print(f"Dynamic planner start to dispatch batch for running tasks...")
        # print(f"seq distribution across tasks: {seq_distribution_across_tasks}")
        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        model.setIntParam("parallel/maxnthreads", self.lp_threads)
        # model.setLongintParam("limits/nodes", node_limit)
        model.setLongintParam("limits/stallnodes", 500)
        # model.setRealParam("limits/gap", 1e-3)

        model.hideOutput()
        model.optimize()
        try:
            model.getObjVal()
        except:
            end_time = time.time()
            return {}, end_time - start_time, 1e8
        end_time = time.time()
        print(f"Dynamic batch planner takes {end_time - start_time:.4f}s to dispatch running batch")
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
                dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] > 1 and seq_len_num_map_list[i][seq_len] > 0],
                                     key=lambda x: self.dps[x], reverse=True)
                overflow_seq_num = dispatched_seq_num - total_seq_num
                max_overflow_seq_num = np.sum([self.dps[i] - 1 for i in dp_strategy])
                if overflow_seq_num > max_overflow_seq_num:
                    non_dp_strategy = sorted([i for i in range(self.strategy_num) if self.dps[i] == 1 and seq_len_num_map_list[i][seq_len] > 0],
                                              key=lambda x: seq_len_num_map_list[x][seq_len], reverse=True)
                    total_overflow_num = overflow_seq_num - max_overflow_seq_num
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
                    # assert total_overflow_num == 0
                    overflow_seq_num = max_overflow_seq_num
                # assert overflow_seq_num <= max_overflow_seq_num, \
                #     f"overflow seq num is larger than the limit for seq_len = {seq_len} where limit = {max_overflow_seq_num}"
                for i in dp_strategy:
                    if overflow_seq_num == 0:
                        break
                    seq_len_num_map_list[i][seq_len] -= min(self.dps[i] - 1, overflow_seq_num)
                    overflow_seq_num -= min(self.dps[i] - 1, overflow_seq_num)
        # print(f"[DEBUG] after tune: seq_len_num_map_list = {seq_len_num_map_list}")

        # dispatch task-specific samples
        strategy_order = np.argsort(self.max_tokens_list)
        multi_task_batch_dispatch_map = {task_id : [{s : 0 for s in self.task_seq_lens} for _ in range(self.strategy_num)] for task_id in self.running_task_ids}
        for seq_idx, seq_len in enumerate(self.task_seq_lens):
            for task_id in self.running_task_ids:
                # for i in range(self.strategy_num):
                for i in strategy_order[::-1]:
                    dispatch_num = min(seq_len_num_map_list[i].get(seq_len, 0), seq_distribution_map[task_id].get(seq_len, 0))
                    if dispatch_num == 0:
                        continue
                    multi_task_batch_dispatch_map[task_id][i][seq_len] += int(dispatch_num)
                    seq_len_num_map_list[i][seq_len] -= dispatch_num
                    seq_distribution_map[task_id][seq_len] -= dispatch_num
                if seq_distribution_map[task_id].get(seq_len, 0) > 0:
                    assert seq_idx < len(self.task_seq_lens) - 1
                    if self.task_seq_lens[seq_idx + 1] not in seq_distribution_map[task_id].keys():
                        seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] = 0
                    seq_distribution_map[task_id][self.task_seq_lens[seq_idx + 1]] += seq_distribution_map[task_id][seq_len]

        # profile
        cost_time = 0
        for i in range(self.strategy_num):
            cost_time = max(cost_time, self.get_estimate_total_time(self.tps[i], self.pps[i], multi_task_batch_dispatch_map, i))
        
        return multi_task_batch_dispatch_map, end_time - start_time, cost_time / 1000
