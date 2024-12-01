import os
import bisect
import time
import numpy as np
from trainer.dp_bucket import get_buckets_dp, get_all_buckets_dp
from data_utils.bucket import Bucket

class MicroBatch:
    def __init__(self, batch_data, batch_size, seq_length, batch_offset_list=None, batch_size_list=None):
        self.batch_data = batch_data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_offset_list = batch_offset_list
        self.batch_size_list = batch_size_list
    
    def token_num(self):
        return self.batch_size * self.seq_length

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys() if k != 'batch_data')
    
    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())

    def task_id(self):
        return [i for i, batch_size in enumerate(self.batch_size_list) if batch_size > 0]

def make_micro_batch(batch_size, seq_len, batch_offset_list=None, batch_size_list=None, train_task_num=1):
    if batch_offset_list is None or batch_size_list is None:
        batch_offset_list = [0] * train_task_num
        batch_size_list = [0] * train_task_num
        batch_size_list[0] = batch_size
    batch_data = []
    for _ in range(batch_size):
        batch_data.append([1] * (seq_len + 1))
    batch_data = np.concatenate(batch_data, axis=0)
    return MicroBatch(batch_data, batch_size, seq_len, batch_offset_list, batch_size_list)

def make_micro_batches(batch_size, num_micro_batches, seq_len, train_task_num, batch_offset_list=None, batch_size_list=None):
    micro_batches = []
    for _ in range(num_micro_batches):
        micro_batches.append(make_micro_batch(batch_size, seq_len, batch_offset_list, batch_size_list, train_task_num))
    return micro_batches

def pad_to_max_tokens(run_batches_list, max_tokens, pad_id):
    for run_batches in run_batches_list:
        pad_seq_num = (max_tokens - run_batches.seq_len * run_batches.batch_size) // run_batches.seq_len
        if pad_seq_num == 0:
            continue
        new_run_batch = np.concatenate([run_batches.run_batch, np.array([[pad_id] * (run_batches.seq_len + 1) for _ in range(pad_seq_num)]).reshape(pad_seq_num, -1)], axis=0).reshape(pad_seq_num + run_batches.micro_batch_size, -1)
        run_batches.batch_size += pad_seq_num
        run_batches.run_batch = new_run_batch
    return run_batches_list

def local_batch_pack_scheduler(
    multi_task_local_batch_map,
    train_task_num,
    bucket_sizes,
    pad_id,
    tp,
    fuse_multi_task,
    alignment=128,
):
    buckets = []
    assert len(bucket_sizes) >= 2, 'len(bucket size) must >= 2'
    left, right = 0, 1
    input_bucket = Bucket(pad_id, bucket_sizes[left], bucket_sizes[right], alignment)
    label_bucket = Bucket(pad_id, bucket_sizes[left], bucket_sizes[right], alignment)
    buckets.append((input_bucket, label_bucket))
    # 把multi_task_local_batch_map中每个task的batch拼到local_batch中
    local_batch = None
    task_indices = None
    valid_seq_lens = None
    for task_id in range(train_task_num):
        if task_id not in multi_task_local_batch_map:
            continue
        batch_data = multi_task_local_batch_map[task_id]
        local_batch = np.concatenate([local_batch, batch_data], axis=0) if local_batch is not None else batch_data
        task_indices = np.concatenate([task_indices, np.array([task_id] * batch_data.shape[0])], axis=0) if task_indices is not None else np.array([task_id] * batch_data.shape[0])
        valid_seq_lens_of_task = np.sum(batch_data != pad_id, axis=1)
        # 放到valid_seq_lens
        valid_seq_lens = np.concatenate([valid_seq_lens, valid_seq_lens_of_task], axis=0) if valid_seq_lens is not None else valid_seq_lens_of_task
    # 对local_batch根据valid_seq_lens排序，并将task_indices也排序
    # 1. disaggregated: 先按task_indices排序，再按valid_seq_lens排序，每次只拼单任务的seq
    # 2. fused: 按valid_seq_lens排序，保证每种任务的长度能被tp整除
    if fuse_multi_task:
        sorted_indices = np.argsort(-valid_seq_lens)
    else:
        sorted_indices = np.lexsort((-valid_seq_lens, task_indices))
    sorted_local_batch = local_batch[sorted_indices]
    sorted_task_indices = task_indices[sorted_indices]
    sorted_valid_seq_lens = valid_seq_lens[sorted_indices]
    for seq, valid_seq_len, task_indice in zip(sorted_local_batch, sorted_valid_seq_lens, sorted_task_indices):
        is_add = input_bucket.add_data(seq, valid_seq_len - 1, task_indice) \
            and label_bucket.add_data(seq[1:], valid_seq_len - 1, task_indice)
        while not is_add and right < len(bucket_sizes) - 1:
            left += 1
            right += 1
            input_bucket = Bucket(pad_id, bucket_sizes[left], bucket_sizes[right], alignment)
            label_bucket = Bucket(pad_id, bucket_sizes[left], bucket_sizes[right], alignment)
            buckets.append((input_bucket, label_bucket))
            is_add = input_bucket.add_data(seq, valid_seq_len - 1, task_indice) \
                and label_bucket.add_data(seq[1:], valid_seq_len - 1, task_indice)
    for (input_bucket, label_bucket) in buckets:
        input_bucket.pack_data(batching_option_matrix=None, tp=tp, static_shape=True, train_task_num=train_task_num, fuse_multi_task=fuse_multi_task)
        label_bucket.pack_data(batching_option_matrix=None, tp=tp, static_shape=True, train_task_num=train_task_num, fuse_multi_task=fuse_multi_task)
    return buckets

def greedy_local_batch_scheduler(
    multi_task_local_batch_map,
    seq_len_num_map,
    max_tokens,
    train_task_num,
):
    run_batches_list = []
    micro_batches_list = []
    micro_batches_rest_bucket_map = {}
    for task_id in range(train_task_num):
        if task_id not in multi_task_local_batch_map:
            continue
        batch_data = multi_task_local_batch_map[task_id]
        # print(f"task id {task_id} batch_data shape = {batch_data.shape}, seq_len_num_map = {seq_len_num_map[task_id]}")
        batch_seq_lens = sorted(seq_len_num_map[task_id].keys())
        start_idx = 0
        micro_batches_list_full = []
        for cur_seq_len in batch_seq_lens:
            if seq_len_num_map[task_id][cur_seq_len] == 0:
                continue
            assert cur_seq_len <= max_tokens
            cache_sample_num = seq_len_num_map[task_id][cur_seq_len]
            while cache_sample_num > 0:
                cur_micro_batch_size = min(max_tokens // cur_seq_len, cache_sample_num)
                cur_max_tokens = max_tokens // cur_seq_len * cur_seq_len
                batch_offset_list = [0] * train_task_num
                batch_size_list = [0] * train_task_num
                batch_size_list[task_id] = cur_micro_batch_size
                # print(f"start_idx = {start_idx}, cur_micro_batch_size = {cur_micro_batch_size}, cur_seq_len = {cur_seq_len}")
                micro_batch_data = np.concatenate(batch_data[start_idx: start_idx + cur_micro_batch_size, :cur_seq_len + 1], axis=0).reshape(cur_micro_batch_size, -1)
                assert micro_batch_data.shape[0] == cur_micro_batch_size and micro_batch_data.shape[1] == cur_seq_len + 1
                micro_batch = MicroBatch(micro_batch_data, cur_micro_batch_size, cur_seq_len, batch_offset_list, batch_size_list)
                if cur_micro_batch_size * cur_seq_len == cur_max_tokens:
                    micro_batches_list_full.append(micro_batch)
                else:
                    micro_batches_rest_bucket_map.setdefault(task_id, {}).setdefault(cur_seq_len, []).append(micro_batch)
                start_idx += cur_micro_batch_size
                cache_sample_num -= cur_micro_batch_size
        micro_batches_list.extend(micro_batches_list_full)
    # handle segmented micro batch
    segmented_micro_batches_list = []
    all_seq_len_set = set()
    for _, m in seq_len_num_map.items():
        all_seq_len_set = all_seq_len_set.union(set(m.keys()))
    for seq_len in sorted(list(all_seq_len_set)):
        batches_of_all_tasks = []
        # batch_size_of_cur_seq_len = 0
        for task_id in range(train_task_num):
            if task_id not in micro_batches_rest_bucket_map:
                continue
            if seq_len in micro_batches_rest_bucket_map[task_id].keys():
                assert len(micro_batches_rest_bucket_map[task_id][seq_len]) == 1, \
                    f"only one micro batch, but got {len(micro_batches_rest_bucket_map[task_id][seq_len])}"
                batches_of_all_tasks.extend(micro_batches_rest_bucket_map[task_id][seq_len])
                # batch_size_of_cur_seq_len += np.sum([micro_batch.batch_size for micro_batch in micro_batches_rest_bucket_map[task_id][seq_len]])
        # concatenate micro batches of all tasks
        batches_of_all_tasks = sorted(batches_of_all_tasks, key=lambda x: x.batch_size)
        cur_max_micro_batch_size = max_tokens // seq_len
        acc_micro_batch_size = 0
        concat_batch_data = []
        cur_batch_offset_list = [0] * train_task_num
        cur_batch_size_list = [0] * train_task_num
        for micro_batch in batches_of_all_tasks:
            task_id = micro_batch.task_id()
            assert len(task_id) == 1
            task_id = task_id[0]
            if acc_micro_batch_size + micro_batch.batch_size <= cur_max_micro_batch_size:
                concat_batch_data.append(micro_batch.batch_data)
                cur_batch_offset_list[task_id] = acc_micro_batch_size
                cur_batch_size_list[task_id] = micro_batch.batch_size
                acc_micro_batch_size += micro_batch.batch_size
            elif acc_micro_batch_size + micro_batch.batch_size == cur_max_micro_batch_size:
                concat_batch_data.append(micro_batch.batch_data)
                cur_batch_offset_list[task_id] = acc_micro_batch_size
                cur_batch_size_list[task_id] = micro_batch.batch_size
                concat_micro_batch = MicroBatch(np.concatenate(concat_batch_data, axis=0).reshape(cur_max_micro_batch_size, -1), \
                                                cur_max_micro_batch_size, seq_len, cur_batch_offset_list, cur_batch_size_list)
                micro_batches_list.append(concat_micro_batch)
                acc_micro_batch_size = 0
                concat_batch_data = []
                cur_batch_offset_list = [0] * train_task_num
                cur_batch_size_list = [0] * train_task_num
            else:
                concat_batch_data.append(micro_batch.batch_data[: cur_max_micro_batch_size - acc_micro_batch_size, :])
                cur_batch_offset_list[task_id] = acc_micro_batch_size
                cur_batch_size_list[task_id] = cur_max_micro_batch_size - acc_micro_batch_size
                concat_micro_batch = MicroBatch(np.concatenate(concat_batch_data, axis=0).reshape(cur_max_micro_batch_size, -1), \
                                                cur_max_micro_batch_size, seq_len, cur_batch_offset_list, cur_batch_size_list)
                micro_batches_list.append(concat_micro_batch)
                concat_batch_data = []
                concat_batch_data.append(micro_batch.batch_data[cur_max_micro_batch_size - acc_micro_batch_size:, :])
                acc_micro_batch_size = acc_micro_batch_size + micro_batch.batch_size - cur_max_micro_batch_size
                cur_batch_offset_list = [0] * train_task_num
                cur_batch_size_list = [0] * train_task_num
                cur_batch_size_list[task_id] = acc_micro_batch_size
        if len(concat_batch_data) > 0:
            segmented_micro_batch = MicroBatch(np.concatenate(concat_batch_data, axis=0).reshape(acc_micro_batch_size, -1), \
                                               acc_micro_batch_size, seq_len, cur_batch_offset_list, cur_batch_size_list)
            segmented_micro_batches_list.append(segmented_micro_batch)
    # 直接和打满的一起跑
    micro_batches_list.extend(segmented_micro_batches_list)
    # 对micro_batches_list进行排序，tokens大的排前面
    # micro_batches_list = sorted(micro_batches_list, key=lambda x: dynamic_planner.get_estimated_time(x.batch_size, x.seq_length, tp, pp), reverse=True)
    micro_batches_list = sorted(micro_batches_list, key=lambda x: (x.batch_size * x.seq_length, x.seq_length), reverse=True)
    run_batches_list.append(micro_batches_list)
    return run_batches_list

def global_batch_scheduler(
    args,
    multi_task_global_batch_map,
    train_task_num,
    pad_id,
    dps,
    tps,
    pps,
    local_dp_rank,
    max_tokens,
    num_strategy,
    strategy_id,
    dynamic_planner,
    step,
    is_pack=False,
):
    valid_token_num = 0
    token_num = 0
    seq_len_distribution_map = {}
    # seq_len_num_distribution_map = {}
    seq_len_num_map = {}
    local_batch_map = {}
    global_batch_seqlen_list = []
    for task_id in range(train_task_num):
        if task_id not in multi_task_global_batch_map:
            continue
        padded_global_batch = []
        for batch_data in multi_task_global_batch_map[task_id]:
            if args.max_seq_length - len(batch_data) >= 0:
                padded_global_batch.append(np.concatenate([batch_data, np.array([pad_id] * (args.max_seq_length + 1 - len(batch_data)))]).reshape(-1).tolist())
            else:
                padded_global_batch.append(np.concatenate([batch_data[:args.max_seq_length], np.array([pad_id])]).reshape(-1).tolist())
        multi_task_global_batch_map[task_id] = padded_global_batch
        global_batch = multi_task_global_batch_map[task_id]
        seq_len_distribution = {}
        if args.bucket_num == 7:
            buckets = [256, 512, 1024, 2048, 4096, 8192, 16384]
        elif args.bucket_num == 16:
            buckets = [144, 256, 304, 512, 640, 800, 1024, 1216, 1504, 1888, 2656, 4096, 4256, 5840, 8192, 16384]
        else:
            buckets = [256, 512, 1024, 2048, 4096, 8192, 16384]
        for i, batch_data in enumerate(global_batch):
            effective_len = np.sum(np.array(batch_data) != pad_id, axis=0)
            valid_token_num += effective_len
            global_batch_seqlen_list.append(effective_len)
            # padded_len = max(min(int(np.ceil(effective_len / 128) * 128), args.max_seq_length), args.min_seq_length)
            padded_len = buckets[bisect.bisect_left(buckets, effective_len)]
            token_num += padded_len
            # padded_len = max(min(2 ** int(np.ceil(np.log2(effective_len))), args.max_seq_length), args.min_seq_length)
            seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
        # seq_len_num_distribution = seq_len_distribution.copy()
        # seq_len_num_distribution_map[task_id] = seq_len_num_distribution
        for seq_len, num in seq_len_distribution.items():
            seq_len_distribution[seq_len] = num / len(global_batch)
        seq_len_distribution_map[task_id] = seq_len_distribution
    # print(f"seq_len_distribution_map = {seq_len_distribution_map}")
    if os.environ.get('DP_BUCKET') == 'TRUE':
        valid_token_num = 0
        token_num = 0
        # bucket_candidates = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384]
        # bucket_candidates = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 4096, 4608, 5120, 7168, 8192, 16384]
        bucket_candidates = set()
        # 遍历global_batch_seqlen_list，将每个seq_len向上取到最接近的16的倍数，并加入到bucket_candidates中
        for seq_len in global_batch_seqlen_list:
            bucket_candidates.add(int(np.ceil(seq_len / 16) * 16))
        bucket_candidates = list(bucket_candidates)
        
        bucket_limit = args.bucket_num
        global_batch_seqlen_list = sorted(global_batch_seqlen_list)
        bucket_candidates = sorted(bucket_candidates)
        buckets = get_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
        print(f"buckets from dp: {buckets}")
        # 将多任务的global_batch放入新的分桶中
        remap_seq_len_distribution_map = {}
        # remap_seq_len_num_distribution_map = {}
        for task_id in range(train_task_num):
            if task_id not in multi_task_global_batch_map:
                continue
            global_batch = multi_task_global_batch_map[task_id]
            seq_len_distribution = {}
            for i, batch_data in enumerate(global_batch):
                effective_len = np.sum(np.array(batch_data) != pad_id, axis=0)
                valid_token_num += effective_len
                # padded_len为buckets中最接近的一个数
                padded_len = buckets[bisect.bisect_left(buckets, effective_len)]
                token_num += padded_len
                seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
            # seq_len_num_distribution = seq_len_distribution.copy()
            # remap_seq_len_num_distribution_map[task_id] = seq_len_num_distribution
            for seq_len, num in seq_len_distribution.items():
                seq_len_distribution[seq_len] = num / len(global_batch)
            remap_seq_len_distribution_map[task_id] = seq_len_distribution
        # print(f"remap_seq_len_distribution_map = {remap_seq_len_distribution_map}")
        
        # with open("effectiveness.txt", 'a') as f:
        #     f.write(f"{remap_seq_len_distribution_map}\n")
        multi_task_batch_dispatch_map, schedule_time = dynamic_planner.schedule(remap_seq_len_distribution_map)
        # multi_task_batch_dispatch_map, schedule_time, cost_time = dynamic_planner.schedule_and_profile(remap_seq_len_distribution_map)
        # with open("effectiveness.txt", 'a') as f:
        #     f.write(f"{cost_time}\n")
        #     f.write(f"{schedule_time}\n")
        #     f.write("\n")
        
        # print(f"token_num = {token_num}, valid_token_num = {valid_token_num}, padding ratio = {1 - valid_token_num / token_num}")
        
        # profile
        # cost_time = 0
        # multi_task_batch_dispatch_map, schedule_time, cost_time = dynamic_planner.schedule_and_profile(remap_seq_len_distribution_map)
        # with open("cost_time.txt", "a") as f:
        #     f.write(f"{cost_time}\n")
        # with open("schedule_time.txt", "a") as f:
        #     f.write(f"{schedule_time}\n")
    elif os.environ.get('DP_BUCKET') == 'ITER':
        bucket_candidates = set()
        # 遍历global_batch_seqlen_list，将每个seq_len向上取到最接近的16的倍数，并加入到bucket_candidates中
        for seq_len in global_batch_seqlen_list:
            bucket_candidates.add(int(np.ceil(seq_len / 16) * 16))
        bucket_candidates = list(bucket_candidates)
        
        bucket_init_num = args.bucket_num
        bucket_limit = bucket_init_num * 2
        global_batch_seqlen_list = sorted(global_batch_seqlen_list)
        bucket_candidates = sorted(bucket_candidates)
        s_time = time.time()
        buckets_list, pad_ratio_list = get_all_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
        e_time = time.time()
        bucket_limit_actual = len(buckets_list)
        print(f"get all buckets takes {e_time - s_time:.3f}s, get {bucket_limit_actual} buckets")
        # 遍历每一个bucket，分配，经过动态调度器，得到map, cost和schedule time
        min_cost = 1e8
        multi_task_batch_dispatch_map = {}
        buckets = []
        pad_ratio = 0
        
        ss_time = time.time()
        # 先获取bucket_init对应的解
        buckets_init = buckets_list[bucket_init_num - 1]
        node_limit = 100000
        s_time = time.time()
        remap_seq_len_distribution_map = {}
        for task_id in range(train_task_num):
            if task_id not in multi_task_global_batch_map:
                continue
            global_batch = multi_task_global_batch_map[task_id]
            seq_len_distribution = {}
            for batch_data in global_batch:
                effective_len = np.sum(np.array(batch_data) != pad_id, axis=0)
                padded_len = buckets_init[bisect.bisect_left(buckets_init, effective_len)]
                seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
            for seq_len, num in seq_len_distribution.items():
                seq_len_distribution[seq_len] = num / len(global_batch)
            remap_seq_len_distribution_map[task_id] = seq_len_distribution
        multi_task_batch_dispatch_map, schedule_time, min_cost_time = dynamic_planner.schedule_and_profile(remap_seq_len_distribution_map, node_limit)        
        pad_ratio = pad_ratio_list[bucket_init_num - 1]
        final_buckets = buckets_init
        e_time = time.time()
        print(f"dispatch of initial bucket takes {e_time - s_time:.3f}s")
        with open("cost_time.txt", "a") as f:
            f.write(f"init bucket : {min_cost_time}\n")
        with open("schedule_time.txt", "a") as f:
            f.write(f"init bucket : {schedule_time}\n")

        # 再遍历一个范围，找更快的解
        node_limit = 100
        r = bucket_init_num // 4
        pending_bucket_idxs = []
        s_time = time.time()
        for i in range((bucket_init_num - 1) - r, (bucket_init_num - 1) + r + 1):
            if i == bucket_init_num - 1:
                continue
            buckets = buckets_list[i]
            # 将多任务的global_batch放入新的分桶中
            remap_seq_len_distribution_map = {}
            for task_id in range(train_task_num):
                if task_id not in multi_task_global_batch_map:
                    continue
                global_batch = multi_task_global_batch_map[task_id]
                seq_len_distribution = {}
                for batch_data in global_batch:
                    effective_len = np.sum(np.array(batch_data) != pad_id, axis=0)
                    padded_len = buckets[bisect.bisect_left(buckets, effective_len)]
                    seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
                for seq_len, num in seq_len_distribution.items():
                    seq_len_distribution[seq_len] = num / len(global_batch)
                remap_seq_len_distribution_map[task_id] = seq_len_distribution
            _, schedule_time, cost_time = dynamic_planner.schedule_and_profile(remap_seq_len_distribution_map, node_limit)
            e_time = time.time()
            # profile
            with open("cost_time.txt", "a") as f:
                f.write(f"bucket {i} : {cost_time}\n")
            with open("schedule_time.txt", "a") as f:
                f.write(f"bucket {i} : {schedule_time}\n")
            if cost_time < min_cost_time:
                pending_bucket_idxs.append(i)
        e_time = time.time()
        print(f"coarse grained search takes {e_time - s_time:.3f}s")

        # 取pending_buckets_idxs的top-k个
        k = min(3, len(pending_bucket_idxs))
        node_limit = 100000
        pending_bucket_idxs = sorted(pending_bucket_idxs)
        min_cost = min_cost_time
        for i in pending_bucket_idxs[:k]:
            buckets = buckets_list[i]
            # 将多任务的global_batch放入新的分桶中
            remap_seq_len_distribution_map = {}
            for task_id in range(train_task_num):
                if task_id not in multi_task_global_batch_map:
                    continue
                global_batch = multi_task_global_batch_map[task_id]
                seq_len_distribution = {}
                for batch_data in global_batch:
                    effective_len = np.sum(np.array(batch_data) != pad_id, axis=0)
                    padded_len = buckets[bisect.bisect_left(buckets, effective_len)]
                    seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
                for seq_len, num in seq_len_distribution.items():
                    seq_len_distribution[seq_len] = num / len(global_batch)
                remap_seq_len_distribution_map[task_id] = seq_len_distribution
            cur_multi_task_batch_dispatch_map, schedule_time, cost_time = dynamic_planner.schedule_and_profile(remap_seq_len_distribution_map, node_limit)
            print(f"fine grained bucket {i}: cost time = {cost_time}, pad ratio = {pad_ratio_list[i]}")
            if cost_time < min_cost_time:
                multi_task_batch_dispatch_map = cur_multi_task_batch_dispatch_map
                min_cost = cost_time
                pad_ratio = pad_ratio_list[i]
                final_buckets = buckets_list[i]
        ee_time = time.time()
        # print(f"profile_cost_time_list = {profile_cost_time_list}")
        # print(f"pad_ratio_list = {profile_pad_ratio_list}")
        print(f"final buckets = {final_buckets}, min_cost = {min_cost}s, pad_ratio = {pad_ratio}, time = {ee_time - ss_time:.3f}s")
    else:
        # with open("effectiveness.txt", 'a') as f:
        #     f.write(f"{seq_len_distribution_map}\n")
        multi_task_batch_dispatch_map, schedule_time = dynamic_planner.schedule(seq_len_distribution_map)
        # multi_task_batch_dispatch_map, schedule_time, cost_time = dynamic_planner.schedule_and_profile(seq_len_distribution_map)
        # with open("effectiveness.txt", 'a') as f:
        #     f.write(f"{cost_time}\n")
        #     f.write(f"{schedule_time}\n")
        #     f.write("\n")
    
    if not multi_task_batch_dispatch_map:
        return [], schedule_time
    # 统计token
    # print(f"multi_task_batch_dispatch_map = {multi_task_batch_dispatch_map}")
    if step > 0:
        dynamic_planner.token_num += token_num
        dynamic_planner.valid_token_num += valid_token_num
    '''
    if multi_task_batch_dispatch_map and step > 0:
        token_num = 0
        for task_id in range(train_task_num):
            if task_id not in multi_task_batch_dispatch_map:
                continue
            for i in range(num_strategy):
                for seq_len, num in multi_task_batch_dispatch_map[task_id][i].items():
                    token_num += seq_len * num
        dynamic_planner.token_num += token_num
        dynamic_planner.valid_token_num += valid_token_num
    '''

    # Get local batch for current strategy
    for task_id in range(train_task_num):
        if task_id not in multi_task_global_batch_map:
            continue
        # print(f"task_id = {task_id}")
        global_batch = multi_task_global_batch_map[task_id]
        global_batch = sorted(global_batch, key=lambda x: np.sum(np.array(x) != pad_id, axis=0))
        # print(f"global batch len: {len(global_batch)}")
        # for sample in global_batch:
        #     print(f"sample: {np.sum(np.array(sample) != pad_id, axis=0)}")
        seq_len_num_map_list = multi_task_batch_dispatch_map[task_id]
        seq_len_list = []
        local_seq_len_num_map_list = []
        for i in range(num_strategy):
            local_seq_len_num_map_list.append({s : 0 for s in seq_len_num_map_list[0].keys()})
        # 假设每个策略的key都是一样的
        seq_len_num_map_keys = sorted(list(seq_len_num_map_list[0].keys()))
        local_batch_dps = []
        start_idx = 0
        gbs_per_dp = 0
        for seq_len in seq_len_num_map_keys:
            # print(f"seq_len = {seq_len}")
            for i in range(num_strategy):
                if i == strategy_id and seq_len_num_map_list[i][seq_len] > 0:
                    # print(f"start_idx = {start_idx}, increase_num = {seq_len_num_map_list[i][seq_len]}")
                    cur_local_batch_dp = np.array(global_batch[start_idx: start_idx + seq_len_num_map_list[i][seq_len]]) \
                                                  .reshape(seq_len_num_map_list[i][seq_len], -1)
                    local_batch_dps.append(cur_local_batch_dp)
                    gbs_per_dp += seq_len_num_map_list[i][seq_len]
                    seq_len_list.extend([seq_len for _ in range(seq_len_num_map_list[i][seq_len])])
                start_idx += int(seq_len_num_map_list[i][seq_len])
        if local_batch_dps == []:
            continue
        local_batch_dps = np.concatenate(local_batch_dps, axis=0).reshape(gbs_per_dp, -1).tolist()
        local_batch_dps = np.array(sorted(local_batch_dps, key=lambda x: np.sum(np.array(x) != pad_id, axis=0)))
        local_batch = None
        if local_dp_rank == -1:
            if len(local_batch_dps[0: gbs_per_dp:dps[strategy_id], :]) > 0:
                local_batch = np.concatenate(local_batch_dps[0: gbs_per_dp:dps[strategy_id], :], axis=0)
                seq_len_list = seq_len_list[0: gbs_per_dp:dps[strategy_id]]
        else:
            if len(local_batch_dps[local_dp_rank: gbs_per_dp:dps[strategy_id], :]) > 0:
                local_batch = np.concatenate(local_batch_dps[local_dp_rank: gbs_per_dp:dps[strategy_id], :], axis=0)
                seq_len_list = seq_len_list[local_dp_rank: gbs_per_dp:dps[strategy_id]]
        for seq_len in seq_len_list:
            local_seq_len_num_map_list[strategy_id][seq_len] = local_seq_len_num_map_list[strategy_id].get(seq_len, 0) + 1
        task_seq_len_num_map = local_seq_len_num_map_list[strategy_id]
        seq_len_num_map[task_id] = task_seq_len_num_map
        if local_batch is not None:
            local_batch = local_batch.reshape(-1, args.max_seq_length + 1)
            local_batch_map[task_id] = local_batch
        # print(f"local_batch_map {task_id} shape = {local_batch.shape}")
    if is_pack:
        return local_batch_pack_scheduler(local_batch_map, train_task_num, bucket_sizes=(max_tokens, 0), tp=tps[strategy_id], pad_id=pad_id, fuse_multi_task=False), schedule_time
    else:
        return greedy_local_batch_scheduler(local_batch_map, seq_len_num_map, max_tokens, train_task_num), schedule_time
