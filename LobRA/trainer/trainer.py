import os
import time
import signal
import hetu as ht
import numpy as np
from types import SimpleNamespace
from .utils import DatasetWrapper, ModelWrapper, OptimizerWrapper, TrainerConfig
from .batch_scheduler import global_batch_scheduler, make_micro_batch
from utils import parse_multi_ds_parallel_config, get_ds_parallel_degrees, write_to_csv, read_from_csv
from data_utils import build_bucket_global_data_loader, Encoder
from .planner import NewDynamicBatchPlanner, GroupedDynamicBatchPlanner

class DatasetContext:
    def __init__(
        self,
        dataset,
        steps,
        epochs
    ):
        self.dataset = dataset
        self.consumed_samples = 0
        self.steps = steps
        self.epochs = epochs
        self.step = 0
        self.epoch = 0

class Trainer:
    def __init__(
        self,
        args,
        dataset_wrapper: DatasetWrapper,
        pretrained_model_wrapper: ModelWrapper,
        finetune_model_wrapper: ModelWrapper,
        optimizer_wrapper: OptimizerWrapper,
        trainer_config: TrainerConfig,
        cost_model,
        ds_parallel_configs,
        max_tokens_list=None,
        dataset_ctxs=None,
        is_pack=False,
    ):
        self.dataset_wrapper = dataset_wrapper
        self.pretrained_model_wrapper = pretrained_model_wrapper
        self.finetune_model_wrapper = finetune_model_wrapper
        self.optimizer_wrapper = optimizer_wrapper
        self.ds_parallel_configs = ds_parallel_configs
        self.trainer_config = trainer_config
        self.cost_model = cost_model
        
        self.dataset_ctxs = dataset_ctxs
        self.train_dataset_pool = {}
        self.build_ops = None
        self.pad_id = None
        self.max_epochs = 0
        self.max_steps = 0
        self.is_pack = is_pack
        
        # ds parallel config
        self.num_strategy, self.dps, self.tps, self.pps, self.sp, self.num_gpus = get_ds_parallel_degrees(ds_parallel_configs)
        if max_tokens_list is not None:
            self.max_tokens_list = max_tokens_list
        else:
            self.max_tokens_list = list(map(int, args.max_tokens.split(','))) if isinstance(args.max_tokens, str) else args.max_tokens
        self.train_task_num = trainer_config.train_task_num
        self.precision = None
        
        # logging
        self.total_tokens = 0
        self.valid_tokens = 0
        self.total_run_times = []
        self.schedule_times = []
        self.dp_grad_reduce_times = []
        
        # padding debug
        self.valid_bucket = {}
        self.total_bucket = {}

    def create_dataset(self, args):
        """Build train dataset."""
        if self.dataset_ctxs is not None:
            self.pad_id = self.dataset_ctxs[0].dataset.encoder.pad_id()
            return
        self.dataset_ctxs = []
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
        for i in range(self.trainer_config.train_task_num):
            task_config = self.trainer_config.task_configs[i]
            if task_config.dataset_name == "" or os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE' and i > 0:
                train_dataset = None
            elif self.train_dataset_pool.get((task_config.dataset_name, task_config.context_length), None) is not None:
                train_dataset = self.train_dataset_pool[(task_config.dataset_name, task_config.context_length)]
            else:
                train_dataset = self.dataset_wrapper.create_dataset(
                    dataset_name=task_config.dataset_name,
                    key=task_config.json_key,
                    max_seq_len=task_config.context_length,
                    vocab_file=args.vocab_file,
                    merge_file=args.merge_file,
                    encoder=encoder)
                self.train_dataset_pool[(task_config.dataset_name, task_config.context_length)] = train_dataset
            dataset_ctx = DatasetContext(
                dataset=train_dataset,
                steps=task_config.steps,
                epochs=task_config.epochs
            )
            self.max_epochs = max(self.max_epochs, task_config.epochs)
            self.max_steps = max(self.max_steps, task_config.steps)
            self.dataset_ctxs.append(dataset_ctx)
        self.pad_id = self.dataset_ctxs[0].dataset.encoder.pad_id()

    def train_data_iterator(self, dataset, consumed_samples, gbs, min_seq_length=256, max_seq_length=8192):
        train_dataloader = build_bucket_global_data_loader(dataset, consumed_samples, gbs, min_seq_length, max_seq_length)
        train_data_iterator = iter(train_dataloader)
        return train_data_iterator

    def get_custom_global_batch(self, seq_distribution, max_seq_len, pad_id):
        global_batch = []
        for seq_len, num in seq_distribution.items():
            for _ in range(num):
                global_batch.append([1] * (seq_len) + [pad_id] * (max_seq_len + 1 - seq_len))
        return global_batch

    def build_model(self, args, ds_parallel_configs):
        # Build dataset
        self.create_dataset(args)
        print("create dataset done")
        # Build model
        # '''
        with ht.graph("define_and_run", num_strategy=self.num_strategy):
            if args.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            self.precision = eval(precision)
            with ht.autocast(eval(precision)):
                self.create_define_and_run_graph(args, ds_parallel_configs)
        # '''
    
    def create_define_and_run_graph(self, args, ds_parallel_configs):
        # 获取ds
        input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        task_multi_ds, task_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
        
        # 获取默认的seq_len, dp, mbs_times_dp, batch_offset, batch_size
        default_seq_len = args.max_seq_length
        default_dp = input_multi_ds[0].get_dim(0)
        default_mbs_times_dp = default_dp
        dummy_size = default_mbs_times_dp * default_seq_len
        default_batch_offset = 0
        default_batch_size = 1
        
        # 构建placeholder
        if self.is_pack:
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
            position_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
            # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')
        else:
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
            position_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
            # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')
        task_batch_idxs = []
        for i in range(self.train_task_num):
            task_batch_idxs.append(ht.parallel_placeholder(ht.int64, global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                                                           multi_ds=task_multi_ds, device_groups=task_device_groups, name='task_batch_idxs_task{}'.format(i), is_cpu=True))

        # 设置symbolic shape
        self.pretrained_model_wrapper.model_config.dp_symbol.set_data(default_dp)
        self.pretrained_model_wrapper.model_config.train_task_num = self.trainer_config.train_task_num
        # 创建预训练模型
        pretrained_model = self.pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
        self.pretrained_model = pretrained_model
        # 创建微调模型
        peft_configs = [task_config.lora_config for task_config in self.trainer_config.task_configs]
        model = self.finetune_model_wrapper.create_model(model=pretrained_model, peft_configs=peft_configs)
        self.model = model
        cu_seqlens_list = []
        if self.is_pack:
            for block_id, block in enumerate(pretrained_model.transformer.h):
                cu_seqlens_list.append(
                    ht.parallel_placeholder(
                        ht.int32,
                        global_shape=[dummy_size], 
                        multi_ds=block.attn.q_proj.base_layer.ds_map['split0_dup'], 
                        device_groups=block.attn.q_proj.base_layer.device_groups,
                        name=f'cu_seqlens_{block_id}'
                    )
                )
            model.config.max_seqlen_symbol = ht.IntSymbol(1)
        print(f"start to build model...")
        # 构建静态图
        if self.is_pack:
            loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                # token_type_ids=None,
                labels=masked_lm_labels,
                task_batch_idxs=task_batch_idxs,
                cu_seqlens_list=cu_seqlens_list
            )
        else:
            loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                # token_type_ids=None,
                labels=masked_lm_labels,
                task_batch_idxs=task_batch_idxs,
            )
        print(f"build model end...")
        # build optimizer
        opt = self.optimizer_wrapper.create_optimizer(lr=args.lr)
        train_op = opt.minimize(loss)
        # build ops
        self.build_ops = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            # 'token_type_ids': None,
            'masked_lm_labels': masked_lm_labels,
            'task_batch_idxs': task_batch_idxs,
            'cu_seqlens_list': cu_seqlens_list,
            'loss': loss,
            'train_op': train_op
        }
    
    def packed_run(self, args):
        # 确定当前的strategy_id
        strategy_id = 0 # default
        device_groups = []
        local_device = ht.local_device()
        all_devices = ht.global_device_group()
        for config in self.ds_parallel_configs:
            device_groups.append(ht.DeviceGroup([all_devices.get(device_id) for device_id in config['devices']]))
        for i, device_group in enumerate(device_groups):
            if device_group.contains(local_device):
                strategy_id = i
                break
        print(f"strategy_id = {strategy_id}")
        # 获取ds，进而得到dp num和rank
        input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'input')
        label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'label')
        input_ds = input_multi_ds[strategy_id]
        input_device_group = input_device_groups[strategy_id]
        label_ds = label_multi_ds[strategy_id]
        label_device_group = label_device_groups[strategy_id]
        dup_group_idx = -1
        for i in range(self.pps[strategy_id]):
            _, block_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'qkv', args.num_layers // self.pps[strategy_id] * i)
            block_device_group = block_device_groups[strategy_id]
            if not block_device_group.contains(local_device):
                continue
            local_device_idx = block_device_group.get_index(local_device)
            dup_group_idx = input_ds.get_dup_group_index(local_device_idx)
            if dup_group_idx != -1:
                break
        assert dup_group_idx != -1, 'local device should belong to one dup group'
        local_dp_rank = -1
        if input_device_group.contains(local_device):
            local_device_idx = input_device_group.get_index(local_device)
            local_dp_rank = input_ds.get_dup_group_index(local_device_idx)
        elif label_device_group.contains(local_device):
            local_device_idx = label_device_group.get_index(local_device)
            local_dp_rank = label_ds.get_dup_group_index(local_device_idx)
        global_dp_rank = strategy_id
        # dup_group_idx = local_dp_rank
        self.finetune_model_wrapper.model_config.dp_symbol.set_data(self.dps[global_dp_rank])
        self.model.config.max_seqlen_symbol.set_data(self.max_tokens_list[strategy_id])
        # train_iter
        train_iter_list = []
        train_task_num = self.trainer_config.train_task_num
        for i in range(train_task_num):
            task_ctx = self.dataset_ctxs[i]
            if task_ctx.dataset is None:
                train_iter_list.append(None)
            else:
                train_iter_list.append(self.train_data_iterator(task_ctx.dataset, task_ctx.consumed_samples,
                                                                self.trainer_config.task_configs[i].global_batch_size,
                                                                args.min_seq_length, args.max_seq_length))
        if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
            # task_seq_len_distribution = {0 : {256: 0, 512: 0, 1024: 5, 2048: 0, 4096: 0, 8192: 0}, \
            #                              1 : {256: 0, 512: 0, 1024: 3, 2048: 0, 4096: 0, 8192: 0}, \
            #                              2 : {256: 0, 512: 0, 1024: 32, 2048: 0, 4096: 0, 8192: 0}, \
            #                              3 : {256: 0, 512: 33, 1024: 4, 2048: 0, 4096: 0, 8192: 0}, \
            #                              4 : {256: 0, 512: 12, 1024: 4, 2048: 0, 4096: 0, 8192: 0}}
            task_seq_len_distribution = {0 : {256: 6, 512: 13, 1024: 0, 2048: 0, 4096: 0, 8192: 0}}
            global_batch_size_list = [sum(task_seq_len_distribution[task_id].values()) for task_id in sorted(task_seq_len_distribution.keys())]
        else:
            task_seq_len_distribution = None
            global_batch_size_list = [task_config.global_batch_size for task_config in self.trainer_config.task_configs]
        # Dynamic Planner
        data_dispatch_pattern = os.environ.get('HETU_DATA_DISPATCH')
        print(f"create dynamic {data_dispatch_pattern} batch planner...")
        if data_dispatch_pattern == 'GROUP':
            dynamic_planner = GroupedDynamicBatchPlanner(self.cost_model, args.num_layers, train_task_num, global_batch_size_list,
                                                         self.num_strategy, self.max_tokens_list, self.dps, self.tps, self.pps, local_device=local_device.index)
        elif data_dispatch_pattern == 'BALANCE':
            dynamic_planner = NewDynamicBatchPlanner(self.cost_model, args.num_layers, train_task_num, global_batch_size_list,
                                                     self.num_strategy, self.max_tokens_list, self.dps, self.tps, self.pps, local_device=local_device.index)
        else:
            pass
            # dynamic_planner = DynamicBatchPlanner(self.cost_model, args.num_layers, train_task_num, global_batch_size_list,
            #                                       self.num_strategy, self.max_tokens_list, self.dps, self.tps, self.pps)
        print(f"strategy {strategy_id} start to train...")
        # 训练        
        for epoch in range(self.max_epochs):
            for step in range(self.max_steps):
                multi_task_global_batch_map = {}
                # task_seq_len_distribution = None
                # if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
                #     task_seq_len_distribution = {0 : {256: 7, 512: 16, 1024: 32, 2048: 9}}
                for task_id in range(train_task_num):
                    if self.dataset_ctxs[task_id].step >= self.dataset_ctxs[task_id].steps or \
                        self.dataset_ctxs[task_id].epoch >= self.dataset_ctxs[task_id].epochs:
                        continue
                    if train_iter_list[task_id] is None and os.environ.get('CUSTOM_DISTRIBUTION') != 'TRUE':
                        continue
                    else:
                        if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
                            # seq_len_distribution = {2048: 88, 4096: 24, 8192: 12, 16384: 4}
                            # seq_len_distribution = {2048: 196, 4096: 62, 8192: 16, 16384: 4}
                            # seq_len_distribution = {2048: 102, 4096: 32, 8192: 8, 16384: 2}
                            # new
                            # seq_len_distribution = {2048: 204, 4096: 64, 8192: 16, 16384: 4}
                            # seq_len_distribution = {2048: 48, 4096: 16, 8192: 10, 16384: 2}
                            # seq_len_distribution = {256: 7, 512: 16, 1024: 32, 2048: 9}
                            seq_len_distribution = task_seq_len_distribution[task_id]
                            # seq_len_distribution = {256: 128}
                            global_batch = self.get_custom_global_batch(seq_len_distribution, args.max_seq_length, self.pad_id)
                        else:
                            try:
                                global_batch = next(train_iter_list[task_id])
                            except StopIteration:
                                train_iter_list[task_id] = self.train_data_iterator(self.dataset_ctxs[task_id].dataset, 0,
                                                                                    self.trainer_config.task_configs[task_id].global_batch_size,
                                                                                    args.min_seq_length, args.max_seq_length)
                                global_batch = next(train_iter_list[task_id])
                    multi_task_global_batch_map[task_id] = global_batch
                    self.dataset_ctxs[task_id].consumed_samples += len(global_batch)
                buckets, schedule_time = global_batch_scheduler(args, multi_task_global_batch_map, train_task_num, \
                                                                         self.pad_id, self.dps, self.tps, self.pps, dup_group_idx, self.max_tokens_list[strategy_id], \
                                                                         self.num_strategy, strategy_id, dynamic_planner, is_pack=self.is_pack)
                self.schedule_times.append(schedule_time)
                if os.environ.get('COST_MODEL_ESTIMATE') == 'TRUE':
                    continue
                input_bucket, label_bucket = buckets[0]
                if local_dp_rank != -1 and (step > 0 or epoch > 0):
                    for packed_cu_seq_lens in input_bucket.packed_cu_seqlens_list():
                        self.total_tokens += input_bucket.max_seq_len()
                        self.valid_tokens += packed_cu_seq_lens[-1]
                if os.environ.get('GET_TOKENS') == 'TRUE':
                    self.total_run_times.append(0)
                    continue
                
                # prepare feed_dict
                num_micro_batches = len(input_bucket.packed_batch())
                task_batch_idxs_list = [[] for _ in range(train_task_num)]
                cu_seqlens_list_batch = input_bucket.packed_cu_seqlens_list()
                task_seq_lens_list = input_bucket.packed_task_seqlens_list()
                if dup_group_idx == -1:
                    input_ids_list = [np.zeros([input_bucket.max_seq_len()]).astype(np.int64) for _ in range(num_micro_batches)]
                    masked_lm_labels_list = [np.zeros([input_bucket.max_seq_len()]).astype(np.int64) for _ in range(num_micro_batches)]
                else:
                    input_ids_list = [micro_batch.astype(np.int64) for micro_batch in input_bucket.packed_batch()]
                    masked_lm_labels_list = [micro_batch.astype(np.int64) for micro_batch in label_bucket.packed_batch()]
                feed_dict = {
                    self.build_ops['input_ids']: input_ids_list,
                    self.build_ops['masked_lm_labels']: masked_lm_labels_list,
                }
                for i in range(self.pretrained_model.config.n_layer):
                    feed_dict[self.build_ops['cu_seqlens_list'][i]] = [x.astype(np.int32) for x in cu_seqlens_list_batch]
                    feed_dict[self.build_ops['cu_seqlens_list'][i]] = [x.astype(np.int32) for x in cu_seqlens_list_batch]
                # print(f"task_seq_lens_list = {task_seq_lens_list}")
                for task_seq_lens in task_seq_lens_list:
                    batch_offset = 0
                    for task_id in range(train_task_num):
                        task_batch_idxs = np.zeros([batch_offset, task_seq_lens[task_id], input_bucket.max_seq_len()], dtype=np.int64)
                        task_batch_idxs_list[task_id].append(task_batch_idxs)
                        batch_offset += task_seq_lens[task_id]
                for i in range(train_task_num):
                    feed_dict[self.build_ops['task_batch_idxs'][i]] = task_batch_idxs_list[i]
                # run
                iter_time = 0
                run_level = ht.run_level("update")
                start_time = time.time()
                try:
                    if os.environ.get('PROFILE_DYNAMIC_PLANNER') == 'TRUE':
                        with ht.autocast(self.precision):
                            with ht.profiler(enabled=True, record_shapes=False) as profiler:
                                results = self.build_ops['train_op'].graph.run(
                                    self.build_ops['loss'],
                                    [self.build_ops['loss'], self.build_ops['train_op']],
                                    feed_dict=feed_dict,
                                    num_micro_batches = num_micro_batches,
                                    cur_strategy_id = strategy_id,
                                    run_level = run_level,
                                    grad_scale = 1.0)
                        e2e_time = float(profiler.summary()['graph_view'][0][1])
                        total_steam_time = float(profiler.summary()['graph_view'][11][1])
                        dp_grad_reduce_time = float(profiler.summary()['graph_view'][5][1])
                        self.dp_grad_reduce_times.append(dp_grad_reduce_time)
                        print(f"strategy {strategy_id} - e2e_time = {e2e_time / 1000:.4f} s")
                        print(f"strategy {strategy_id} - total_steam_time = {total_steam_time / 1000:.4f} s")
                        print(f"strategy {strategy_id} - dp_grad_reduce_time = {dp_grad_reduce_time / 1000:.4f} s")
                    else:
                        with ht.autocast(self.precision):
                            results = self.build_ops['train_op'].graph.run(
                                self.build_ops['loss'],
                                [self.build_ops['loss'], self.build_ops['train_op']],
                                feed_dict=feed_dict,
                                num_micro_batches = num_micro_batches,
                                cur_strategy_id = strategy_id,
                                run_level = run_level,
                                grad_scale = 1.0)
                except RuntimeError as e:
                    print(e)
                    os.killpg(0, signal.SIGTERM)
                end_time = time.time()
                iter_time += end_time - start_time
                if (step > 0 or epoch > 0) and run_level == ht.run_level("update"):
                    self.total_run_times.append(iter_time)
                # TODO: consumed samples of each task
                if run_level == ht.run_level("update"):
                    if label_device_group.contains(local_device):
                        loss_out = results[0].numpy(force=True).mean()
                        consumed_samples = np.sum(self.dataset_ctxs[i].consumed_samples for i in range(train_task_num))
                        print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {iter_time:.4f}")
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
        tokens_entry = {
            'dp': self.dps[strategy_id],
            'tp': self.tps[strategy_id],
            'pp': self.pps[strategy_id],
            'total_tokens': self.total_tokens,
            'valid_tokens': self.valid_tokens
        }
        write_to_csv(tokens_entry, f"temp/tokens_{self.num_strategy}_{strategy_id}_{local_host_name}_{local_device.index}")
        if label_device_group.contains(local_device):
            total_run_time = np.mean(self.total_run_times)
            schedule_time = np.mean(self.schedule_times)
            run_time_entry = {
                'dp': self.dps[global_dp_rank],
                'tp': self.tps[strategy_id],
                'pp': self.pps[strategy_id],
                'total_run_time': total_run_time,
                'schedule_time': schedule_time
            }
            write_to_csv(run_time_entry, f"temp/run_time_{self.num_strategy}_{local_host_name}_{local_device.index}")
        if local_device.index == 0 and local_host_name == 'worker-0':
            print(f"handler: {local_host_name}")
            time.sleep(180)
            total_cnt = 0
            valid_cnt = 0
            for i in range(self.num_strategy):
                total_cnt_strategy = 0
                valid_cnt_strategy = 0
                token_file_names = [f for f in os.listdir('temp') if f.startswith(f"tokens_{self.num_strategy}_{i}")]
                for token_file_name in token_file_names:
                    rows = read_from_csv(f"temp/{token_file_name}")
                    if len(rows) == 0:
                        continue
                    total_cnt_strategy += np.sum([row['total_tokens'] for row in rows])
                    valid_cnt_strategy += np.sum([row['valid_tokens'] for row in rows])
                    os.remove(f"temp/{token_file_name}")
                total_cnt += (total_cnt_strategy // (self.tps[i] * min(2, self.pps[i])))
                valid_cnt += (valid_cnt_strategy // (self.tps[i] * min(2, self.pps[i])))
            run_times = []
            schedule_times = []
            run_time_file_names = [f for f in os.listdir('temp') if f.startswith(f"run_time_{self.num_strategy}")]
            for run_time_file_name in run_time_file_names:
                rows = read_from_csv(f"temp/{run_time_file_name}")
                if len(rows) == 0:
                    continue
                run_times.append(rows[0]['total_run_time'])
                schedule_times.append(rows[0]['schedule_time'])
                os.remove(f"temp/{run_time_file_name}")
            log_entry = {
                'dp': self.dps,
                'tp': self.tps,
                'pp': self.pps,
                'max_tokens': self.max_tokens_list,
                'total_tokens': total_cnt,
                'effective_tokens': valid_cnt,
                'padding_ratio': (total_cnt - valid_cnt) / total_cnt,
                'run_time': np.min(run_times),
                'schedule_time': np.min(schedule_times)
            }
            if os.environ['PROFILE_E2E_COST'] == 'TRUE':
                write_to_csv(log_entry, f"exp_result/e2e/run_statistics.csv")
            # os.system("rm -rf temp")
            print(f"total_cnt = {total_cnt}, valid_cnt = {valid_cnt}, mean_run_time = {np.min(run_times)}, mean_schedule_time = {np.min(schedule_times)}")
        # print(f"{local_host_name} - {local_device.index}: dp_grad_reduce_time = {np.mean(self.dp_grad_reduce_times)}")
    
    def run(self, args):
        # 确定当前的strategy_id
        strategy_id = 0 # default
        # '''
        device_groups = []
        local_device = ht.local_device()
        all_devices = ht.global_device_group()
        for config in self.ds_parallel_configs:
            device_groups.append(ht.DeviceGroup([all_devices.get(device_id) for device_id in config['devices']]))
        for i, device_group in enumerate(device_groups):
            if device_group.contains(local_device):
                strategy_id = i
                break
        print(f"strategy_id = {strategy_id}")
        # 获取ds，进而得到dp num和rank
        input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'input')
        label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'label')
        input_ds = input_multi_ds[strategy_id]
        input_device_group = input_device_groups[strategy_id]
        label_ds = label_multi_ds[strategy_id]
        label_device_group = label_device_groups[strategy_id]
        dup_group_idx = -1
        for i in range(self.pps[strategy_id]):
            _, block_device_groups = parse_multi_ds_parallel_config(self.ds_parallel_configs, 'qkv', args.num_layers // self.pps[strategy_id] * i)
            block_device_group = block_device_groups[strategy_id]
            if not block_device_group.contains(local_device):
                continue
            local_device_idx = block_device_group.get_index(local_device)
            dup_group_idx = input_ds.get_dup_group_index(local_device_idx)
            if dup_group_idx != -1:
                break
        assert dup_group_idx != -1, 'local device should belong to one dup group'
        local_dp_rank = -1
        if input_device_group.contains(local_device):
            local_device_idx = input_device_group.get_index(local_device)
            local_dp_rank = input_ds.get_dup_group_index(local_device_idx)
        elif label_device_group.contains(local_device):
            local_device_idx = label_device_group.get_index(local_device)
            local_dp_rank = label_ds.get_dup_group_index(local_device_idx)
        global_dp_rank = strategy_id
        # dup_group_idx = local_dp_rank
        self.finetune_model_wrapper.model_config.dp_symbol.set_data(self.dps[global_dp_rank])
        # '''
        # train_iter
        train_iter_list = []
        train_task_num = self.trainer_config.train_task_num
        for i in range(train_task_num):
            task_ctx = self.dataset_ctxs[i]
            if task_ctx.dataset is None:
                train_iter_list.append(None)
            else:
                train_iter_list.append(self.train_data_iterator(task_ctx.dataset, task_ctx.consumed_samples,
                                                                self.trainer_config.task_configs[i].global_batch_size,
                                                                args.min_seq_length, args.max_seq_length))
        if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
            task_seq_len_distribution = {0 : {8192: 14, 16384: 2}}
            global_batch_size_list = [sum(task_seq_len_distribution[task_id].values()) for task_id in sorted(task_seq_len_distribution.keys())]
        else:
            task_seq_len_distribution = None
            global_batch_size_list = [task_config.global_batch_size for task_config in self.trainer_config.task_configs]
        # Dynamic Planner
        data_dispatch_pattern = os.environ.get('HETU_DATA_DISPATCH')
        print(f"create dynamic {data_dispatch_pattern} batch planner...")
        if data_dispatch_pattern == 'GROUP':
            dynamic_planner = GroupedDynamicBatchPlanner(self.cost_model, args.num_layers, train_task_num, global_batch_size_list,
                                                         self.num_strategy, self.max_tokens_list, self.dps, self.tps, self.pps, local_device=local_device.index, cur_strategy_id=strategy_id)
        elif data_dispatch_pattern == 'BALANCE':
            dynamic_planner = NewDynamicBatchPlanner(self.cost_model, args.num_layers, train_task_num, global_batch_size_list,
                                                     self.num_strategy, self.max_tokens_list, self.dps, self.tps, self.pps, local_device=local_device.index, cur_strategy_id=strategy_id)
        else:
            pass
        if (os.environ.get('DP_BUCKET') == 'TRUE' or os.environ.get('DP_BUCKET') == 'ITER') and \
            os.environ.get('COST_MODEL_ESTIMATE') != 'TRUE' and os.environ.get('GET_TOKENS') != 'TRUE':
            print(f"{local_device}: warmup begin...")
            warmup_step = 5
            for _ in range(warmup_step):
                num_micro_batches = 2
                warmup_micro_batch = make_micro_batch(1, self.max_tokens_list[strategy_id], train_task_num=train_task_num)
                cur_batch_size = warmup_micro_batch.batch_size
                cur_seq_len = warmup_micro_batch.seq_length
                cur_batch_offset_list = warmup_micro_batch.batch_offset_list
                cur_batch_size_list = warmup_micro_batch.batch_size_list
                task_batch_idxs_list = [[] for _ in range(train_task_num)]
                for task_id in range(train_task_num):
                    task_batch_idxs = np.zeros([cur_batch_offset_list[task_id], cur_batch_size_list[task_id], cur_batch_size], dtype=np.int64)
                    task_batch_idxs_list[task_id] = [task_batch_idxs for _ in range(num_micro_batches)]
                input_ids_list = [np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64) for _ in range(num_micro_batches)]
                masked_lm_labels_list = [np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64) for _ in range(num_micro_batches)]
                feed_dict = {
                    self.build_ops['input_ids']: input_ids_list,
                    self.build_ops['masked_lm_labels']: masked_lm_labels_list,
                }
                for i in range(train_task_num):
                    feed_dict[self.build_ops['task_batch_idxs'][i]] = task_batch_idxs_list[i]
                with ht.autocast(self.precision):
                    _results = self.build_ops['train_op'].graph.run(
                                self.build_ops['loss'],
                                [self.build_ops['loss'], self.build_ops['train_op']],
                                feed_dict=feed_dict,
                                num_micro_batches = num_micro_batches,
                                cur_strategy_id = strategy_id,
                                run_level = ht.run_level("compute_only"),
                                grad_scale = 1.0)
            print(f"{local_device}: warmup end...")
        
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
        print(f"strategy {strategy_id} start to train...")
        # dup_group_idx = 0
        for epoch in range(self.max_epochs):
            for step in range(self.max_steps):
                multi_task_global_batch_map = {}
                for task_id in range(train_task_num):
                    if self.dataset_ctxs[task_id].step >= self.dataset_ctxs[task_id].steps or \
                        self.dataset_ctxs[task_id].epoch >= self.dataset_ctxs[task_id].epochs:
                        continue
                    if train_iter_list[task_id] is None and os.environ.get('CUSTOM_DISTRIBUTION') != 'TRUE':
                        continue
                    else:
                        if os.environ.get('CUSTOM_DISTRIBUTION') == 'TRUE':
                            seq_len_distribution = task_seq_len_distribution[task_id]
                            global_batch = self.get_custom_global_batch(seq_len_distribution, args.max_seq_length, self.pad_id)
                        else:
                            try:
                                global_batch = next(train_iter_list[task_id])
                            except StopIteration:
                                train_iter_list[task_id] = self.train_data_iterator(self.dataset_ctxs[task_id].dataset, 0,
                                                                                    self.trainer_config.task_configs[task_id].global_batch_size,
                                                                                    args.min_seq_length, args.max_seq_length)
                                global_batch = next(train_iter_list[task_id])
                    multi_task_global_batch_map[task_id] = global_batch
                    self.dataset_ctxs[task_id].consumed_samples += len(global_batch)
                # with open("effectiveness.txt", 'a') as f:
                #     f.write(f"step: {step}\n")
                if not os.path.exists(f"case_study/{data_dispatch_pattern}"):
                    os.makedirs(f"case_study/{data_dispatch_pattern}")
                with open(f"case_study/{data_dispatch_pattern}/{local_host_name}-{local_device.index}.txt", 'a') as f:
                    f.write(f"step: {step}\n")
                run_batches_list, schedule_time = global_batch_scheduler(args, multi_task_global_batch_map, train_task_num, \
                                                                         self.pad_id, self.dps, self.tps, self.pps, dup_group_idx, self.max_tokens_list[strategy_id], \
                                                                         self.num_strategy, strategy_id, dynamic_planner, step=step)
                if not run_batches_list:
                    step -= 1
                    continue
                self.schedule_times.append(schedule_time)
                print(f"step = {step}")
                # if step == 1:
                #     with open(f"effective_logs/{local_host_name}-{local_device.index}.txt", 'a') as f:
                #         f.write(f"\n")
                # elif step > 20:
                #     break
                # continue
                if os.environ.get('GET_TOKENS') == 'TRUE':
                    self.total_run_times.append(0)
                    continue
                if os.environ.get('COST_MODEL_ESTIMATE') == 'TRUE':
                    continue
                # 统计tokens
                if local_dp_rank != -1 and (step > 0 or epoch > 0):
                    for micro_batches_list in run_batches_list:
                        for micro_batch in micro_batches_list:
                            batch_data = np.array(micro_batch.batch_data)
                            self.total_tokens += micro_batch.batch_size * micro_batch.seq_length
                            self.valid_tokens += np.sum(batch_data != self.pad_id)
                            if micro_batch.seq_length not in self.valid_bucket:
                                self.valid_bucket[micro_batch.seq_length] = 0
                                self.total_bucket[micro_batch.seq_length] = 0
                            self.valid_bucket[micro_batch.seq_length] += np.sum(batch_data != self.pad_id)
                            self.total_bucket[micro_batch.seq_length] += micro_batch.batch_size * micro_batch.seq_length
                if os.environ.get('GET_TOKENS') == 'TRUE':
                    self.total_run_times.append(0)
                    continue
                
                # prepare feed_dict
                feed_dict_list = []
                for run_id, micro_batches_list in enumerate(run_batches_list):
                    # for i, run_batch in enumerate(run_batches_list[0]):
                    #     print(f"run batch {i}: {run_batch}")
                    input_ids_list = []
                    # position_ids_list = []
                    # token_type_ids_list = []
                    masked_lm_labels_list = []
                    task_batch_idxs_list = [[] for _ in range(train_task_num)]
                    for micro_batch in micro_batches_list:
                        cur_batch_size = micro_batch.batch_size
                        cur_batch_data = np.array(micro_batch.batch_data).reshape(cur_batch_size, -1)
                        cur_seq_len = micro_batch.seq_length
                        cur_batch_offset_list = micro_batch.batch_offset_list
                        cur_batch_size_list = micro_batch.batch_size_list
                        for task_id in range(train_task_num):
                            task_batch_idxs = np.zeros([cur_batch_offset_list[task_id], cur_batch_size_list[task_id], cur_batch_size], dtype=np.int64)
                            task_batch_idxs_list[task_id].append(task_batch_idxs)
                        labels = cur_batch_data[:, 1:]
                        tokens = cur_batch_data[:, :-1]
                        # _, _position_ids = get_mask_and_position_ids(tokens, self.pad_id)
                        # _token_type_ids = np.zeros([cur_batch_size, cur_seq_len])
                        # print(f"cur_batch_size = {cur_batch_size}, cur_seq_len = {cur_seq_len}")
                        if dup_group_idx != -1:
                            input_ids_list.append(tokens.astype(np.int64))
                            # position_ids_list.append(_position_ids.astype(np.int64))
                            # token_type_ids_list.append(_token_type_ids.astype(np.int64))
                            masked_lm_labels_list.append(labels.astype(np.int64))
                        else:
                            input_ids_list.append(np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64))
                            # position_ids_list.append(get_position_ids(cur_batch_size, cur_seq_len).astype(np.int64))
                            # token_type_ids_list.append(np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64))
                            masked_lm_labels_list.append(np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64))
                    feed_dict = {
                        self.build_ops['input_ids']: input_ids_list,
                        # self.build_ops['position_ids']: position_ids_list,
                        # self.build_ops['token_type_ids']: token_type_ids_list,
                        self.build_ops['masked_lm_labels']: masked_lm_labels_list,
                    }
                    for i in range(train_task_num):
                        feed_dict[self.build_ops['task_batch_idxs'][i]] = task_batch_idxs_list[i]
                    feed_dict_list.append(feed_dict)

                iter_time = 0
                for run_id, feed_dict in enumerate(feed_dict_list):
                    if run_id == len(run_batches_list) - 1:
                        run_level = ht.run_level("compute_only")
                    else:
                        run_level = ht.run_level("local_grad")
                    ht.global_comm_barrier()
                    start_time = time.time()
                    try:
                        if os.environ.get('PROFILE_DYNAMIC_PLANNER') == 'TRUE':
                            with ht.autocast(self.precision):
                                with ht.profiler(enabled=True, record_shapes=False) as profiler:
                                    results = self.build_ops['train_op'].graph.run(
                                        self.build_ops['loss'],
                                        [self.build_ops['loss'], self.build_ops['train_op']],
                                        feed_dict=feed_dict,
                                        num_micro_batches = len(run_batches_list[run_id]),
                                        cur_strategy_id = strategy_id,
                                        run_level = run_level,
                                        grad_scale = 1.0)
                            e2e_time = float(profiler.summary()['graph_view'][0][1])
                            total_steam_time = float(profiler.summary()['graph_view'][11][1])
                            dp_grad_reduce_time = float(profiler.summary()['graph_view'][5][1])
                            self.dp_grad_reduce_times.append(dp_grad_reduce_time)
                            print(f"strategy {strategy_id} - e2e_time = {e2e_time / 1000:.4f} s")
                            print(f"strategy {strategy_id} - total_steam_time = {total_steam_time / 1000:.4f} s")
                            print(f"strategy {strategy_id} - dp_grad_reduce_time = {dp_grad_reduce_time / 1000:.4f} s")
                        else:
                            with ht.autocast(self.precision):
                                results = self.build_ops['train_op'].graph.run(
                                    self.build_ops['loss'],
                                    [self.build_ops['loss'], self.build_ops['train_op']],
                                    feed_dict=feed_dict,
                                    num_micro_batches = len(run_batches_list[run_id]),
                                    cur_strategy_id = strategy_id,
                                    run_level = run_level,
                                    grad_scale = 1.0)
                    except RuntimeError as e:
                        print(e)
                        os.killpg(0, signal.SIGTERM)
                    end_time = time.time()
                    iter_time += end_time - start_time
                    if (step > 0 or epoch > 0) and (run_level == ht.run_level("compute_only") or run_level == ht.run_level("update")):
                        print(f"iter {step}: {iter_time:.3f}s")
                        self.total_run_times.append(iter_time)
                        if run_level == ht.run_level("compute_only"):
                            if not os.path.exists("effective_logs"):
                                os.makedirs("effective_logs")
                            # with open(f"effective_logs/{local_host_name}-{local_device.index}.txt", 'a') as f:
                            #     f.write(f"{iter_time}\n")
                            with open(f"case_study/{data_dispatch_pattern}/{local_host_name}-{local_device.index}.txt", 'a') as f:
                                f.write(f"{iter_time}\n")
                                f.write("\n")
                            ht.global_comm_barrier()
                    # TODO: consumed samples of each task
                    if run_level == ht.run_level("update"):
                        if label_device_group.contains(local_device):
                            loss_out = results[0].numpy(force=True).mean()
                            consumed_samples = np.sum(self.dataset_ctxs[i].consumed_samples for i in range(train_task_num))
                            print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {iter_time:.4f}")
        print(f"token_num = {dynamic_planner.token_num}, valid_num = {dynamic_planner.valid_token_num}, padding_ratio = {1 - dynamic_planner.valid_token_num / dynamic_planner.token_num}, max schedule time = {max(self.schedule_times)}, min schedule time = {min(self.schedule_times)}, avg schedule time = {np.mean(self.schedule_times)}")
        # '''
        if local_device.index == 0 and local_host_name == 'worker-0':
            with open('tokens_statistics.txt', 'a') as f:
                f.write(f"{dynamic_planner.token_num}, {dynamic_planner.valid_token_num}, {1 - dynamic_planner.valid_token_num / dynamic_planner.token_num}\n")
                f.write(f"{max(self.schedule_times)}, {min(self.schedule_times)}, {np.mean(self.schedule_times)}\n")
                f.write(f"\n")
        # '''
        
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
        for seq_len in self.valid_bucket.keys():
            print(f"{seq_len}: padding ratio = {(self.total_bucket[seq_len] - self.valid_bucket[seq_len]) / self.total_bucket[seq_len]}")
        tokens_entry = {
            'dp': self.dps[strategy_id],
            'tp': self.tps[strategy_id],
            'pp': self.pps[strategy_id],
            'total_tokens': self.total_tokens,
            'valid_tokens': self.valid_tokens
        }
        write_to_csv(tokens_entry, f"temp/tokens_{self.num_strategy}_{strategy_id}_{local_host_name}_{local_device.index}")
        if label_device_group.contains(local_device):
            # total_run_time = np.mean(self.total_run_times)
            total_run_time = self.total_run_times
            schedule_time = np.mean(self.schedule_times)
            run_time_entry = {
                'dp': self.dps[global_dp_rank],
                'tp': self.tps[strategy_id],
                'pp': self.pps[strategy_id],
                'total_run_time': total_run_time,
                'schedule_time': schedule_time
            }
            print(f"total run time num = {len(total_run_time)}")
            write_to_csv(run_time_entry, f"temp/run_time_{self.num_strategy}_{local_host_name}_{local_device.index}")
        ht.global_comm_barrier()
        if local_device.index == 0 and local_host_name == 'worker-0':
            print(f"handler: {local_host_name}")
            time.sleep(30)
            total_cnt = dynamic_planner.token_num 
            valid_cnt = dynamic_planner.valid_token_num
            run_times = []
            schedule_times = []
            run_time_file_names = [f for f in os.listdir('temp') if f.startswith(f"run_time_{self.num_strategy}")]
            for run_time_file_name in run_time_file_names:
                rows = read_from_csv(f"temp/{run_time_file_name}")
                if len(rows) == 0:
                    continue
                run_times.append(rows[0]['total_run_time'])
                schedule_times.append(rows[0]['schedule_time'])
                os.remove(f"temp/{run_time_file_name}")
            log_entry = {
                'dp': self.dps,
                'tp': self.tps,
                'pp': self.pps,
                'max_tokens': self.max_tokens_list,
                'total_tokens': total_cnt,
                'effective_tokens': valid_cnt,
                'padding_ratio': (total_cnt - valid_cnt) / total_cnt,
                'run_time': np.min(run_times),
                'schedule_time': np.min(schedule_times)
            }
            '''
            with open('sensitivity_result.txt', 'a') as f:
                f.write(f"{(dynamic_planner.token_num - dynamic_planner.valid_token_num) / dynamic_planner.token_num}\n")
                f.write(f"{np.min(run_times, axis=0).tolist()}\n")
                f.write("\n")
            '''
            if os.environ['PROFILE_E2E_COST'] == 'TRUE':
                write_to_csv(log_entry, f"exp_result/e2e/run_statistics.csv")
            # os.system("rm -rf temp")
            print(f"total_cnt = {total_cnt}, valid_cnt = {valid_cnt}, mean_run_time = {np.min(run_times)}, mean_schedule_time = {np.min(schedule_times)}")
        # print(f"{local_host_name} - {local_device.index}: dp_grad_reduce_time = {np.mean(self.dp_grad_reduce_times)}")

