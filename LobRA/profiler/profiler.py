import os
import signal
import hetu as ht
import numpy as np
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from utils import parse_multi_ds_parallel_config
from data_utils import get_position_ids
from trainer.batch_scheduler import make_micro_batch

class Profiler:
    def __init__(
        self,
        profile_args,
        pretrained_model_wrapper: ModelWrapper,
        finetune_model_wrapper: ModelWrapper,
        optimizer_wrapper: OptimizerWrapper,
        trainer_config: TrainerConfig,
        ds_parallel_configs,
    ):
        self.pretrained_model_wrapper = pretrained_model_wrapper
        self.finetune_model_wrapper = finetune_model_wrapper
        self.optimizer_wrapper = optimizer_wrapper
        self.trainer_config = trainer_config
        self.ds_parallel_configs = ds_parallel_configs
        
        self.build_ops = None
        self.profile_steps = profile_args.profile_steps
        self.warmup_steps = profile_args.warmup_steps
        self.train_task_num = profile_args.train_task_num
        self.precision = None
        
        # ds parallel config
        self.dp, self.tp, self.pp, self.sp, self.num_gpus = self.get_ds_parallel_configs(profile_args)
        
        # logging
        # self.fw_run_time = []
        # self.bw_run_time = []
        self.block_stream_time = []
        self.total_stream_time = []
        
    def get_ds_parallel_configs(self, profile_args):
        dp = profile_args.dp
        tp = profile_args.tp
        pp = profile_args.pp
        sp = profile_args.sp
        num_gpus = tp
        return dp, tp, pp, sp, num_gpus

    def build_model(self, profile_args, ds_parallel_configs):
        # Build model
        with ht.graph("define_and_run", num_strategy=1):
            if profile_args.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            self.precision = eval(precision)
            with ht.autocast(eval(precision)):
                self.create_define_and_run_graph(profile_args, ds_parallel_configs)
    
    def create_define_and_run_graph(self, profile_args, ds_parallel_configs):
        # 获取ds
        input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        task_multi_ds, task_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
        
        # 获取默认的seq_len, dp, mbs_times_dp, batch_offset, batch_size
        default_seq_len = profile_args.default_seq_len
        default_dp = self.dp
        default_mbs_times_dp = profile_args.default_mbs * default_dp
        default_batch_offset = 0
        default_batch_size = profile_args.default_mbs
        
        # 构建placeholder
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
        position_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
        # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')
        task_batch_idxs = []
        for i in range(profile_args.train_task_num):
            task_batch_idxs.append(ht.parallel_placeholder(ht.int64, global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                                                           multi_ds=task_multi_ds, device_groups=task_device_groups, name='task_batch_idxs_task{}'.format(i), is_cpu=True))

        # 设置symbolic shape
        self.pretrained_model_wrapper.model_config.dp_symbol.set_data(default_dp)
        self.pretrained_model_wrapper.model_config.train_task_num = self.trainer_config.train_task_num
        # 创建预训练模型
        pretrained_model = self.pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
        # 创建微调模型
        peft_configs = [task_config.lora_config for task_config in self.trainer_config.task_configs]
        model = self.finetune_model_wrapper.create_model(model=pretrained_model, peft_configs=peft_configs)
        # 构建静态图
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=None,
            labels=masked_lm_labels,
            task_batch_idxs=task_batch_idxs
        )
        # build optimizer
        opt = self.optimizer_wrapper.create_optimizer(lr=profile_args.lr)
        train_op = opt.minimize(loss)
        # build ops
        self.build_ops = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            # 'token_type_ids': token_type_ids,
            'masked_lm_labels': masked_lm_labels,
            'task_batch_idxs': task_batch_idxs,
            'loss': loss,
            'train_op': train_op
        }
    
    def profile(self, mbs, seq_len):
        # self.fw_run_time = []
        # self.bw_run_time = []
        self.block_time = []
        self.total_stream_time = []
        for step in range(self.profile_steps + self.warmup_steps):
            print(f"profiler: step = {step}")
            profile_micro_batch = make_micro_batch(mbs, seq_len, train_task_num=self.train_task_num)
            batch_data = np.array(profile_micro_batch.batch_data).reshape(mbs, -1)
            labels = batch_data[:, 1:]
            tokens = batch_data[:, :-1]
            _position_ids = get_position_ids(mbs, seq_len)
            # _token_type_ids = np.zeros([mbs, seq_len])
            task_batch_idxs_list = []
            for train_task_idx in range(self.train_task_num):
                task_batch_idxs_i = np.zeros([profile_micro_batch.batch_offset_list[train_task_idx], profile_micro_batch.batch_size_list[train_task_idx], profile_micro_batch.batch_size], dtype=np.int64)
                task_batch_idxs_list.append(task_batch_idxs_i)
            feed_dict = {
                self.build_ops['input_ids']: tokens.astype(np.int64),
                self.build_ops['position_ids']: _position_ids.astype(np.int64),
                # self.build_ops['token_type_ids']: _token_type_ids.astype(np.int64),
                self.build_ops['masked_lm_labels']: labels.astype(np.int64),
            }
            for i in range(self.train_task_num):
                feed_dict[self.build_ops['task_batch_idxs'][i]] = task_batch_idxs_list[i]
            run_level = ht.run_level("update")
            try:
                with ht.autocast(self.precision):
                    with ht.profiler(enabled=True, record_shapes=False) as profiler:
                        results = self.build_ops['train_op'].graph.run(
                            self.build_ops['loss'],
                            [self.build_ops['loss'], self.build_ops['train_op']],
                            feed_dict=feed_dict,
                            num_micro_batches=1,
                            cur_strategy_id=0,
                            run_level = run_level,
                            grad_scale=1.0)
                if step >= self.warmup_steps:
                    self.total_stream_time.append(float(profiler.summary()['graph_view'][11][1])) # total-time-stream
                    self.block_time.append(float(profiler.summary()['graph_view'][12][1])) # block time
                    with open("block_time.txt", "a") as f:
                        f.write(f"{float(profiler.summary()['graph_view'][12][1])}, {float(profiler.summary()['graph_view'][13][1])}, {float(profiler.summary()['graph_view'][14][1])}, {float(profiler.summary()['graph_view'][15][1])}, {float(profiler.summary()['graph_view'][16][1])}, {float(profiler.summary()['graph_view'][17][1])}\n")
                    # self.fw_run_time.append(float(profiler.summary()['graph_view'][12][1])) # block-forward
                    # self.bw_run_time.append(float(profiler.summary()['graph_view'][13][1])) # block-backward
            except RuntimeError as e:
                print(e)
                # break
                os.killpg(0, signal.SIGTERM)