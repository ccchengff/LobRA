import os
import hetu
from queue import Queue
from typing import Dict, Optional, Tuple, List
from hetu.nn.modules.module import Module
import hetu.nn.modules.parallel_multi_ds as parallel_multi_ds
from hetu.nn.modules.parallel_utils import get_multi_ds_parallel_config

class LoraLayer():
    def __init__(self, base_layer: Module, multi_ds_parallel_config, name) -> None:
        self.base_layer = base_layer
        self.multi_ds_parallel_config = multi_ds_parallel_config
        self.rank = 0
        self.lora_alpha = 1
        self.lora_dropout: Module = Module()
        self.lora_A: Module = Module()
        self.lora_B: Module = Module()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.name = name
    
    def update_layer(self, rank, lora_alpha, lora_dropout, use_rslora):
        if rank <= 0:
            raise ValueError(f"`rank` should be a positive integer value but the value passed is {rank}")
        self.rank = rank
        self.lora_alpha = lora_alpha
        if use_rslora:
            self.scaling = lora_alpha / (rank ** 0.5)
        else:
            self.scaling = lora_alpha / rank

        if lora_dropout > 0.0:
            self.lora_dropout = hetu.nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = hetu.nn.Identity()
        
        lora_a_multi_ds_parallel_configs = get_multi_ds_parallel_config(self.multi_ds_parallel_config, 'lora_A')
        lora_b_multi_ds_parallel_configs = get_multi_ds_parallel_config(self.multi_ds_parallel_config, 'lora_B')
        
        # ablation experiment of LoRA split method
        dup_a1 = False
        split_b2 = False
        if os.environ.get('LORA_SPLIT_METHOD') == 'DUP_A1':
            dup_a1 = True            
        elif os.environ.get('LORA_SPLIT_METHOD') == 'SPLIT_B2':
            split_b2 = True

        if isinstance(self.base_layer, parallel_multi_ds.HtMultiRowParallelLinear):
            self.lora_A = parallel_multi_ds.HtMultiRowParallelLinear(self.in_features, self.rank, lora_a_multi_ds_parallel_configs,
                                                                     bias=False, init_method='he_uniform_', exp_is_lora=split_b2,
                                                                     dtype=self.base_layer.dtype, name=f'lora_A_{self.name}')
            self.lora_B = parallel_multi_ds.HtMultiColumnParallelLinear(self.rank, self.out_features, lora_b_multi_ds_parallel_configs,
                                                                        bias=False, gather_output=split_b2, dup=(not split_b2), init_method='zeros_',
                                                                        dtype=self.base_layer.dtype, name=f'lora_B_{self.name}')
        elif isinstance(self.base_layer, parallel_multi_ds.HtMultiColumnParallelLinear):
            self.lora_A = parallel_multi_ds.HtMultiColumnParallelLinear(self.in_features, self.rank, lora_a_multi_ds_parallel_configs,
                                                                        bias=False, gather_output=True, dup=dup_a1, is_exp_A=dup_a1, init_method='he_uniform_',
                                                                        dtype=self.base_layer.dtype, name=f'lora_A_{self.name}')
            self.lora_B = parallel_multi_ds.HtMultiColumnParallelLinear(self.rank, self.out_features, lora_b_multi_ds_parallel_configs,
                                                                        bias=False, gather_output=False, init_method='zeros_',
                                                                        dtype=self.base_layer.dtype, name=f'lora_B_{self.name}')

class MultiLoraLayers():
    def __init__(self, base_layer: Module, multi_ds_parallel_config, name) -> None:
        self.base_layer = base_layer
        self.multi_ds_parallel_config = multi_ds_parallel_config
        self.lora_layers: Dict[int, LoraLayer] = {}
        self.name = name
    
    def update_layers(self, ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices):
        for i, task_indice in enumerate(task_indices):
            self.lora_layers[task_indice] = LoraLayer(self.base_layer, self.multi_ds_parallel_config, f'{self.name}_task{task_indice}')
            self.lora_layers[task_indice].update_layer(ranks[i], lora_alphas[i], lora_dropouts[i], use_rsloras[i])

class HtMultiColumnParallelLinear(Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='colp_lora'
    ):
        super(HtMultiColumnParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        LoraLayer.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layer(rank, lora_alpha, lora_dropout, use_rslora)
    
    def forward(self, input_p):
        base_result = self.base_layer(input_p)
        lora_result = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
            lora_comm_result = lora_result
        else:
            lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}')
        output = hetu.add(base_result, lora_comm_result, name=f'sync_add_{self.name}')
        return output

class HtMultiRowParallelLinear(Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='rowp_lora'
    ):
        super(HtMultiRowParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        LoraLayer.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layer(rank, lora_alpha, lora_dropout, use_rslora)
    
    def forward(self, input_p):
        base_result = self.base_layer(input_p)
        lora_result = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
            lora_comm_result = lora_result
        else:
            lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}')
        output = hetu.add(base_result, lora_comm_result, name=f'sync_add_{self.name}')
        return output

class HtMultiLoRAColumnParallelLinear(Module, MultiLoraLayers):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        config,
        ranks: List[int] = [0],
        lora_alphas: List[int] = [1],
        lora_dropouts: List[float] = [0.0],
        use_rsloras: List[bool] = [False],
        task_indices: List[int] = [0],
        name='colp_lora'
    ):
        super(HtMultiLoRAColumnParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        self.config = config
        MultiLoraLayers.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layers(ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices)
    
    def forward(self, input_p, **kwargs):
        assert 'task_batch_idxs' in kwargs, 'task_batch_idxs should be passed in kwargs!'
        task_batch_idxs = kwargs['task_batch_idxs']
        if input_p.check_multi_ds_equal(self.base_layer.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.base_layer.ds_map['split0_dup'])
            print(f"warning: column parallel linear need extra communication for \
                    adapt input tensor distributed_states into {self.base_layer.ds_map['split0_dup']}!")
        base_result = self.base_layer(tensor_split0_dup)
        if self.config.train_task_num == 1:
            lora_result = self.lora_layers[0].lora_B(self.lora_layers[0].lora_A(tensor_split0_dup))
            if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
                lora_comm_result = lora_result
            else:
                lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}_task0')
            lora_comm_result = hetu.mul(lora_comm_result, self.lora_layers[0].scaling, name=f'mul_{self.name}_task0')
            base_result = hetu.index_add_(base_result, lora_comm_result, task_batch_idxs[0], 0, name=f'index_add_{self.name}_task0')
        else: 
            # TODO: 改成支持packing
            task_tensor_split0_dup_list = hetu.split(tensor_split0_dup, task_batch_idxs, dim=0, name=f'split_task_{self.name}')
            for i in range(self.config.train_task_num):
                # task_tensor_split0_dup = hetu.slice(tensor_split0_dup, task_batch_idxs[i], [hetu.IntSymbol(-1), hetu.IntSymbol(0)], \
                #                                     [hetu.IntSymbol(-1), hetu.IntSymbol(self.base_layer.in_features)], name=f'slice_{self.name}_{i}')
                # lora_result = hetu.mul(self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup)), self.lora_layers[i].scaling, name=f'mul_{self.name}_{i}')
                lora_result = self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup_list[i]))
                if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
                    lora_comm_result = lora_result
                else:
                    lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}_task{i}')
                lora_comm_result = hetu.mul(lora_comm_result, self.lora_layers[i].scaling, name=f'mul_{self.name}_task{i}')
                base_result = hetu.index_add_(base_result, lora_comm_result, task_batch_idxs[i], 0, name=f'index_add_{self.name}_task{i}')
        return base_result

class HtMultiLoRARowParallelLinear(Module, MultiLoraLayers):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        config,
        ranks: List[int] = [0],
        lora_alphas: List[int] = [1],
        lora_dropouts: List[float] = [0.0],
        use_rsloras: List[bool] = [False],
        task_indices: List[int] = [0],
        name='rowp_lora'
    ):
        super(HtMultiLoRARowParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        self.config = config
        MultiLoraLayers.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layers(ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices)
    
    def forward(self, input_p, **kwargs):
        assert 'task_batch_idxs' in kwargs, 'task_batch_idxs should be passed in kwargs!'
        task_batch_idxs = kwargs['task_batch_idxs']
        if input_p.check_multi_ds_equal(self.base_layer.ds_map['split01']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.base_layer.ds_map['split01'])
            print(f"warning: column parallel linear need extra communication for \
                    adapt input tensor distributed_states into {self.base_layer.ds_map['split01']}!")
        base_result = self.base_layer(tensor_split0_dup)
        if self.config.train_task_num == 1:
            lora_result = self.lora_layers[0].lora_B(self.lora_layers[0].lora_A(tensor_split0_dup))
            if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
                lora_comm_result = lora_result
            else:
                lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}_task0')
            lora_comm_result = hetu.mul(lora_comm_result, self.lora_layers[0].scaling, name=f'mul_{self.name}_task0')
            base_result = hetu.index_add_(base_result, lora_comm_result, task_batch_idxs[0], 0, name=f'index_add_{self.name}_task0')
        else:
            task_tensor_split0_dup_list = hetu.split(tensor_split0_dup, task_batch_idxs, dim=0, name=f'split_task_{self.name}')
            for i in range(self.config.train_task_num):
                # task_tensor_split0_dup = hetu.slice(tensor_split0_dup, task_batch_idxs[i], [hetu.IntSymbol(-1), hetu.IntSymbol(0)], \
                #                                     [hetu.IntSymbol(-1), tensor_split0_dup.symbolic_shape[1]], name=f'slice_{self.name}_task{i}')
                # lora_result = hetu.mul(self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup)), self.lora_layers[i].scaling, name=f'mul_{self.name}_{i}')
                lora_result = self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup_list[i]))
                if lora_result.check_multi_ds_equal(base_result.multi_distributed_states):
                    lora_comm_result = lora_result
                else:
                    lora_comm_result = hetu.comm(lora_result, base_result.multi_distributed_states, name=f'comm_{self.name}_task{i}')
                lora_comm_result = hetu.mul(lora_comm_result, self.lora_layers[i].scaling, name=f'mul_{self.name}_task{i}')
                base_result = hetu.index_add_(base_result, lora_comm_result, task_batch_idxs[i], 0, name=f'index_add_{self.name}_task{i}')
        return base_result

def dispatch_lora_layer(target, **kwargs) -> Optional[Module]:
    new_module = None
    ds_parallel_configs = get_multi_ds_parallel_config(target.multi_ds_parallel_config, 'lora')
    
    if isinstance(target, parallel_multi_ds.HtMultiRowParallelLinear):
        new_module = HtMultiRowParallelLinear(target, ds_parallel_configs, name=target.name, **kwargs)
    elif isinstance(target, parallel_multi_ds.HtMultiColumnParallelLinear):
        new_module = HtMultiColumnParallelLinear(target, ds_parallel_configs, name=target.name, **kwargs)
    else:
        print(f"Not Supported for module {target}")
    
    return new_module

def dispatch_multi_lora_layers(target, config, **kwargs) -> Optional[Module]:
    new_module = None
    ds_parallel_configs = get_multi_ds_parallel_config(target.multi_ds_parallel_config, 'lora')
    
    if isinstance(target, parallel_multi_ds.HtMultiRowParallelLinear):
        new_module = HtMultiLoRARowParallelLinear(target, ds_parallel_configs, config, name=target.name, **kwargs)
    elif isinstance(target, parallel_multi_ds.HtMultiColumnParallelLinear):
        new_module = HtMultiLoRAColumnParallelLinear(target, ds_parallel_configs, config, name=target.name, **kwargs)
    else:
        print(f"Not Supported for module {target}")
    
    return new_module