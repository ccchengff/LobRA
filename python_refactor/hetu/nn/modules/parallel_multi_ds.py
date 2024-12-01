import hetu
from .module import Module
import numbers
from .parallel_utils import get_device_index, config2ds

__all__ = [
    'HtMultiColumnParallelLinear', 
    'HtMultiRowParallelLinear', 
    'HtMultiParallelEmbedding',
    'HtMultiVocabParallelEmbedding',
    'HtMultiParallelLayerNorm',
    'HtMultiParallelRMSNorm',
]

class HtMultiParallelRMSNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, dtype=hetu.float32, recompute=False, name='rmsnorm'):
        super(HtMultiParallelRMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.name = name
        self.ds_map = {'dup': [], 'split0': [], 'split0_dup': [], 'split0_or_split0_dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup, device_group = config2ds(ds_parallel_config)
            dp, tp, sp, num_devices = ds_parallel_config['dup'] // ds_parallel_config['tp'], ds_parallel_config['tp'], ds_parallel_config['sp'], len(ds_parallel_config['device_group'])
            self.device_groups.append(device_group)
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            ds_split0 = hetu.DistributedStates(num_devices, {0: dp * tp}, [0], False)
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1])
            self.ds_map['dup'].append(ds_dup)
            self.ds_map['split0'].append(ds_split0)
            self.ds_map['split0_dup'].append(ds_split0_dup)
            if sp:
                self.ds_map['split0_or_split0_dup'].append(ds_split0)
            else:
                self.ds_map['split0_or_split0_dup'].append(ds_split0_dup)
        # workaround
        self.sp = sp
        self.recompute = recompute
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, self.ds_map['dup'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')

    def forward(self, input_p):
        # [bsz*seq_len, hidden_size]
        if input_p.check_multi_ds_equal(self.ds_map['split0_or_split0_dup']):
            input_p = input_p
        else:
            input_p = hetu.comm(input_p, self.ds_map['split0_or_split0_dup'])
        # output_rms_split0 = hetu.rms_norm(input_p, None, self.weight, None, is_rms_norm=True, \
        #                                   device_groups=self.device_groups, name=self.name+'_sp')[0]
        output_rms_split0 = hetu.fused_rmsnorm(input_p, self.weight, self.normalized_shape, \
                                               device_groups=self.device_groups, name=self.name)[0]
        if output_rms_split0.check_multi_ds_equal(self.ds_map['split0_dup']):
            output_rms = output_rms_split0
        else:
            if self.sp and self.recompute:
                with hetu.recompute():
                    output_rms = hetu.comm(output_rms_split0, self.ds_map['split0_dup'])
            else:
                output_rms = hetu.comm(output_rms_split0, self.ds_map['split0_dup'])
        return output_rms

class HtMultiParallelLayerNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, eps=1e-5, dtype=hetu.float32, recompute=False, name='ln'):
        super(HtMultiParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.name = name
        self.ds_map = {'dup': [], 'split0': [], 'split0_dup': [], 'split0_or_split0_dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup, device_group = config2ds(ds_parallel_config)
            dp, tp, sp, num_devices = ds_parallel_config['dup'] // ds_parallel_config['tp'], ds_parallel_config['tp'], ds_parallel_config['sp'], len(ds_parallel_config['device_group'])
            self.device_groups.append(device_group)
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            ds_split0 = hetu.DistributedStates(num_devices, {0: dp * tp}, [0], False) # for activation, no zero
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1])
            self.ds_map['dup'].append(ds_dup)
            self.ds_map['split0'].append(ds_split0)
            self.ds_map['split0_dup'].append(ds_split0_dup)
            if sp:
                self.ds_map['split0_or_split0_dup'].append(ds_split0)
            else:
                self.ds_map['split0_or_split0_dup'].append(ds_split0_dup)
        
        self.sp = sp
        self.recompute = recompute
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, self.ds_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')
        self.bias = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                              self.normalized_shape, self.ds_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_bias')

    def forward(self, input_p):
        if input_p.check_multi_ds_equal(self.ds_map['split0_or_split0_dup']):
            input_p = input_p
        else:
            input_p = hetu.comm(input_p, self.ds_map['split0_or_split0_dup'])
        out_ln_split0 = hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, self.eps, 
                                             device_groups=self.device_groups, name=self.name+'_sp')[0]      
        if out_ln_split0.check_multi_ds_equal(self.ds_map['split0_dup']):
            out_ln = out_ln_split0
        else:
            if self.sp and self.recompute:
                with hetu.recompute():
                    out_ln = hetu.comm(out_ln_split0, self.ds_map['split0_dup'])
            else:
                out_ln = hetu.comm(out_ln_split0, self.ds_map['split0_dup'])
        return out_ln

class HtMultiParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                 init_method='xavier_normal_', dtype=hetu.float32, name='embedding'):
        super(HtMultiParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ds_map = {'dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            self.ds_map['dup'].append(ds_dup)
        
        # embedding_table should not be splited in any dimension!
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_map['dup'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_groups=self.device_groups, name=f'{name}_table')
    
    def forward(self, input_p):
        return hetu.embedding_lookup(self.embedding_table, input_p, device_groups=self.device_groups, name=self.name)
    
class HtMultiVocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                init_method='xavier_normal_', dtype=hetu.float32, name='vocab_embedding'):
        super(HtMultiVocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

        self.ds_map = {'split0_dup': [], 'dup_split0': []}
        self.device_index = []
        self.device_groups = []
        self.vocab_start_index = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup_split0, device_group = config2ds(ds_parallel_config) # for embedding table
            self.device_groups.append(device_group)
            dp, tp, num_devices = ds_parallel_config['dup'], ds_parallel_config['split'].get('0', 1), len(ds_parallel_config['device_group'])
            assert dp * tp == num_devices, f'VocabParallelEmbedding get wrong ds_parallel_config: {ds_parallel_config}!'
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data
            self.ds_map['split0_dup'].append(ds_split0_dup)
            self.ds_map['dup_split0'].append(ds_dup_split0)
            
            dup_group_idx = ds_dup_split0.get_dup_group_index(device_index)
            vocab_start_index = num_embeddings // tp * dup_group_idx
            self.vocab_start_index.append(vocab_start_index)

        # embedding_table was splited in vocab dimension
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_map['dup_split0'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_groups=self.device_groups, name=f'{name}_table')
    
    def forward(self, input_p):
        if input_p.check_multi_ds_equal(self.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.ds_map['split0_dup'])
            print(f"warning: vocab parallel embedding need extra communication for \
                    adapt input tensor distributed_states into {self.ds_map['split0_dup']}!")

        # walkaround: do offset inside embedding lookup op 
        # input_offset = tensor_split0_dup - self.vocab_start_index[0] # should do in embedding_lookup op for multi ds?
        lookup_split0_partial = hetu.embedding_lookup(self.embedding_table, tensor_split0_dup, self.vocab_start_index, 
                                                      device_groups=self.device_groups, name=self.name+"_"+tensor_split0_dup.name)
        # if lookup_split0_partial.check_multi_ds_equal(self.ds_map['split0_dup']): # pure dp
        #     output = lookup_split0_partial
        # else:
        #     output = hetu.comm(lookup_split0_partial, self.ds_map['split0_dup'])
        # return output
        return lookup_split0_partial

class HtMultiColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, multi_ds_parallel_config,
                 bias=True, gather_output=True, dup=False, is_exp_A=False, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='colp'):
        super(HtMultiColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.multi_ds_parallel_config = multi_ds_parallel_config
        self.name = name
        self.dup = dup # whether weights are duplicated
        
        # experiment
        self.is_exp_A = is_exp_A

        self.ds_map = {'dup_split0': [], 'split0_dup': [], 'dup': [], 'split0': [], 'split0_or_split0_dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup_split1, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            dp, tp, sp, num_devices, zero = ds_parallel_config['dup'], \
                                            ds_parallel_config['split'].get('1', 1), \
                                            ds_parallel_config['sp'], \
                                            len(ds_parallel_config['device_group']), \
                                            ds_parallel_config['zero']
            # assert sp == self.exp_sp, f'we only support same sp for multi-strategy'
            assert dp * tp == num_devices, f'ColumnParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'        
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            # when dp=1 tp=8, weights: ds_dup_split0->ds_split0, data: ds_split0_dup->ds_dup
            # when dp=8 tp=1, weights: ds_dup_split0->ds_dup, data: ds_split0_dup->ds_split0
            ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: dp, 0: tp}, [-1, 0], zero) # for weights with trans_b
            ds_dup = hetu.DistributedStates(num_devices, {-1: dp * tp}, [-1], zero) # for weights with trans_b
            ds_split0 = hetu.DistributedStates(num_devices, {0: dp * tp}, [0], zero) # for weights with trans_b
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data
            self.ds_map['dup_split0'].append(ds_dup_split0)
            self.ds_map['split0_dup'].append(ds_split0_dup)
            self.ds_map['dup'].append(ds_dup)
            self.ds_map['split0'].append(ds_split0)
            if sp:
                self.ds_map['split0_or_split0_dup'].append(ds_split0)
            else:
                self.ds_map['split0_or_split0_dup'].append(ds_split0_dup)
        
        if dup:
            self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                [out_features, in_features], 
                                                self.ds_map['dup'], self.device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_weight')
        else:
            self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                [out_features, in_features], 
                                                self.ds_map['dup_split0'], self.device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_weight')
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_map['dup_split0'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_bias')
        else:
            self.bias = None
      
    def forward(self, input_p, **kwargs):
        if self.dup and not self.is_exp_A:
            input_ds = self.ds_map['split0_or_split0_dup']
        else:
            input_ds = self.ds_map['split0_dup']

        if input_p.check_multi_ds_equal(input_ds):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, input_ds)
            # print(f"warning: column parallel linear need extra communication for \
            #         adapt input tensor distributed_states into {input_ds}!")

        tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, self.bias, trans_b=True, device_groups=self.device_groups, name=f'linear_{self.name}')
        if not self.gather_output:
            output = tensor_split01
        else:
            if tensor_split01.check_multi_ds_equal(self.ds_map['split0_dup']): # pure dp
                output = tensor_split01
            else:
                output = hetu.comm(tensor_split01, self.ds_map['split0_dup'])

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtMultiRowParallelLinear(Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(self, in_features, out_features, 
                 multi_ds_parallel_config, bias=True, exp_is_lora=False,
                 init_method='xavier_normal_', 
                 dtype=hetu.float32, name='rowp'):
        super(HtMultiRowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multi_ds_parallel_config = multi_ds_parallel_config
        self.name = name
        
        self.ds_map = {'dup_split0': [], 'dup_split1': [], 'dup': [], 'split01': [], \
                       'split0_dup': [], 'split0': [], 'split0_or_split0_dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup_split0, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            dp, tp, sp, num_devices, zero = ds_parallel_config['dup'], \
                                            ds_parallel_config['split'].get('0', 1), \
                                            ds_parallel_config['sp'], \
                                            len(ds_parallel_config['device_group']), \
                                            ds_parallel_config['zero']
            assert dp * tp == num_devices, f'RowParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            ds_dup_split1 = hetu.DistributedStates(num_devices, {-1: dp, 1: tp}, [-1, 1], zero) # for weight with trans_b
            ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1], zero) # for bias
            ds_split01 = hetu.DistributedStates(num_devices, {0: dp, 1: tp}, [0, 1]) # for data split in dimension 1
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data reduce partial to dup
            ds_split0 = hetu.DistributedStates(num_devices, {0: tp * dp}, [0]) # for sequence parallel
            self.ds_map['dup_split0'].append(ds_dup_split0)
            self.ds_map['dup_split1'].append(ds_dup_split1)
            self.ds_map['dup'].append(ds_dup)
            self.ds_map['split01'].append(ds_split01)
            self.ds_map['split0_dup'].append(ds_split0_dup)
            self.ds_map['split0'].append(ds_split0)
            if sp and not exp_is_lora:
                self.ds_map['split0_or_split0_dup'].append(ds_split0)
            else:
                self.ds_map['split0_or_split0_dup'].append(ds_split0_dup)
            
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [in_features, out_features], 
                                              self.ds_map['dup_split0'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')        
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_map['dup'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_bias')
        else:
            self.bias = None

    def forward(self, input_p, **kwargs):
        if input_p.check_multi_ds_equal(self.ds_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hetu.comm(input_p, self.ds_map['split01']) # exists src_ds == dst_ds case, just ignore it in comm_op

        tensor_split0_partial = hetu.linear(tensor_split01, self.weight, trans_b=False, device_groups=self.device_groups, name=f'linear_{self.name}')
        if tensor_split0_partial.check_multi_ds_equal(self.ds_map['split0_or_split0_dup']): # pure dp
            tensor_split0_dup = tensor_split0_partial
        else:
            tensor_split0_dup = hetu.comm(tensor_split0_partial, self.ds_map['split0_or_split0_dup'])
        # output = tensor_split0_dup + self.bias if self.bias is not None else tensor_split0_dup
        output = hetu.add(tensor_split0_dup, self.bias, name=f'add_{self.name}') if self.bias is not None else tensor_split0_dup

        return output