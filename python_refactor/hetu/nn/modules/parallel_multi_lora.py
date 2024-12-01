import hetu
from .module import Module
import numbers
from queue import Queue
from .parallel_multi_ds import HtMultiColumnParallelLinear, HtMultiRowParallelLinear
from .parallel_utils import get_multi_ds_parallel_config

__all__ = [
    'HtLoRAMultiColumnParallelLinear', 
    'HtLoRAMultiRowParallelLinear',
]

class HtLoRAMultiColumnParallelLinear(Module):
    """Linear layer with column parallelism and parallelized LoRA adapter.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    
    For LoRA adapters, we adapt the TP strategy from S-LoRA (first part):
    https://arxiv.org/abs/2311.03285
    LoRA_A is parallelized along its second dimension, then do all-gather communication.
    LoRA_B is parallelized along its second dimension, then add to base model's partial result.
    
    Note: ColumnParallel here refers to base model's parallelism.
    """
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='colp_lora'
    ):
        super(HtLoRAMultiColumnParallelLinear, self).__init__()
        self.name = name
        self.base_layer = base_layer
        # Rank-stabilized LoRA scaling option from
        # https://doi.org/10.48550/arXiv.2312.03732
        if use_rslora:
            self.scaling = lora_alpha / (rank ** 0.5)
        else:
            self.scaling = lora_alpha / rank
        
        lora_a_multi_ds_parallel_configs = get_multi_ds_parallel_config(multi_ds_parallel_config, 'lora_A')
        lora_b_multi_ds_parallel_configs = get_multi_ds_parallel_config(multi_ds_parallel_config, 'lora_B')

        self.lora_A = HtMultiColumnParallelLinear(base_layer.in_features, rank, lora_a_multi_ds_parallel_configs,
                                                  bias=False, gather_output=True, init_method='he_uniform_',
                                                  dtype=base_layer.dtype, name=f'lora_A_{self.name}')
        self.lora_B = HtMultiColumnParallelLinear(rank, base_layer.out_features, lora_b_multi_ds_parallel_configs,
                                                  bias=False, gather_output=base_layer.gather_output, init_method='zeros_',
                                                  dtype=base_layer.dtype, name=f'lora_B_{self.name}')
        
        if lora_dropout > 0.0:
            self.lora_dropout = hetu.nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = hetu.nn.Identity()
      
    def forward(self, input_p):
        base_split01 = self.base_layer(input_p)
        lora_split01 = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_split01.check_multi_ds_equal(base_split01.multi_distributed_states):
            lora_split0_dup = lora_split01
        else:
            lora_split0_dup = hetu.comm(lora_split01, base_split01.multi_distributed_states, name=f'comm_{self.name}')
        output = hetu.add(base_split01, lora_split0_dup, name=f'sync_add_{self.name}')
        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtLoRAMultiRowParallelLinear(Module):
    """Linear layer with row parallelism and parallelized LoRA adapter.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    For LoRA adapters, we adapt the TP strategy from S-LoRA (second part):
    https://arxiv.org/abs/2311.03285
    LoRA_A is parallelized along its first dimension, then do all-reduce communication.
    LoRA_B is parallelized along its second dimension, then add to base model's partial result.
    
    Note: S-LoRA fuses all-gather and all-reduce finally, here we do all-gather and all-reduce separately.
          Specifically, for LoRA_B, do all-gather and add to base model's all-reduce result.
          ColumnParallel here refers to base model's parallelism.
    """
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='rowp_lora'
    ):
        super(HtLoRAMultiRowParallelLinear, self).__init__()
        self.name = name
        self.base_layer = base_layer
        # Rank-stabilized LoRA scaling option from
        # https://doi.org/10.48550/arXiv.2312.03732
        if use_rslora:
            self.scaling = lora_alpha / (rank ** 0.5)
        else:
            self.scaling = lora_alpha / rank
        
        lora_a_multi_ds_parallel_configs = get_multi_ds_parallel_config(multi_ds_parallel_config, 'lora_A')
        lora_b_multi_ds_parallel_configs = get_multi_ds_parallel_config(multi_ds_parallel_config, 'lora_B')
        
        self.lora_A = HtMultiRowParallelLinear(base_layer.in_features, rank, lora_a_multi_ds_parallel_configs,
                                               bias=False, init_method='he_uniform_',
                                               dtype=base_layer.dtype, name=f'lora_A_{self.name}')
        self.lora_B = HtMultiColumnParallelLinear(rank, base_layer.out_features, lora_b_multi_ds_parallel_configs,
                                                  bias=False, gather_output=True, init_method='zeros_',
                                                  dtype=base_layer.dtype, name=f'lora_B_{self.name}')
        
        if lora_dropout > 0.0:
            self.lora_dropout = hetu.nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = hetu.nn.Identity()

    def forward(self, input_p):
        base_result = self.base_layer(input_p)
        lora_split0 = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_split0.check_multi_ds_equal(base_result.multi_distributed_states):
            lora_result = lora_split0
        else:
            lora_result = hetu.comm(lora_split0, base_result.multi_distributed_states, name=f'comm_{self.name}')
        result = hetu.add(base_result, lora_result, name=f'sync_add_{self.name}')
        return result