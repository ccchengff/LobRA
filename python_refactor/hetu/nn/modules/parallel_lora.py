import hetu
from .module import Module
import numbers
from .parallel_ds import HtColumnParallelLinear, HtParallelEmbedding, HtParallelLayerNorm,\
                        HtRowParallelLinear, HtVocabParallelEmbedding
from .parallel_utils import get_device_index, config2ds

__all__ = [
    'LoRAColumnParallelLinear', 
    'LoRARowParallelLinear', 
    'LoRAParallelEmbedding',
    'LoRAVocabParallelEmbedding',
    'LoRAParallelLayerNorm',
    'LoRAModel',
]

class LoRAParallelLayerNorm(HtParallelLayerNorm):
    def __init__(self, ori_model, blocksize = 64, qtype = hetu.bfloat16,
                 map_absmax = {}, name='ln'):
        HtParallelLayerNorm.__init__(self, ori_model.normalized_shape, ori_model.ds_parallel_config,
                                     ori_model.eps, dtype=ori_model.dtype, name='ln', ori_model = ori_model)
        ds, self.device_group = config2ds(self.ds_parallel_config)
        device_index = get_device_index(self.device_group)
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hetu.float4) or (lora_dtype == hetu.nfloat4):
            lora_dtype = qtype
        # self.weight = ori_model.weight
        self.weight.set_requires_grad(False)
        # self.bias = ori_model.bias
        self.bias.set_requires_grad(False)
        if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
            absmax_size = self.weight.global_numel() // blocksize
            if (self.weight.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weight.id, "blocksize": blocksize}
                self.weight_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                             [absmax_size,], ds, device_index, 
                                                             dtype=hetu.float32, requires_grad=False,
                                                             parameter_dict = parameter_dict, 
                                                             device_group=self.device_group, name=f'{name}_weight_absmax')
                map_absmax[self.weight.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weight.id]
            # self.weight.add_in_dep_linker(self.weight_absmax)
            if (self.bias.id not in map_absmax):
                parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                self.bias_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                            [absmax_size,], ds, device_index, 
                                                            dtype=hetu.float32, requires_grad=False, 
                                                            parameter_dict = parameter_dict,
                                                            device_group=self.device_group, name=f'{name}_bias_absmax')
                map_absmax[self.bias.id] = self.bias_absmax
            else:
                self.bias_absmax = map_absmax[self.bias.id]
            # self.bias.add_in_dep_linker(self.bias_absmax)

    def forward(self, input_p):
        return hetu.layer_norm(input_p, self.weight, self.bias, self.normalized_shape, 
                               self.eps, device_group=self.device_group, name=self.name)[0]

class LoRAParallelEmbedding(HtParallelEmbedding):
    def __init__(self, ori_model, rank, blocksize = 64, qtype = hetu.bfloat16,
                 map_absmax = {}, name='embedding_lora'):
        HtParallelEmbedding.__init__(self, ori_model.num_embeddings, ori_model.embedding_dim, 
                                     ori_model.ds_parallel_config,
                                     init_method='xavier_normal_', dtype=ori_model.dtype, name='embedding',
                                     ori_model = ori_model)
        self.rank = rank
        # self.embedding_table = ori_model.embedding_table
        self.embedding_table.set_requires_grad(False)
        ds_dup = hetu.DistributedStates(self.ds_parallel_config['dup'], {-1: self.ds_parallel_config['dup']}, [-1])
        if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
            absmax_size = self.embedding_table.global_numel() // blocksize
            if (self.embedding_table.id not in map_absmax):
                parameter_dict = {"tensor_id": self.embedding_table.id, "blocksize": blocksize}
                self.embedding_table_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                                      [absmax_size,], ds_dup, 
                                                                      get_device_index(self.device_group),
                                                                      dtype=hetu.float32, requires_grad=False, 
                                                                      parameter_dict = parameter_dict,
                                                                      device_group=self.device_group, name=f'{name}_table_absmax')
                map_absmax[self.embedding_table.id] = self.embedding_table_absmax
            else:
                self.embedding_table_absmax = map_absmax[self.embedding_table.id]
    
    def forward(self, input_p):
        return HtParallelEmbedding.forward(self, input_p)
    
class LoRAVocabParallelEmbedding(HtVocabParallelEmbedding):
    def __init__(self, ori_model, rank, blocksize = 64,
                 lora_alpha = 1, qtype = hetu.bfloat16,
                 map_absmax = {}, name='vocab_embedding_lora'):
        HtVocabParallelEmbedding.__init__(self, ori_model.num_embeddings, ori_model.embedding_dim, 
                                          ori_model.ds_parallel_config,
                                          init_method='zeros_', dtype=ori_model.dtype, 
                                          name='vocab_embedding', ori_model = ori_model)
        self.rank = rank
        # self.embedding_table = ori_model.embedding_table
        self.embedding_table.set_requires_grad(False)
        if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
            absmax_size = self.embedding_table.global_numel() // blocksize
            if (self.embedding_table.id not in map_absmax):
                parameter_dict = {"tensor_id": self.embedding_table.id, "blocksize": blocksize}
                self.embedding_table_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                                      [absmax_size,], self.ds_map['dup'], 
                                                                      get_device_index(self.device_group),
                                                                      dtype=hetu.float32, requires_grad=False, 
                                                                      parameter_dict = parameter_dict,
                                                                      device_group=self.device_group, name=f'{name}_table_absmax')
                map_absmax[self.embedding_table.id] = self.embedding_table_absmax
            else:
                self.embedding_table_absmax = map_absmax[self.embedding_table.id]
            # self.embedding_table.add_in_dep_linker(self.embedding_table)
                
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hetu.float4) or (lora_dtype == hetu.nfloat4):
            lora_dtype = qtype
        self.lora_dtype = lora_dtype
        if self.lora_alpha != 0:
            self.embedding_tableA = HtVocabParallelEmbedding(self.num_embeddings, rank, self.ds_parallel_config,
                                                            init_method='zeros_', 
                                                            dtype=lora_dtype, name='vocab_embedding_loraA')
            self.embedding_tableB = HtRowParallelLinear(rank, self.embedding_dim, self.ds_parallel_config,
                                                        bias=False, init_method='normal_', 
                                                        dtype=lora_dtype, name='vocab_embedding_loraB')
        
    
    def forward(self, input_p):
        output = HtVocabParallelEmbedding.forward(self, input_p)
        # if self.lora_alpha != 0:
        #     output = output + self.embedding_tableB(self.embedding_tableA(input_p)) * self.lora_alpha
        return output
    
class LoRAColumnParallelLinear(HtColumnParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, ori_model, rank, blocksize = 64,
                 lora_alpha = 1, qtype = hetu.bfloat16, 
                 map_absmax = {}, name='colp_lora'):
        HtColumnParallelLinear.__init__(self, ori_model.in_features, ori_model.out_features, 
                                        ori_model.ds_parallel_config,
                                        bias=False if (ori_model.bias is None) else True, 
                                        gather_output = ori_model.gather_output, 
                                        init_method='zeros_', dtype=ori_model.dtype, 
                                        name='colp', ori_model = ori_model)
        self.rank = rank     
        # self.weight = ori_model.weight
        self.weight.set_requires_grad(False)
        if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
            absmax_size = self.weight.global_numel() // blocksize
            if (self.weight.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weight.id, "blocksize": blocksize}
                self.weight_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                             [absmax_size], self.ds_map['dup'], 
                                                             get_device_index(self.device_group), 
                                                             dtype=hetu.float32, requires_grad=False, 
                                                             parameter_dict = parameter_dict,
                                                             device_group=self.device_group, 
                                                             name=f'{name}_weight_absmax')
                map_absmax[self.weight.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weight.id]
            # self.weight.add_in_dep_linker(self.weight_absmax)

        lora_dtype = self.dtype
        if (lora_dtype == hetu.float4) or (lora_dtype == hetu.nfloat4):
            lora_dtype = qtype
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        if self.lora_alpha != 0:
            self.lora_A = HtColumnParallelLinear(self.in_features, rank, self.ds_parallel_config,
                                                 bias=False, gather_output=True, init_method='he_uniform_', 
                                                 dtype=lora_dtype, name='colp_lora_A')
            self.lora_B = HtColumnParallelLinear(rank, self.out_features, self.ds_parallel_config,
                                                 bias=False, gather_output=self.gather_output, init_method='zeros_', 
                                                 dtype=lora_dtype, name='colp_lora_B')
        if self.bias is not None:
            # self.bias = ori_model.bias
            self.bias.set_requires_grad(False)
            if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
                absmax_size = self.bias.global_numel() // blocksize
                if (self.bias.id not in map_absmax):
                    parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                    self.bias_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                               [absmax_size], self.ds_map['dup'], 
                                                               get_device_index(self.device_group),
                                                               dtype=hetu.float32, requires_grad=False,
                                                               parameter_dict = parameter_dict, 
                                                               device_group=self.device_group, 
                                                               name=f'{name}_bias_absmax')
                    map_absmax[self.bias.id] = self.bias_absmax
                else:
                    self.bias_absmax = map_absmax[self.bias.id]
                # self.bias.add_in_dep_linker(self.bias_absmax)
      
    def forward(self, input_p):
        output = HtColumnParallelLinear.forward(self, input_p)  
        if self.lora_alpha != 0:
            output = output + self.lora_B(self.lora_A(input_p)) * self.lora_alpha

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class LoRARowParallelLinear(HtRowParallelLinear):
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
    def __init__(self, ori_model, rank, blocksize = 64, 
                 lora_alpha = 1, qtype = hetu.bfloat16,
                 map_absmax = {}, name='rowp_lora'):
        HtRowParallelLinear.__init__(self, ori_model.in_features, ori_model.out_features, 
                                     ori_model.ds_parallel_config,
                                     bias=False if (ori_model.bias is None) else True,
                                     init_method='zeros_', dtype=ori_model.dtype, 
                                     name='rowp', ori_model = ori_model)
        self.rank = rank
        # self.weight = ori_model.weight
        self.weight.set_requires_grad(False)
        if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
            absmax_size = self.weight.global_numel() // blocksize
            if (self.weight.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weight.id, "blocksize": blocksize}
                self.weight_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                             [absmax_size], self.ds_map['dup'],
                                                             get_device_index(self.device_group), 
                                                             dtype=hetu.float32, requires_grad=False, 
                                                             parameter_dict = parameter_dict,
                                                             device_group=self.device_group, 
                                                             name=f'{name}_weight_absmax')
                map_absmax[self.weight.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weight.id]
            # self.weight.add_in_dep_linker(self.weight_absmax)
        
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hetu.float4) or (lora_dtype == hetu.nfloat4):
            lora_dtype = qtype
        if self.lora_alpha != 0:
            # lora_b_ds_parallel_config = self.ds_parallel_config.copy()
            # lora_b_ds_parallel_config["split"] = {}
            # if "0" in self.ds_parallel_config["split"]:
            #     lora_b_ds_parallel_config["split"]["1"] = self.ds_parallel_config["split"]["0"]
            # if "1" in self.ds_parallel_config["split"]:
            #     lora_b_ds_parallel_config["split"]["0"] = self.ds_parallel_config["split"]["1"]
            
            # self.lora_A = HtRowParallelLinear(self.in_features, rank, self.ds_parallel_config,
            #                                 bias=False, init_method='he_uniform_', 
            #                                 dtype=lora_dtype, name=f'rowp_lora_A_{self.name}')
            # self.lora_B = HtColumnParallelLinear(rank, self.out_features, lora_b_ds_parallel_config,
            #                                     bias=False, gather_output=True, init_method='zeros_', 
            #                                     dtype=lora_dtype, name=f'colp_lora_B_{self.name}')  
            self.lora_A = HtRowParallelLinear(self.in_features, rank, self.ds_parallel_config,
                                              bias=False, init_method='he_uniform_', 
                                              dtype=lora_dtype, name='rowp_lora_A')
            self.lora_B = HtRowParallelLinear(rank, self.out_features, self.ds_parallel_config,
                                              bias=False, init_method='zeros_', 
                                              dtype=lora_dtype, name='rowp_lora_B')      
        if self.bias is not None:
            # self.bias = ori_model.bias
            self.bias.set_requires_grad(False)
            if self.dtype == hetu.float4 or self.dtype == hetu.nfloat4:
                absmax_size = self.bias.global_numel() // blocksize
                if (self.bias.id not in map_absmax):
                    parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                    self.bias_absmax = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                               [absmax_size], self.ds_map['dup'],
                                                               get_device_index(self.device_group), 
                                                               dtype=hetu.float32, requires_grad=False,
                                                               parameter_dict = parameter_dict,
                                                               device_group=self.device_group, name=f'{name}_bias_absmax')
                    map_absmax[self.bias.id] = self.bias_absmax
                else:
                    self.bias_absmax = map_absmax[self.bias.id]
                # self.bias.add_in_dep_linker(self.bias_absmax)

    def forward(self, input_p):
        output = HtRowParallelLinear.forward(self, input_p)  
        if self.lora_alpha != 0:
            output = output + self.lora_B(self.lora_A(input_p)) * self.lora_alpha
        return output
    
class LoRAModel(Module):
    def __init__(self, ori_model, rank, blocksize = 64, 
                 lora_alpha = 1, qtype = hetu.bfloat16, name='lora_model'):
        super(LoRAModel, self).__init__()
        self.model = ori_model
        self.forward = self.model.forward
        self.state_dict = self.model.state_dict
        self.load_state_dict = self.model.load_state_dict
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        self.qtype = qtype
        self.id_to_absmax = {}
        self.prepare_for_lora(self.model)
            

    def prepare_for_lora(self, model):
        for key, model_ in model.named_modules():
            lora_model_ = None
            parent_model_ = None
            parent_key = ''
            if isinstance(model_, hetu.nn.HtParallelLayerNorm):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAParallelLayerNorm(model_, 
                                                    blocksize = self.blocksize, 
                                                    qtype = self.qtype,
                                                    map_absmax = self.id_to_absmax)
            # if isinstance(model_, hetu.nn.HtColumnParallelLinear):
            #     parent_key = ".".join(key.split(".")[:-1])
            #     alpha = 0
            #     if "attn" in parent_key:
            #         alpha = self.lora_alpha
            #     parent_model_ = model.find_module(parent_key)
            #     lora_model_ = LoRAColumnParallelLinear(model_, self.rank, 
            #                                            blocksize = self.blocksize, 
            #                                            lora_alpha = alpha, 
            #                                            qtype = self.qtype,
            #                                            map_absmax = self.id_to_absmax)
            if isinstance(model_, hetu.nn.HtRowParallelLinear):
                parent_key = ".".join(key.split(".")[:-1])
                alpha = 0
                if "attn" in parent_key:
                    alpha = self.lora_alpha
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRARowParallelLinear(model_, self.rank,
                                                    blocksize = self.blocksize, 
                                                    lora_alpha = alpha, 
                                                    qtype = self.qtype,
                                                    map_absmax = self.id_to_absmax)
            if isinstance(model_, hetu.nn.HtParallelEmbedding):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAParallelEmbedding(model_, self.rank, 
                                                    blocksize = self.blocksize,
                                                    map_absmax = self.id_to_absmax)
            if isinstance(model_, hetu.nn.HtVocabParallelEmbedding):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAVocabParallelEmbedding(model_, self.rank, 
                                                         blocksize = self.blocksize,
                                                         lora_alpha = 0,
                                                         qtype = self.qtype,
                                                         map_absmax = self.id_to_absmax)
            if parent_model_ is not None:
                setattr(parent_model_, key.split(".")[-1], lora_model_)
            model_ = lora_model_  
        # for key, model_ in self.model.named_modules():
        #     print("key:", key, "model:", model_) 
            

        