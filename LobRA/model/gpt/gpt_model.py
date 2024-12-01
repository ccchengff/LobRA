import hetu as ht
import numpy as np
from hetu.nn.modules.parallel_utils import get_multi_ds_parallel_config, parallel_multi_data_provider, get_device_index
  
# self-attn
class GPTAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.name = name
        self.use_flash_attn = config.use_flash_attn
        self.add_bias = False

        # max_positions = config.max_position_embeddings
        # self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
        #             1, 1, max_positions, max_positions))
        self.masked_value = -1e4

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.q_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_q_{name}'
        )

        self.k_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_k_{name}'
        )
        
        self.v_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_v_{name}'
        )
        
        self.o_proj = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
            bias=self.add_bias,
            name=f'rowp_base_o_{name}'
        )

        self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key_t, value, attention_mask=None):
        # q*k^T, shape=[micro_batch_size, num_heads, seq_len, seq_len]
        attn_weights = ht.bmm(query, key_t)
        micro_batch_size, num_heads, seq_len, seq_len = attn_weights.global_shape

        # scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.global_shape[-1]) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # mask
        device_index = get_device_index(self.qkv_dense.device_groups[0])
        # todo: move causal_mask outside and turn to a placeholder
        causal_mask = ht.from_numpy_parallel(parallel_multi_data_provider(
                                               np.tile(self.bias[:, :, :seq_len, :seq_len], 
                                                 (micro_batch_size, num_heads, 1, 1)),
                                               attn_weights.multi_distributed_states,
                                               self.qkv_dense.device_groups),
                                             attn_weights.multi_distributed_states, requires_grad=False,
                                             device_groups=self.qkv_dense.device_groups, name='causal_mask')
        
        # todo: move mask outside and turn to a placeholder
        mask = ht.from_numpy_parallel(parallel_multi_data_provider(
                                        np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
                                        attn_weights.multi_distributed_states, 
                                        self.qkv_dense.device_groups), 
                                      attn_weights.multi_distributed_states, requires_grad=False,
                                      device_groups=self.qkv_dense.device_groups, name='mask')        
        attn_weights = ht.where(causal_mask, attn_weights, mask)
        if attention_mask is not None:
            # attn_weights: shape=[micro_batch_size, num_heads, seq_len, seq_len]
            # attention_mask: shape=[micro_batch_size, 1, 1, seq_len], 注意ds的设置
            # 被mask的<pad>位置上值为-1e4, 没有被mask的位置上值为0
            # todo: +-*/允许对应维度一个为n一个为1的情况下, n被切分
            # print(f'attn_weights global_shape={attn_weights.global_shape}, attention_mask.global_shape={attention_mask.global_shape}')
            # print(f'attn_weights shape={attn_weights.shape}, attention_mask.shape={attention_mask.shape}')
            attn_weights = attn_weights + attention_mask
        # softmax
        attn_weights = ht.softmax(attn_weights, 3)
        # dropout
        # attn_weights = self.attn_dropout(attn_weights)
        # weight sum, shape=[micro_batch_size, num_heads, seq_len, head_dim]
        attn_output = ht.bmm(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        task_batch_idxs,
        mbs_times_dp_symbol,
        seq_len_symbol,
        attention_mask=None,
    ):
        # embed_dim = hidden_states.global_shape[-1]
        # mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        # mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * 2
        # seq_len_symbol = hidden_states.symbolic_shape[1]
        # [micro_batch_size*seq_len, embed_dim]
        # hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(embed_dim)], name=f'reshape_{self.name}')
        # print(f'hidden_states.global_shape={hidden_states.global_shape}, hidden_states.shape={hidden_states.shape}, hidden_states.distributed_states={hidden_states.distributed_states}')        
        # column parallel, [micro_batch_size*seq_len, 3*embed_dim]
        q = self.q_proj(hidden_states, task_batch_idxs=task_batch_idxs)
        k = self.k_proj(hidden_states, task_batch_idxs=task_batch_idxs)
        v = self.v_proj(hidden_states, task_batch_idxs=task_batch_idxs)
        
        query = q.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(self.head_dim)], name=f'reshape_q_{self.name}')
        key = k.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(self.head_dim)], name=f'reshape_k_{self.name}')
        value = v.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(self.head_dim)], name=f'reshape_v_{self.name}')

        if self.use_flash_attn:
            attn_output = ht.attn(query, key, value, 0, -1, True, name=f'attn_{self.name}')[0]
        else:
            # [micro_batch_size, num_heads, seq_len, head_dim]
            query = query.transpose([0, 2, 1, 3], name="AttentionOp_query")
            value = value.transpose([0, 2, 1, 3], name="AttentionOp_value")
            # [micro_batch_size, num_heads, head_dim, seq_len]
            key_t = key.transpose([0, 2, 3, 1], name="AttentionOp_key") # k^T

            # self-attn, shape=[micro_batch_size, num_heads, seq_len, head_dim]
            attn_output, attn_weights = self._attn(query, key_t, value, attention_mask)

            # [micro_batch_size, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose([0, 2, 1, 3])
        
        # [micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([mbs_times_dp_symbol * seq_len_symbol, ht.IntSymbol(self.num_heads * self.head_dim)], name=f'reshape_{self.name}')
        # row parallel, shape=[micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = self.o_proj(attn_output, task_batch_idxs=task_batch_idxs)
        # [micro_batch_size, seq_len, num_heads*head_dim]
        # attn_output = attn_output.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(self.num_heads * self.head_dim)], name=f'reshape_{self.name}')
        # dropout
        # attn_output = self.resid_dropout(attn_output)

        # [micro_batch_size, seq_len, num_heads*head_dim]
        return attn_output

class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.name = name
        self.add_bias = False

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_h_to_4h_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        # self.activation_func = ht.nn.NewGeLU()
        
        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            bias=self.add_bias,
            name=f'rowp_base_4h_to_h_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, task_batch_idxs):
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states, task_batch_idxs=task_batch_idxs)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        intermediate_parallel = ht.relu(intermediate_parallel, name=f'relu_{self.name}')

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel, task_batch_idxs=task_batch_idxs)
        # output = self.dropout(output)
        return output

class GPTMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(GPTMLP, self).__init__()
        self.config = config
        self.name = name
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states, task_batch_idxs):
        # origin_shape = hidden_states.global_shape # [b, seq_len, hidden_size]
        # if len(origin_shape) != 2: # shape adaptor
        #     hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(origin_shape[-1])], name=f'reshape1_{self.name}')
        hidden_states = self.parallel_mlp(hidden_states, task_batch_idxs=task_batch_idxs)
        # if len(origin_shape) != 2: # shape adaptor
        #     # two undetermined dim, we therefore should use symbolic shape here
        #     hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(origin_shape[-1])], name=f'reshape2_{self.name}')
        return hidden_states

class GPTBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.name = f'GPTBlock{layer_idx}'
        hidden_size = config.hidden_size

        self.ln_1 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), eps=config.layer_norm_epsilon, name=f'ln1_{self.name}')
        self.attn = GPTAttention(config, ds_parallel_configs, layer_idx=layer_idx, name=f'GPTAttn{layer_idx}_{self.name}')
        self.ln_2 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), eps=config.layer_norm_epsilon, name=f'ln2_{self.name}')
        self.mlp = GPTMLP(config, ds_parallel_configs, layer_idx=layer_idx, name=f'GPTMLP{layer_idx}_{self.name}')

    def forward(
        self,
        hidden_states,
        task_batch_idxs,
        mbs_times_dp_symbol,
        seq_len_symbol,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [bsz*seq_len, hidden_size]
            task_batch_idxs,
            mbs_times_dp_symbol,
            seq_len_symbol,
            attention_mask=attention_mask, # [b, 1, 1, seq_len]
        )
        # residual connection
        # hidden_states = attn_output + residual
        hidden_states = ht.add(attn_output, residual, name=f'add_{self.name}')

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, task_batch_idxs=task_batch_idxs)
        # residual connection
        # hidden_states =  feed_forward_hidden_states + residual
        # hidden_states =  residual + feed_forward_hidden_states
        hidden_states = ht.add(residual, feed_forward_hidden_states, name=f'add_{self.name}')

        return hidden_states


class GPTModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(GPTModel, self).__init__()
        self.config = config

        self.embed_dim = config.hidden_size
        self.wte = ht.nn.HtMultiVocabParallelEmbedding(config.vocab_size, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        self.wpe = ht.nn.HtMultiParallelEmbedding(config.max_position_embeddings, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

        self.drop = ht.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
            blocks.append(GPTBlock(config, ds_parallel_configs, layer_idx=i))
            # for _, block_config in ds_parallel_config['blocks'].items():
            #     if i >= block_config['range'][0] and i <= block_config['range'][1]:
            #         blocks.append(GPTBlock(config, block_config, layer_idx=i))
            #         break
        self.h = ht.nn.ModuleList(blocks)
        self.ln_f = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm_final'), eps=config.layer_norm_epsilon, name='ln_final')

    def forward(
        self,
        input_ids,
        position_ids,
        task_batch_idxs,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b, seq_len]        
        # token_type_ids: [b, seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # attention_mask: [b, 1, 1, seq_len]
        if attention_mask is not None:
            assert attention_mask.global_shape == input_ids.global_shape \
                and attention_mask.distributed_states.check_equal(attention_mask.distributed_states), \
                'attention_mask global_shape and distributed_states should be equal to input_ids!'
            mbs_times_dp_symbol = input_ids.symbolic_shape[0] * self.config.dp_symbol
            # attention_mask = attention_mask.reshape([input_ids.symbolic_shape[0] * 2, ht.IntSymbol(1), ht.IntSymbol(1), input_ids.symbolic_shape[1]])
            attention_mask = attention_mask.reshape([mbs_times_dp_symbol, ht.IntSymbol(1), ht.IntSymbol(1), input_ids.symbolic_shape[1]])
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embeddding: [b, seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b, seq_len, embed_dim]
        position_embeds = self.wpe(position_ids) # [b, seq_len, embed_dim]
        # todo: fix backward grad tensor reduce bug for add(extension dims)
        hidden_states = inputs_embeds + position_embeds # [b, seq_len, embed_dim]
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b, seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        # dropout
        # hidden_states = self.drop(hidden_states)
        mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        seq_len_symbol = hidden_states.symbolic_shape[1]
        embed_dim = hidden_states.global_shape[-1]
        hidden_states = hidden_states.reshape([mbs_times_dp_symbol * seq_len_symbol, ht.IntSymbol(embed_dim)])
        
        # for sp
        if hidden_states.check_multi_ds_equal(self.h[0].ln_1.ds_map['split0_or_split0_dup']):
            hidden_states = hidden_states
        else:
            hidden_states = ht.comm(hidden_states, self.h[0].ln_1.ds_map['split0_or_split0_dup'])

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b*seq_len, embed_dim]
                task_batch_idxs,
                mbs_times_dp_symbol,
                seq_len_symbol,
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
        # layernorm
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(embed_dim)])
        return hidden_states

class GPTLMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPTModel(config, ds_parallel_configs)
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        self.lm_head.weight = self.transformer.wte.embedding_table # share embedding table
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        task_batch_idxs=None
    ):
        # [b, seq_len, n_embd]
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            task_batch_idxs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [b, s, h] -> [b, s-1, h]
        shift_hidden_states = ht.slice(hidden_states, [ht.IntSymbol(0), ht.IntSymbol(0), ht.IntSymbol(0)], [hidden_states.symbolic_shape[0], hidden_states.symbolic_shape[1] - 1, hidden_states.symbolic_shape[2]])
        hidden_size = hidden_states.global_shape[-1]
        mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        seq_len_symbol = hidden_states.symbolic_shape[1]
        # [b*(s-1), h]
        shift_hidden_states = shift_hidden_states.reshape([mbs_times_dp_symbol * (seq_len_symbol - 1), ht.IntSymbol(hidden_size)])
        # column parallel, [b*(s-1), h]->[b*(s-1), vocab_size], and splited in vocab dimension
        shift_lm_logits = self.lm_head(shift_hidden_states, task_batch_idxs=task_batch_idxs)

        total_loss = None
        if labels is not None:
            if self.config.train_task_num == 1:
                shift_labels = ht.slice(labels, [ht.IntSymbol(-1), ht.IntSymbol(1)], \
                                        [ht.IntSymbol(-1), labels.symbolic_shape[1] - 1], name=f'shift_labels')
                loss = ht.vocab_parallel_cross_entropy(slice_shift_lm_logits,  
                        shift_labels, ignored_index = -1, reduction = "mean", name=f"vocab_cross_entropy")
                total_loss = loss
            else:
                for i in range(self.config.train_task_num):
                    shift_labels = ht.slice(labels, task_batch_idxs[i], [ht.IntSymbol(-1), ht.IntSymbol(1)], \
                                            [ht.IntSymbol(-1), labels.symbolic_shape[1] - 1], name=f'slice_shift_labels_{i}')
                    slice_shift_lm_logits = ht.slice(shift_lm_logits, task_batch_idxs[i], [ht.IntSymbol(-1), ht.IntSymbol(0)], \
                                            [ht.IntSymbol(-1), shift_lm_logits.symbolic_shape[1]], name=f'slice_shift_lm_logits_{i}')
                    loss = ht.vocab_parallel_cross_entropy(slice_shift_lm_logits,  
                        shift_labels, ignored_index = -1, reduction = "mean", name=f"vocab_cross_entropy_task{i}")
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss = ht.add(total_loss, loss, name=f'loss_add_{i}')
        # output = (shift_lm_logits,)
        # output = ((loss,) + output) if loss is not None else output
        return total_loss # ((loss), (shift_lm_logits))