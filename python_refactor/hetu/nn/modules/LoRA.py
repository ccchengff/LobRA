import hetu
from hetu import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'LoRALinear', 
]

class LoRALinear(Module):
    
    def __init__(self, in_features: int, out_features: int, rank: int, p: float = 0, bias: bool = True, device_group = None) -> None:
        with hetu.graph("define_and_run"):
            super(LoRALinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            # self.weight = hetu.nn.functional.kaiming_uniform_([out_features, in_features], a = math.sqrt(5), requires_grad=True, device_group = device_group)
            self.weight = hetu.nn.functional.kaiming_uniform_([out_features, in_features], 
                                                              a = math.sqrt(5), requires_grad=False, 
                                                              device_group = device_group)
            self.weightA = hetu.nn.functional.kaiming_uniform_([rank, in_features], 
                                                               a = math.sqrt(5), requires_grad=True, 
                                                               device_group = device_group)
            self.weightB = hetu.zeros([out_features, rank], requires_grad=True, device_group = device_group)
            if p < 0 or p > 1:
                raise ValueError("dropout probability has to be between 0 and 1, "
                                "but got {}".format(p))
            self.p = 1 - p
            if bias:
                fan_in, _ = hetu.nn.functional._calculate_fan_in_and_fan_out(self.weight.shape)
                bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias = hetu.rand([out_features], -bound, bound, requires_grad=True, device_group = device_group)
            else:
                self.register_parameter('bias', None)
    
    
    def forward(self, input: Tensor, weight: Tensor = None) -> Tensor:
        if self.p < 1:
            input = hetu.dropout(input, self.p, False)
        if self.bias is not None:
            return hetu.matmul(input, self.weight, trans_b=True) + \
                    hetu.matmul(hetu.matmul(input, self.weightA, trans_b=True), 
                                self.weightB, trans_b=True) + self.bias
        else:
            return hetu.matmul(input, self.weight, trans_b=True) + \
                    hetu.matmul(hetu.matmul(input, self.weightA, trans_b=True), 
                                self.weightB, trans_b=True)


class LoRAEmbedding(Module):
    
    def __init__(self, num_embeddings, embedding_dim, rank, device_group = None) -> None:
        with hetu.graph("define_and_run"):
            super(LoRAEmbedding, self).__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = hetu.nn.functional.xavier_normal_([num_embeddings, embedding_dim], 
                                                            requires_grad=False, device_group = device_group)
            self.weightA = hetu.zeros([num_embeddings, rank], requires_grad=True, device_group = device_group)
            self.weightB = hetu.randn([embedding_dim, rank], 0, 1, requires_grad=True, device_group = device_group)
    
    def forward(self, input: Tensor) -> Tensor:
        return hetu.embedding_lookup(self.weight, input) + \
               hetu.matmul(hetu.embedding_lookup(self.weightA, input), self.weightB, trans_b=True)
