# TODO: output the library to `hetu` directory
import _hetu_core
from _hetu_core import *

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
_tensor_type_t = TypeVar('T', bound='Tensor')

from hetu import nn as nn
from hetu import utils as utils
# from hetu import optim as optim

import builtins # bool is resovled as hetu.bool

cur_graph_contexts = []

class _OpContext(object):
    def __init__(self, eager_device=None, devices=None, stream_index=None, extra_deps=None):
        self.eager_device = eager_device
        self.devices = devices
        self.stream_index = stream_index
        self.extra_deps = extra_deps
        if devices is not None:
            self.devices = DeviceGroup(devices)
        if stream_index is not None:
            if ((not isinstance(stream_index, builtins.int)) or 
                isinstance(stream_index, builtins.bool)):
                raise ValueError(
                    "Expected 'int' type for stream index, got "
                    f"'{type(stream_index).__name__}'")
        if extra_deps is not None:
            if not isinstance(extra_deps, (list, tuple)):
                raise ValueError("Extra dependencies should be lists or tuples of tensors")
            for x in extra_deps:
                if not isinstance(x, Tensor):
                    raise ValueError(f"'{type(x).__name__}' object is not a tensor")
            self.extra_deps = [group(extra_deps)]

    def __enter__(self):
        _hetu_core._internal_context.push_op_ctx(
            eager_device=self.eager_device,
            device_groups=self.devices, 
            stream_index=self.stream_index, 
            extra_deps=self.extra_deps)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_op_ctx(
            pop_eager_device=(self.eager_device is not None),
            pop_device_groups=(self.devices is not None), 
            pop_stream_index=(self.stream_index is not None), 
            pop_extra_deps=(self.extra_deps is not None))

def context(eager_device=None, devices=None, stream_index=None, extra_deps=None):
    return _OpContext(eager_device=eager_device, devices=devices, stream_index=stream_index, extra_deps=extra_deps)

def control_dependencies(control_inputs):
    return _OpContext(extra_deps=control_inputs)

class _GraphContext(object):
    def __init__(self, g, create_new=False, prefix='default', num_strategy=-1):
        if isinstance(g, Graph):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = g
        elif isinstance(g, builtins.str):
            if create_new:
                if g == "define_and_run":
                    self.graph = _hetu_core._internal_context.make_new_define_and_run_graph(prefix + '_' + g)
                else:
                    raise NotImplementedError("Can only create define_and_run graph for now.")
            else:
                if prefix == 'default':
                    if g == "eager":
                        self.graph = _hetu_core._internal_context.get_default_eager_graph()
                    elif g == "define_and_run":
                        self.graph = _hetu_core._internal_context.get_default_define_and_run_graph()
                    elif g == "define_by_run":
                        self.graph = _hetu_core._internal_context.get_default_define_by_run_graph()
                else:
                     self.graph = _hetu_core._internal_context.get_graph(prefix + '_' + g)
        elif isinstance(g, builtins.int):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = _hetu_core._internal_context.get_graph(g)
        else:
            raise ValueError(f"Cannot parse type '{type(g).__name__}' as hetu.Graph")
        if num_strategy >= 1:
          # print(f'before set num_startegy: {self.graph.num_strategy}')
          self.graph.set_num_strategy(num_strategy)
          # print(f'after set num_startegy: {self.graph.num_strategy}, cur_strategy_id: {self.graph.cur_strategy_id}')

    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            if str(self.graph) != str(cur_graph_contexts[0].graph):
                print(f"Caution! You wrap {self.graph} graph in {cur_graph_contexts[0].graph} graph, we just return you the latter!")
            self.wrapped_context = True
            return cur_graph_contexts[0]
        self.wrapped_context = False
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        if self.wrapped_context:
            return
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)

def graph(g, create_new=False, prefix='default', num_strategy=-1):
    return _GraphContext(g, create_new=create_new, prefix=prefix, num_strategy=num_strategy)

class _AutocastContext(object):
    def __init__(self, dtype = None):
        if dtype is None:
            self.autocast = _hetu_core._internal_context.get_default_autocast()
        else:
            self.autocast = _hetu_core._internal_context.make_new_autocast(True, dtype)

    def __enter__(self):
        _hetu_core._internal_context.push_autocast_ctx(self.autocast.id)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_autocast_ctx()

def autocast():
    return _AutocastContext()

def autocast(dtype):
    return _AutocastContext(dtype)

class _RunLevel(object):
    def __init__(self, type):
        if type == "update":
            self.type = 0
        elif type == "grad":
            self.type = 1
        elif type == "compute_only":
            self.type = 2
        elif type == "alloc":
            self.type = 3
        elif type == "topo":
            self.type = 4
        else:
            raise ValueError(f"No level for '{type}' in hetu.Graph")
        
def run_level(type = "update"):
    return _RunLevel(type).type

class _RecomputeContext(object):
    def __enter__(self):
        _hetu_core._internal_context.push_recompute_ctx()
        return self

    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_recompute_ctx()

def recompute():
    return _RecomputeContext()

class _CPUOffloadContext(object):
    def __enter__(self):
        _hetu_core._internal_context.push_cpu_offload_ctx()
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_cpu_offload_ctx()

def cpu_offload():
    return _CPUOffloadContext()

class _ProfileContex(object):
    def __init__(self, enabled : bool = True, use_cpu : bool = False, use_cuda : bool = False,
                 record_shapes : bool = False , profile_memory : bool = False):
      self.profile = _hetu_core._internal_context.make_new_profile(enabled, use_cpu, use_cuda, record_shapes, profile_memory)

    def __enter__(self):
        _hetu_core._internal_context.push_profile_ctx(self.profile.id)
        return self.profile

    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_profile_ctx()


def profiler(enabled : bool = True, use_cpu : bool = False, use_cuda : bool = False,
             record_shapes : bool = False , profile_memory : bool = False):
    return _ProfileContex(enabled, use_cpu, use_cuda, record_shapes, profile_memory)