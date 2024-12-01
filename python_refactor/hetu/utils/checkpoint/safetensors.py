import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import hetu
import numpy as np

from safetensors import deserialize, safe_open, serialize, serialize_file
from collections import OrderedDict

WEIGHTS_NAME = 'hetu_pytorch_model'
WEIGHTS_FORMAT = '.safetensors'

def need_transfer(src_dtype, dst_dtype):
    if dst_dtype is None:
        return False
    if (src_dtype != hetu.float32) and \
       (src_dtype != hetu.float16) and \
       (src_dtype != hetu.bfloat16):
        return False
    if (src_dtype == dst_dtype):
        return False
    return True

def need_quantization(src_dtype, dst_dtype):
    if (not need_transfer(src_dtype, dst_dtype)):
        return False
    if (dst_dtype != hetu.float4) and (dst_dtype != hetu.nfloat4):
        return False
    return True

def data_transform_for_store(key, param, value, config, save_dtype, ht_state_dict, local_device):
    device_group = param.get_device_group()
    local_value = hetu.parallel_placeholder(param.dtype, global_shape=param.global_shape, 
                                            ds=param.distributed_states, device_group=device_group)
    feed_dict = {local_value: value}
    flag = 0
    if (param.dtype == hetu.float4) or (param.dtype == hetu.nfloat4):
        flag += 1
        key_absmax = key + "_absmax"
        weight_absmax = ht_state_dict[key_absmax]
        global_value = hetu.dequantization(local_value, weight_absmax, 
                                           config.dqtype, 64,
                                           device_group = device_group)
    else:
        global_value = local_value + 0
    key_to_model = ".".join(key.split("."))
    key_lora_A = key_to_model + ".lora_A.weight"
    key_lora_B = key_to_model + ".lora_B.weight"
    if key_lora_A in ht_state_dict:
        global_value = global_value + hetu.matmul(ht_state_dict[key_lora_A], 
                                                  ht_state_dict[key_lora_B],
                                                  device_group = device_group)
        
    if not param.distributed_states.is_pure_duplicate:
        flag += 2
        num_devices = device_group.num_devices
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        global_value = hetu.comm(global_value, ds_dup)
        global_value = global_value + 0 
    if ("qkv_dense") in key and ("lora" not in key) and ("absmax" not in key):
        flag += 4
        assert config != None, "There should be a config when using qkv_dense."
        num_heads, num_splits, hidden_size = 3, config.num_attention_heads, \
                                             config.hidden_size // config.num_attention_heads
        input_shape = global_value.shape
        saved_shape = [num_heads, num_splits, hidden_size] + input_shape[1:]
        global_value = global_value.reshape(saved_shape)
        perm = [i for i in range(len(saved_shape))]
        perm[0] = 1
        perm[1] = 0
        global_value = global_value.transpose(perm).contiguous()
        global_value = global_value.reshape(input_shape)
    if need_quantization(global_value.dtype, save_dtype):
        flag += 8
        global_values = global_value.quantization(save_dtype, 64)
        global_value = global_values[0]
        abs_max = global_values[1]
        abs_max = abs_max.to(dtype = abs_max.dtype, dev="cpu")
        global_value = global_value.to(dtype = global_value.dtype, dev="cpu")
        results = global_value.graph.run(global_value, [global_value, abs_max], feed_dict=feed_dict)
        return results[0], results[1]
    if need_transfer(global_value.dtype, save_dtype):
        global_value = global_value.to(dtype = save_dtype, dev="cpu")
    else:
        global_value = global_value.to(dtype = global_value.dtype, dev="cpu")
    results = global_value.graph.run(global_value, [global_value], feed_dict=feed_dict, num_micro_batches=1)
    return results[0], None

def change_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_heads * num_splits * hidden_size, :]
    # The original layout of param tensor is [num_splits * num_heads * hidden_size, :]
    input_shape = param.shape
    original_shape = [num_splits, num_heads, hidden_size] + input_shape[1:]
    ndim = len(original_shape)
    perm = [i for i in range(ndim)]
    perm[0] = 1 
    perm[1] = 0
    param = param.view(original_shape)
    param = param.transpose(perm)
    param = param.view(input_shape)
    return param


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop


def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]


def _filter_shared_not_shared(tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors


def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if v.device != torch.device("meta") and storage_ptr(v) != 0 and storage_size(v) != 0:
            # Need to add device as key because of multiple GPU.
            tensors[(v.device, storage_ptr(v), storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors


def _is_complete(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() == storage_ptr(tensor) and tensor.nelement() * _SIZE[tensor.dtype] == storage_size(tensor)


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: Optional[List[str]] = None,
    discard_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                " since you could be storing much more memory than needed. Please refer to"
                " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                " issue."
            )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def save_model(
    model: hetu.nn.Module, filename: str, config=None, 
    local_device=None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None
):
    with hetu.graph("define_and_run", create_new=True, prefix="save_model"):
        global_state_dict = OrderedDict()
        all_device_groups = []
        state_dict = model.state_dict()
        ht_state_dict = model.state_dict(format='hetu')
        total_sum = len(state_dict.items())
        cur_sum = 0
        visit_tensors = set()

        for key, data in state_dict.items():
            # absmax need recompute
            if "absmax" in key:
                continue
            # only_lora mode only save loraA and loraB 
            if only_lora and ("lora" not in key):
                continue
            if (not only_lora) and ("lora" in key):
                continue
            param = ht_state_dict[key]
            device_group = param.get_device_group()
                
            if device_group not in all_device_groups:
                all_device_groups.append(device_group)
            # TODO: implement allgather_inter_group_param()
            if not device_group.contains(local_device):
                continue
            
            global_value = param
            # shared tensors only compute once
            if param.id in visit_tensors:
                continue
            visit_tensors.add(param.id)

            global_value, abs_max = data_transform_for_store(key, param, data, config, save_dtype, 
                                                             ht_state_dict, local_device)
            global_state_dict[key] = global_value.numpy(force=True, save=True)
            if ((save_dtype == hetu.float4) or (save_dtype == hetu.nfloat4)) and \
               (param.dtype != save_dtype):
                abs_max_key = key + "_absmax"
                global_state_dict[abs_max_key] = abs_max.numpy(force=True, save=True)
            cur_sum += 1

        if force_contiguous:
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        for i, device_group in enumerate(all_device_groups):
            if device_group.contains(local_device):
                if device_group.get_index(local_device) == 0:
                    archive_file = WEIGHTS_NAME + f'-{i + 1}-of-{len(all_device_groups)}' + WEIGHTS_FORMAT
                    archive_file = os.path.join(
                        filename, archive_file
                    )
                    try:
                        save_file(global_state_dict, archive_file, metadata=metadata)
                    except ValueError as e:
                        msg = str(e)
                        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                        raise ValueError(msg)


def load_model(model: hetu.nn.Module, filename: Union[str, os.PathLike],
               config=None, local_device=None, strict=True) -> Tuple[List[str], List[str]]:
    local_state_dict = model.state_dict()
    all_device_groups = []
    parameter_to_dtype = {}
     
    for k in local_state_dict:
        param = model.state_dict(format='hetu')[k]
        parameter_to_dtype[k] = param.dtype
        device_group = param.get_device_group()
        if device_group not in all_device_groups:
            all_device_groups.append(device_group)
        # TODO: implement allgather_inter_group_param()
        if not device_group.contains(local_device):
            continue

    for i, device_group in enumerate(all_device_groups):
        if device_group.contains(local_device):
            archive_file = WEIGHTS_NAME + f'-{i + 1}-of-{len(all_device_groups)}' + WEIGHTS_FORMAT
            archive_file = os.path.join(
                filename, archive_file
            )
            state_dict = load_file(archive_file, parameter_to_dtype)
            for k in state_dict:
                # TODO: maybe implement it elsewhere
                # qkv_dense的weight和bias原先是[3 * num_heads * hidden_size, :]
                # 要先转化成为[num_heads * 3 * hidden_size, :]才可以
                # 因为会在num_heads这一维度上进行distributed tensor的split切分
                if ("qkv_dense" in k) and ("absmax" not in k):
                    assert config != None, "There should be a config when using qkv_dense."
                    state_dict[k] = change_query_key_value_ordering(state_dict[k], 
                                                                    3, 
                                                                    config.num_attention_heads, 
                                                                    config.hidden_size // config.num_attention_heads)
            model.load_state_dict(state_dict, local_device, False)
    return model


def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.torch import save
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    serialized = serialize(_flatten(tensors), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.torch import save_file
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    if not os.path.exists(filename):
        open(filename, "w").close()
    for k, v in tensors.items():
        print("save:", k, v.dtype, v.flatten()[:10])
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensors.items()}
    serialize_file(flattened, filename, metadata=metadata)

def load_file(filename: Union[str, os.PathLike], parameter_to_dtype, device="cpu") -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in f.keys():
            print("load:", k, f.get_tensor(k).dtype, parameter_to_dtype[k], f.get_tensor(k).flatten()[:10])
            result[k] = hetu.numpy_to_NDArray(f.get_tensor(k), parameter_to_dtype[k])
    return result


def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor` on cpu

    Example:

    ```python
    from safetensors.torch import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
    return _view2torch(flat)

_SIZE = {
    hetu.int64: 8,
    hetu.float32: 4,
    hetu.int32: 4,
    hetu.bfloat16: 2,
    hetu.float16: 2,
    hetu.int16: 2,
    hetu.uint8: 1,
    hetu.int8: 1,
    hetu.bool: 1,
    hetu.float64: 8,
    hetu.float4: 1,
    hetu.nfloat4: 1,
}

_TYPES = {
    "F64": hetu.float64,
    "F32": hetu.float32,
    "F16": hetu.float16,
    "BF16": hetu.bfloat16,
    "I64": hetu.int64,
    # "U64": hetu.uint64,
    "I32": hetu.int32,
    # "U32": hetu.uint32,
    "I16": hetu.int16,
    # "U16": hetu.uint16,
    "I8": hetu.int8,
    "U8": hetu.uint8,
    "BOOL": hetu.bool,
    "F4": hetu.float4,
    "NF4": hetu.nfloat4,
}


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = hetu.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr

    return result

def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == "=":
        if sys.byteorder == "little":
            return True
        else:
            return False
    elif byteorder == "|":
        return True
    elif byteorder == "<":
        return True
    elif byteorder == ">":
        return False
    raise ValueError(f"Unexpected byte order {byteorder}")

def _tobytes(tensor: np.ndarray) -> bytes:
    if not _is_little_endian(tensor):
        tensor = tensor.byteswap(inplace=False)
    return tensor.tobytes()

def _flatten(tensors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected a dict of [str, hetu.NDArray] but received {type(tensors)}")

    for k, v in tensors.items():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Key `{k}` is invalid, expected hetu.NDArray but received {type(v)}")

    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }