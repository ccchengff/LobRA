import hetu
from queue import Queue

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds_parallel_config = []
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                multi_ds_parallel_config.append(config[module_name])
                break
            else:
                for value in config.values():
                    if type(value) == dict:
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][1]):
                            continue
                        config_queue.put(value)
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), 'ds_parallel_configs parse error!'
    return multi_ds_parallel_config

def parallel_data_provider(global_data, ds, device_index):
    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        local_data = local_data.take(range(start, stop), axis=dim)
    return local_data

def parallel_multi_data_provider(global_data, multi_ds, device_groups):
    multi_local_data = []
    for i in range(len(multi_ds)):
        ds = multi_ds[i]
        device_group = device_groups[i]
        device_index = get_device_index(device_group)
        order, states = ds.order, ds.states
        local_map = hetu.map_to_local_data(ds, device_index)
        local_data = global_data.copy()
        for dim in order:
            if dim < 0:
                continue
            splits = states[dim]
            split_index = local_map[dim]
            start = int(split_index * (global_data.shape[dim] / splits))
            stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
            local_data = local_data.take(range(start, stop), axis=dim)
        multi_local_data.append(local_data)
    return multi_local_data

def get_device_index(device_group):
    local_device = hetu.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    num_devices = len(config['device_group'])
    zero = False
    split = {}
    for key, value in config['split'].items():
        split[int(key)] = value
    states = {-1: config['dup'], **split}
    if config['type'] == 'placeholder':
        order = sorted(split.keys()) + [-1]
    elif config['type'] == 'variable':
        order = [-1] + sorted(split.keys())
        assert 'zero' in config, f"variable config must have zero!"
        zero = config['zero']
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")
    ds = hetu.DistributedStates(num_devices, states, order, zero)
    
    all_devices = hetu.global_device_group()
    device_group = hetu.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group']])
    return ds, device_group