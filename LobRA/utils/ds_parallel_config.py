import os
import json
import socket
import argparse
import numpy as np
import hetu as ht
from dataclasses import dataclass, field
from typing import List
from queue import Queue
from hetu.nn.modules.parallel_utils import get_multi_ds_parallel_config, config2ds

def distributed_init(use_two_node: bool = False, use_tencent: bool = False):
    if use_tencent:
        hostname = socket.gethostname()
        os.environ['HETU_LOCAL_HOSTNAME'] = os.environ['LOCAL_HOSTNAME']
    elif use_two_node:
        hostname = socket.gethostname()
        if hostname == 'job-83e1033f-9636-44b3-bf8b-2b627707b95f-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-0'
        elif hostname == 'job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-1'
        else:
            raise ValueError(f"Unknown hostname: {hostname}")

    ht.init_comm_group(8)

def assign_global_to_all_variables(ds_parallel_config):
    zero = ds_parallel_config['zero']
    sp = ds_parallel_config['sp']
    # assign zero to all variables
    config_queue = Queue()
    for value in ds_parallel_config.values():
        config_queue.put(value)
    while (not config_queue.empty()):
        config = config_queue.get()
        if type(config) == dict:
            if 'type' in config:
                if config['type'] == 'variable' and ('zero' not in config or 'sp' not in config):
                    config['zero'] = zero
                    config['sp'] = sp
                if 'lora' in config:
                    config_queue.put(config['lora'])
            else:
                for value in config.values():
                    config_queue.put(value)
    return ds_parallel_config

def get_ds_parallel_degrees(ds_parallel_configs):
    num_strategy = len(ds_parallel_configs)
    num_gpus = 0
    sp = ds_parallel_configs[0]['sp']
    dps, tps, pps = [], [], []
    for ds_parallel_config in ds_parallel_configs:
        dp = ds_parallel_config['input']['split'].get('0', 1)
        tp = ds_parallel_config['input']['dup']
        pp = len(ds_parallel_config['gpt']['blocks'])
        cur_sp = ds_parallel_config['sp']
        assert cur_sp == sp, \
            "sp should be same for multi-strategy"
        dps.append(dp)
        tps.append(tp)
        pps.append(pp)
        num_gpus += dp * tp * pp
    return num_strategy, dps, tps, pps, sp, num_gpus

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    config_paths = args.ds_parallel_config.split('@')
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = json.load(open(config_path, 'r'))
        ds_parallel_config = assign_global_to_all_variables(ds_parallel_config)
        ds_parallel_configs.append(ds_parallel_config)
    return ds_parallel_configs

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds = []
    device_groups = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds, device_group = config2ds(ds_parallel_config)
        multi_ds.append(ds)
        device_groups.append(device_group)
    return multi_ds, device_groups

def generate_ds_parallel_config(num_layers=16, num_gpus=8, dps=[], tps=[], pps=[], sps=[], zero=False):
    if num_gpus >= 8:
        num_nodes = int(num_gpus // 8)
    else:
        num_nodes = 1
    ds_configs = zip(dps, tps, pps, sps)
    ds_config_indexs = np.array(tps)
    ds_config_indexs = np.argsort(-ds_config_indexs)
    ds_configs = sorted(ds_configs, key=lambda x: -x[1])
    multi_device_groups = []
    node_idxs = [0] * num_nodes
    for (dp, tp, pp, _) in ds_configs:
        device_groups = []
        for p in range(pp):
            device_groups.append([])
        for d in range(dp):
            for p in range(pp):
                idx = p % num_nodes
                cnt = 0
                while node_idxs[idx] + tp > 8 and cnt <= num_nodes:
                    cnt += 1
                    idx = (idx + 1) % num_nodes
                node_idx = node_idxs[idx]
                device_group = list(range(node_idx + idx * 8, node_idx + idx * 8 + tp))
                node_idxs[idx] += tp
                device_groups[p] += device_group
        multi_device_groups.append(device_groups)
    # print(f"device groups: {multi_device_groups}")

    ds_parallel_configs = []

    for i, (dp, tp, pp, sp) in enumerate(ds_configs):
        num_layers_per_stage = num_layers // pp
        device_groups = multi_device_groups[i]
        devices = []
        for device_group in device_groups:
            devices += device_group
        ds_parallel_config = {
            'zero': zero,
            'sp': sp,
            'devices': devices,
            'task_batch_idxs': {
                'split': {},
                'dup': dp * tp * pp,
                'device_group': devices,
                'type': 'placeholder'
            },
            'input': {
                'split': {'0': dp},
                'dup': tp,
                'device_group': device_groups[0],
                'type': 'placeholder'
            },
            'gpt': {
                'wte': {
                    'split': {'0': tp},
                    'dup': dp,
                    'device_group': device_groups[0],
                    'type': 'variable'
                },
                'wpe': {
                    'split': {},
                    'dup': dp * tp,
                    'device_group': device_groups[0],
                    'type': 'variable'
                },
                'blocks': {
                    
                },
                'layernorm_final': {
                    'split': {},
                    'dup': dp * tp,
                    'tp': tp,
                    'device_group': device_groups[-1],
                    'type': 'variable'
                }
            },
            'lm_head': {
                'split': {'1': tp},
                'dup': dp,
                'device_group': device_groups[-1],
                'type': 'variable'
            },
            'label': {
                'split': {'0': dp},
                'dup': tp,
                'device_group': device_groups[-1],
                'type': 'placeholder'
            }
        }
        
        for stage_id in range(pp):
            block_start_id = num_layers_per_stage * stage_id
            block_end_id = num_layers_per_stage * (stage_id + 1) - 1
            blocks_json = ds_parallel_config['gpt']['blocks']
            blocks_json[f'blocks{block_start_id}-{block_end_id}'] = {
                "range": [block_start_id, block_end_id],
                "layernorm1": {
                    "split": {},
                    "dup": dp * tp,
                    "tp": tp,
                    "device_group": device_groups[stage_id],
                    "type": "variable"
                },
                "attn": {
                    "qkv": {
                        "split": {"1": tp},
                        "dup": dp,
                        "device_group": device_groups[stage_id],
                        "type": "variable",
                        "lora": {
                            "lora_A": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            },
                            "lora_B": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            }
                        }
                    },
                    "dense": {
                        "split": {"0": tp},
                        "dup": dp,
                        "device_group": device_groups[stage_id],
                        "type": "variable",
                        "lora": {
                            "lora_A": {
                                "split": {"0": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            },
                            "lora_B": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            }
                        }
                    }
                },
                "layernorm2": {
                    "split": {},
                    "dup": dp * tp,
                    "tp": tp,
                    "device_group": device_groups[stage_id],
                    "type": "variable"
                },
                "mlp": {
                    "dense_h_to_4h": {
                        "split": {"1": tp},
                        "dup": dp,
                        "device_group": device_groups[stage_id],
                        "type": "variable",
                        "lora": {
                            "lora_A": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            },
                            "lora_B": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            }
                        }
                    },
                    "dense_4h_to_h": {
                        "split": {"0": tp},
                        "dup": dp,
                        "device_group": device_groups[stage_id],
                        "type": "variable",
                        "lora": {
                            "lora_A": {
                                "split": {"0": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            },
                            "lora_B": {
                                "split": {"1": tp},
                                "dup": dp,
                                "device_group": device_groups[stage_id],
                                "type": "variable"
                            }
                        }
                    }
                }
            }
        ds_parallel_configs.append(ds_parallel_config)
    
    sort_ds_parallel_configs = [None] * len(dps)
    for i, index in enumerate(ds_config_indexs):
        sort_ds_parallel_configs[index] = ds_parallel_configs[i]

    return sort_ds_parallel_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_size', type=str, default='7b', help='size of gpt, 7b or 13b.'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=8, help='num of gpus.'
    )
    parser.add_argument(
        '--dps', type=str, default='', help='dp.'
    )
    parser.add_argument(
        '--tps', type=str, default='', help='tp.'
    )
    parser.add_argument(
        "--num_layers", type=int, default=32, help='num of Transformer layers'
    )
    parser.add_argument(
        '--pps', type=str, default='', help='pp.'
    )
    parser.add_argument(
        '--sps', type=str, default='', help='sp.'
    )
    parser.add_argument(
        '--save_folder', type=str, default=''
    )
    # parser.add_argument(
    #     '--zero', action='store_true', help='use zero or not.'
    # )
    args = parser.parse_args()
    num_layers = args.num_layers
    dps = args.dps.split(",")
    dps = [int(dp) for dp in dps]
    tps = args.tps.split(",")
    tps = [int(tp) for tp in tps]
    pps = args.pps.split(",")
    pps = [int(pp) for pp in pps]
    sps = args.sps.split(",")
    sps = [bool(int(sp)) for sp in sps]
    num_gpus = 0
    for (dp, tp, pp) in zip(dps, tps, pps):
        num_gpus += dp * tp * pp
    ds_parallel_configs = generate_ds_parallel_config(num_layers, num_gpus=num_gpus, dps=dps, tps=tps, pps=pps, sps=sps)
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i, ds_parallel_config in enumerate(ds_parallel_configs):
        file_name = f'dp{dps[i]}_tp{tps[i]}_pp{pps[i]}_{int(sps[i])}_lora_{i}.json'
        with open(f'{save_folder}/{file_name}', 'w') as f:
            json.dump(ds_parallel_config, f, indent=4)

