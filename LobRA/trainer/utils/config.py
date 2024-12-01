from peft.lora.config import LoraConfig
from typing import List, Dict
from queue import Queue
import json
import argparse

class TaskConfig():
    lora_config: LoraConfig = None
    global_batch_size: int = 64
    dataset_name: str = ""
    context_length: int = 0
    json_key: str = ""
    steps: int = 10
    epochs: int = 1
    
    __params_map: Dict[str, str] = {
        "global_batch_size": "global_batch_size",
        "dataset_name": "dataset_name",
        "context_length": "context_length",
        "json_key": "json_key",
        "steps": "steps",
        "epochs": "epochs"
    }
    
    def __init__(self, config):
        for key in self.__params_map:
            setattr(self, key, config[self.__params_map[key]])
        self.lora_config = LoraConfig(rank=config['rank'], lora_alpha=config['lora_alpha'], target_modules=config['target_modules'])

class TrainerConfig():
    config_path: str = ""
    variant: str = "canonical"
    train_task_num: int = 0
    task_configs: List[TaskConfig] = []
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        print(f"Reading trainer config from {config_path}...")
        trainer_config = json.load(open(config_path, 'r'))
        self.variant = trainer_config['variant']
        self.train_task_num = trainer_config['train_task_num']
        print(f"Detected {self.train_task_num} fine-tuning tasks")
        task_config_queue = Queue()
        for value in trainer_config['task']:
            if type(value) == dict:
                task_config_queue.put(value)
        while (not task_config_queue.empty()):
            task_config = task_config_queue.get()
            for target_module in task_config['target_modules']:
                if self.variant == 'fused':
                    assert target_module in ['qkv_proj', 'o_proj', 'dense_h_to_4h', 'dense_4h_to_h']
                else:
                    assert target_module in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense_h_to_4h', 'dense_4h_to_h']
            self.task_configs.append(TaskConfig(task_config))
    
    def split_and_dump_task_configs(self, split_num: int, save_dir: str=None):
        save_dir = self.config_path[:self.config_path.rfind('/') + 1]
        trainer_config = json.load(open(self.config_path, 'r'))
        task_configs = trainer_config['task']
        task_num = len(task_configs)
        assert task_num % split_num == 0
        split_task_num = task_num // split_num
        for i in range(split_num):
            split_task_configs = task_configs[i * split_task_num: (i + 1) * split_task_num]
            split_config = {
                "variant": self.variant,
                "train_task_num": split_task_num,
                "task": [task_config for task_config in split_task_configs]
            }
            split_config_path = save_dir + self.config_path[self.config_path.rfind('/') + 1: -5] + f"_{i}.json"
            with open(split_config_path, 'w') as f:
                json.dump(split_config, f, indent=4)
            print(f"Split task configs {i} saved to {split_config_path}")
    
    def get_global_batch_size_list(self):
        return [task.global_batch_size for task in self.task_configs]

if __name__ == "__main__":
    # 从args中读取配置文件路径和分割数  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/trainer_config.json")
    parser.add_argument("--split_num", type=int, default=1)
    args = parser.parse_args()
    trainer_config = TrainerConfig(args.config_path)
    trainer_config.split_and_dump_task_configs(args.split_num)
