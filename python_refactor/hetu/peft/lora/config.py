from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union, Dict
from queue import Queue
import json

@dataclass
class LoraConfig():
    rank: int = field(default=8, metadata={'help': 'Lora attention dimension'})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you shoud specify the target modules manually."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA doi.org/10.48550/arXiv.2312.03732"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )

class TaskConfig():
    lora_config: LoraConfig = None
    global_batch_size: int = 64
    json_file: str = ""
    json_key: str = ""
    steps: int = 10
    
    __params_map: Dict[str, str] = {
        "global_batch_size": "global_batch_size",
        "json_file": "json_file",
        "json_key": "json_key",
        "steps": "steps"
    }
    
    def __init__(self, config):
        for key in self.__params_map:
            setattr(self, key, config[self.__params_map[key]])
        self.lora_config = LoraConfig(rank=config['rank'], lora_alpha=config['lora_alpha'], target_modules=config['target_modules'])

class TrainerConfig():
    train_task_num: int = 0
    task_configs: List[TaskConfig] = []
    
    def __init__(self, config_path: str):
        trainer_config = json.load(open(config_path, 'r'))
        self.train_task_num = trainer_config['train_task_num']
        task_config_queue = Queue()
        for value in trainer_config['task']:
            if type(value) == dict:
                task_config_queue.put(value)
        while (not task_config_queue.empty()):
            task_config = task_config_queue.get()
            self.task_configs.append(TaskConfig(task_config))
    