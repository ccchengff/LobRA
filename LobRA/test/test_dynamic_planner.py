import random
from trainer.utils import TrainerConfig
from model import LLamaConfig
from profiler.cost_model import CostModel
from trainer.planner import NewDynamicBatchPlanner, GroupedDynamicBatchPlanner

trainer_config_path = "trainer_config/task4.json"
profile_path = "exp_result/profile/cost_model/profile_time_llama_7B_4tasks_sp1.csv"
max_tokens_path = "exp_result/profile/memory/max_tokens_llama_7B_4tasks_sp1.csv"
num_strategy = 3
gbs = 64
max_tokens_list = [2048, 4096, 8192]
dps = [10, 1, 1]
tps = [1, 2, 2]
pps = [1, 1, 2]
seq_lens = [256, 512, 1024, 2048, 4096, 8192]

def generate_seq_distribution_map(train_task_num, gbs):
    seq_len_distribution_map = {}
    for i in range(train_task_num):
        seq_len_distribution = {s: 0 for s in seq_lens}
        num = 0
        for s in seq_lens:
            seq_len_distribution[s] = random.randint(0, gbs - num)
            num += seq_len_distribution[s]
        print(f"seq_len_distribution = {seq_len_distribution}")
        for s in seq_lens:
            seq_len_distribution[s] /= gbs
        seq_len_distribution_map[i] = seq_len_distribution
    return seq_len_distribution_map

trainer_config = TrainerConfig(trainer_config_path)
global_batch_size_list = [gbs for _ in range(trainer_config.train_task_num)]
cost_model_config = LLamaConfig(
    ffn_hidden_size=11008,
    n_layer=32,
    n_embd=4096,
    n_head=32)
cost_model = CostModel(cost_model_config, "llama", trainer_config.train_task_num, \
                        trainer_config_path, profile_path, max_tokens_path=max_tokens_path, sp=1)
dynamic_planner = NewDynamicBatchPlanner(cost_model, 32, trainer_config.train_task_num, global_batch_size_list,
                                         num_strategy, max_tokens_list, dps, tps, pps)

# seq_len_distribution_map = generate_seq_distribution_map(trainer_config.train_task_num, gbs)
seq_len_distribution_map = {0: {256: 0.75, 512: 0.1875, 1024: 0.0625}, 1: {256: 0.09375, 512: 0.28125, 1024: 0.5, 2048: 0.125}, 2: {256: 0.1875, 512: 0.40625, 1024: 0.3125, 2048: 0.09375}, 3: {256: 0.09375, 512: 0.84375, 1024: 0.0625}}

print(f"generated: {seq_len_distribution_map}")
multi_task_batch_dispatch_map, _ = dynamic_planner.schedule(seq_len_distribution_map)

print("============", flush=True)
for task_id, seq_len_distribution_list in multi_task_batch_dispatch_map.items():
    seq_num_map = {s: 0 for s in seq_lens}
    for i, seq_len_distribution in enumerate(seq_len_distribution_list):
        for s in seq_lens:
            seq_num_map[s] += seq_len_distribution.get(s, 0)
    print(f"{task_id} : {seq_num_map}")