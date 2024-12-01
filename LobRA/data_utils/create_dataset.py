import json
import io
import os
from data_utils.utils import jload, jdump
from data_utils.gpt_load_dataset import _load_dataset_to_json
from data_utils.prompt_template import AlpacaInstructTemplate, StackExchangedPairedTemplate, SummarizeTemplate

root_folder = "data"
dataset_name = "commitpackft"

if os.path.exists(f"{root_folder}/{dataset_name}/{dataset_name}.json"):
    json_file = f"{root_folder}/{dataset_name}/{dataset_name}.json"
else:
    json_file = _load_dataset_to_json(dataset_name, root_folder=root_folder)

datas = []
try:
    jdict = jload(json_file)
except BaseException:
    with open(json_file, 'r') as f:
        lines = f.readlines()
    jdict = [json.loads(line.strip()) for line in lines]    
print(jdict[0])

# # for pubmedqa
# def form_question(obj):
#     st = ""    
#     for i, label in enumerate(obj['context']['labels']):
#         st += f"{label}: {obj['context']['contexts'][i]}\n"
#     st += f"QUESTION: {obj['question']}\n"
#     st += f" ### ANSWER (yes|no|maybe): "
#     return st

# def form_translation(obj):
#     st = "You are an expert translator with fluency in English and Polish languages. Translate the given text from English to Polish.\n\n"
#     st += f"Text: {obj['en']}\n"
#     st += f" ### Translation: "
#     return st

for example in jdict:
    text = AlpacaInstructTemplate.format(example)
    example['text'] = text + example['output']
    datas.append(example)

jdump(datas, f"{root_folder}/{dataset_name}/{dataset_name}.json")

new_jdict = jload(f"{root_folder}/{dataset_name}/{dataset_name}.json")
print(new_jdict[0])