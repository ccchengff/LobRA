mkdir -p data/dolly
wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/main/databricks-dolly-15k.jsonl -O data/dolly/databricks-dolly-15k.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/merges.txt
python3 data_utils/gpt_load_dataset.py \
    --dataset dolly \
    --key text \
    --root_folder data \