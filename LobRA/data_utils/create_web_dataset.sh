# mkdir -p data/web
# wget https://huggingface.co/datasets/tiiuae/falcon-refinedweb/resolve/main/data/train-00000-of-05534-b8fc5348cbe605a5.parquet -O data/web/refinedweb0.parquet
# wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/vocab.json
# wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/merges.txt
python3 data_utils/gpt_load_dataset.py \
    --dataset web/refinedweb0 \
    --key content \
    --root_folder data \