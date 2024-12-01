# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
NUM_GPUS_LIMIT=${4:-1}
TRAIN_TASK_NUM=${5:-1}
SEQ_LEN_RANGE=${6:-1024}
SP=${7:-1}

FFN_HIDDEN_SIZE=$(($HIDDEN_SIZE * 4))
if [ $NUM_LAYERS -eq 32 ] && [ $HIDDEN_SIZE -eq 4096 ] && [ $NUM_HEADS -eq 32 ]; then
    FFN_HIDDEN_SIZE=11008
    MODEL_SIZE=7B
elif [ $NUM_LAYERS -eq 40 ] && [ $HIDDEN_SIZE -eq 5120 ] && [ $NUM_HEADS -eq 40 ]; then
    FFN_HIDDEN_SIZE=13824
    MODEL_SIZE=13B
elif [ $NUM_LAYERS -eq 80 ] && [ $HIDDEN_SIZE -eq 8192 ] && [ $NUM_HEADS -eq 64 ]; then
    FFN_HIDDEN_SIZE=28672
    MODEL_SIZE=70B
else
    MODEL_SIZE=UNKNOWN
fi

TRAINER_CONFIG_PATH=trainer_config/task10.json
SAVE_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv

python3 scripts/profile_memory.py \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --save_path $SAVE_PATH \
    --hidden_size $HIDDEN_SIZE \
    --num_attention_heads $NUM_HEADS \
    --train_task_num $TRAIN_TASK_NUM \
    --num_layers $NUM_LAYERS \
    --num_gpus_limit $NUM_GPUS_LIMIT \
    --train_task_num $TRAIN_TASK_NUM \
    --sp $SP \
    # --seq_len_range $SEQ_LEN_RANGE \
