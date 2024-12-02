NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
TRAIN_TASK_NUM=${4:-1}
SP=${5:-1}

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

export LORA_SPLIT_METHOD=OURS

TRAINER_CONFIG_PATH=trainer_config/example.json
MEMORY_PROFILE_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv
SAVE_PATH=exp_result/throughput/throughput_per_gpu_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv
RAW_PATH=$SAVE_PATH.raw
SEQ_LEN_RANGE=(256 512 1024 2048 4096 8192 16384)
NUM_MICRO_BATCHES=64

# 从MEMORY_PROFILE_PATH读取，每一行形如tp,pp,sp,max_tokens
for line in `tail -n +2 ${MEMORY_PROFILE_PATH}`
do
    DP=1
    TP=`echo $line | awk -F ',' '{print $1}'`
    PP=`echo $line | awk -F ',' '{print $2}'`
    SP=`echo $line | awk -F ',' '{print $3}'`
    if [ $TP -ne 2 ]; then
        continue
    fi
    if [ $PP -ne 2 ]; then
        continue
    fi
    MAX_TOKENS=`echo $line | awk -F ',' '{print $4}'`
    MAX_TOKENS=${MAX_TOKENS%$'\r'}
    echo "dp: $DP, tp: $TP, pp: $PP, sp: $SP, max_tokens: $MAX_TOKENS"
    for SEQ_LEN in ${SEQ_LEN_RANGE[@]}
    do
        if [ $SEQ_LEN -gt $MAX_TOKENS ]; then
            break
        fi
        MICRO_BATCH_SIZE=$(expr $MAX_TOKENS / $SEQ_LEN)
        bash scripts/run_benchmark.sh \
        $NUM_LAYERS $HIDDEN_SIZE $NUM_HEADS $TRAIN_TASK_NUM \
        $SEQ_LEN $MICRO_BATCH_SIZE $NUM_MICRO_BATCHES \
        $DP $TP $PP $SP \
        $RAW_PATH $TRAINER_CONFIG_PATH throughput_experiment
    done
done

# filter required keys
python3 utils/csv_filter.py \
    --input_path $RAW_PATH \
    --output_path $SAVE_PATH \
    --filter_column dp tp pp sp mbs seq_len throughput_per_gpu
