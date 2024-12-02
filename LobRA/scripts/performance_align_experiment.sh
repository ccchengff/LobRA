NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
TRAIN_TASK_NUM=${4:-1}
NUM_GPUS_LIST=${5:-8}

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

export LORA_SPLIT_METHOD=SPLIT_B2

TRAINER_CONFIG_PATH=trainer_config/example_fused.json
MEMORY_PROFILE_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv
SAVE_PATH=exp_result/performance_align/lora_split_${LORA_SPLIT_METHOD}/performance_align_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv
RAW_PATH=$SAVE_PATH.raw
GLOBAL_BATCH_SIZE=64

IFS=',' read -r -a NUM_GPUS_LIST <<< $NUM_GPUS_LIST

# for line in `tail -n +2 ${MEMORY_PROFILE_PATH}`
# do
#     TP=`echo $line | awk -F ',' '{print $1}'`
#     PP=`echo $line | awk -F ',' '{print $2}'`
#     SP=`echo $line | awk -F ',' '{print $3}'`
#     MAX_TOKENS=`echo $line | awk -F ',' '{print $4}'`
#     MAX_TOKENS=${MAX_TOKENS%$'\r'}
#     SP=1
#     for NUM_GPUS in ${NUM_GPUS_LIST[@]}
#     do
#         if [ $(($TP * $PP)) -gt $NUM_GPUS ]; then
#             continue
#         fi
#         DP=$(($NUM_GPUS / ($TP * $PP)))
#         for SEQ_LEN in 2048 4096 8192 16384; do
#             if [ $SEQ_LEN -gt $MAX_TOKENS ]; then
#                 break
#             fi
#             echo "dp: $DP, tp: $TP, pp: $PP, sp: $SP, seq_len: $SEQ_LEN"
#             MICRO_BATCH_SIZE=1
#             NUM_MICRO_BATCHES=$(($GLOBAL_BATCH_SIZE / ($DP * $MICRO_BATCH_SIZE)))
#             bash scripts/run_benchmark.sh \
#             $NUM_LAYERS $HIDDEN_SIZE $NUM_HEADS $TRAIN_TASK_NUM \
#             $SEQ_LEN $MICRO_BATCH_SIZE $NUM_MICRO_BATCHES \
#             $DP $TP $PP $SP \
#             $RAW_PATH $TRAINER_CONFIG_PATH performance_align
#         done
#     done
# done

for TP in 1 2 4 8; do
# for TP in 8; do
    # for PP in 1; do
    for PP in 1 2 4 8; do
        SP=1
        for NUM_GPUS in ${NUM_GPUS_LIST[@]}
        do
            if [ $(($TP * $PP)) -gt $NUM_GPUS ]; then
                continue
            fi
            DP=$(($NUM_GPUS / ($TP * $PP)))
            for SEQ_LEN in 2048 4096 8192 16384; do
            # for SEQ_LEN in 8192; do
                echo "dp: $DP, tp: $TP, pp: $PP, sp: $SP, seq_len: $SEQ_LEN"
                MICRO_BATCH_SIZE=1
                NUM_MICRO_BATCHES=$(($GLOBAL_BATCH_SIZE / ($DP * $MICRO_BATCH_SIZE)))
                bash scripts/run_benchmark.sh \
                $NUM_LAYERS $HIDDEN_SIZE $NUM_HEADS $TRAIN_TASK_NUM \
                $SEQ_LEN $MICRO_BATCH_SIZE $NUM_MICRO_BATCHES \
                $DP $TP $PP $SP \
                $RAW_PATH $TRAINER_CONFIG_PATH performance_align
            done
        done
    done
done

# filter required keys
python3 utils/csv_filter.py \
    --input_path $RAW_PATH \
    --output_path $SAVE_PATH \
    --filter_column dp tp pp sp mbs seq_len num_micro_batches e2e_time