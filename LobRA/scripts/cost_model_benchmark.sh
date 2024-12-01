# NCCL_DEBUG=info
MODEL_TYPE=${1:-llama}
NUM_LAYERS=${2:-32}
HIDDEN_SIZE=${3:-4096}
NUM_HEADS=${4:-32}
TRAIN_TASK_NUM=${5:-1}
TPS=${6:-2}
SP=${7:-1}

export NCCL_DEBUG=VERSION
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export EVENT_TIMING=TRUE

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

export LORA_SPLIT_METHOD=SPLIT_B2

FFN_HIDDEN_SIZE=$(($HIDDEN_SIZE * 4))
if [ $HIDDEN_SIZE -eq 4096 ] && [ $NUM_HEADS -eq 32 ]; then
    FFN_HIDDEN_SIZE=11008
    MODEL_SIZE=7B
elif [ $HIDDEN_SIZE -eq 5120 ] && [ $NUM_HEADS -eq 40 ]; then
    FFN_HIDDEN_SIZE=13824
    MODEL_SIZE=13B
elif [ $HIDDEN_SIZE -eq 8192 ] && [ $NUM_HEADS -eq 64 ]; then
    FFN_HIDDEN_SIZE=28672
    MODEL_SIZE=70B
else
    MODEL_SIZE=UNKNOWN
fi

TRAINER_CONFIG_PATH=trainer_config/exp_task${TRAIN_TASK_NUM}.json
PROFILE_MEMORY_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
PROFILE_PATH=exp_result/profile/cost_model/profile_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
# PROFILE_FW_PATH=exp_result/profile/cost_model/profile_fw_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
# PROFILE_BW_PATH=exp_result/profile/cost_model/profile_bw_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
VALIDATION_PATH=exp_result/profile/cost_model/validation_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv

IFS=',' read -r -a tps <<< "$TPS"

SAVE_FOLDER=generated_ds_configs

for i in $(seq 0 $(( ${#tps[@]} - 1 ))); do
    TP=${tps[$i]}
    NUM_GPUS=${tps[$i]}
    # for SEQ_LEN in 256 512 1024 2048 4096 8192 16384; do
    for SEQ_LEN in 8192; do
        PROFILE_STEPS=15
        if [ $TP -eq 16 ]; then
            mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
            -H job-83e1033f-9636-44b3-bf8b-2b627707b95f-master-0:8,job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0:8 \
            -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x EVENT_TIMING \
            -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_DEBUG -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x NCCL_IB_GID_INDEX=3 \
            -x LORA_SPLIT_METHOD \
            --output-filename logs/cost_model_${MODEL_TYPE}_${MODEL_SIZE}/ds_parallel_${NUM_GPUS}_tp${TP}_sp${SP} --merge-stderr-to-stdout \
            python3 scripts/cost_model_benchmark.py \
            --trainer_config_path $TRAINER_CONFIG_PATH \
            --profile_path $PROFILE_PATH \
            --profile_memory_path $PROFILE_MEMORY_PATH \
            --validation_path $VALIDATION_PATH \
            --vocab_size 30592 \
            --hidden_size $HIDDEN_SIZE \
            --ffn_hidden_size $FFN_HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --tp $TP \
            --sp $SP \
            --train_task_num $TRAIN_TASK_NUM \
            --num_layers $NUM_LAYERS \
            --model_type $MODEL_TYPE \
            --lr 1e-4 \
            --seq_len_range $SEQ_LEN \
            --profile_steps $PROFILE_STEPS \
            --warmup_steps 5 \
            --dropout_prob 0 \
            --bf16 \
            --use_flash_attn \
            --use_two_node
        else
            mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
            --output-filename logs/cost_model_${MODEL_TYPE}_${MODEL_SIZE}/ds_parallel_${NUM_GPUS}_tp${TP}_sp${SP} --merge-stderr-to-stdout \
            python3 scripts/cost_model_benchmark.py \
            --trainer_config_path $TRAINER_CONFIG_PATH \
            --profile_path $PROFILE_PATH \
            --profile_memory_path $PROFILE_MEMORY_PATH \
            --validation_path $VALIDATION_PATH \
            --vocab_size 30592 \
            --hidden_size $HIDDEN_SIZE \
            --ffn_hidden_size $FFN_HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --tp $TP \
            --sp $SP \
            --train_task_num $TRAIN_TASK_NUM \
            --num_layers $NUM_LAYERS \
            --model_type $MODEL_TYPE \
            --lr 1e-4 \
            --seq_len_range $SEQ_LEN \
            --profile_steps $PROFILE_STEPS \
            --warmup_steps 5 \
            --dropout_prob 0 \
            --bf16 \
            --use_flash_attn
        fi
    done
done