# NCCL_DEBUG=info
MODEL_TYPE=${1:-llama}
NUM_LAYERS=${2:-16}
HIDDEN_SIZE=${3:-4096}
NUM_HEADS=${4:-32}
TRAIN_TASK_NUM=${5:-1}
NUM_GPUS=${6:-8}
SP=${7:-0}

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

ROOT_FOLDER=data
SAVE_FOLDER=generated_ds_configs
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
TRAINER_CONFIG_PATH=trainer_config/task10.json
PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
# PROFILE_FW_PATH=exp_result/profile/cost_model/profile_fw_time_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
# PROFILE_BW_PATH=exp_result/profile/cost_model/profile_bw_time_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
MEMORY_PROFILE_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
THROUGHPUT_PATH=exp_result/throughput/throughput_per_gpu_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv

export NCCL_DEBUG=VERSION
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRATEGY_FILTER=COST_MODEL
if [ -z ${HETU_DATA_DISPATCH} ]; then
    export HETU_DATA_DISPATCH=BALANCE
fi

python3 scripts/multi_task_workflow.py \
--trainer_config_path $TRAINER_CONFIG_PATH \
--profile_path $PROFILE_PATH \
--max_tokens_path $MEMORY_PROFILE_PATH \
--throughput_path $THROUGHPUT_PATH \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--sp $SP \
--model_type $MODEL_TYPE \
--num_gpus $NUM_GPUS \
--max_seq_length 16384 \
--min_seq_length 256 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn