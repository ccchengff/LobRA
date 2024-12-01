# NCCL_DEBUG=info
MODEL_TYPE=${1:-llama}
NUM_LAYERS=${2:-32}
HIDDEN_SIZE=${3:-4096}
NUM_HEADS=${4:-32}
TRAIN_TASK_NUM=${5:-1}
NUM_GPUS=${6:-8}
SP=${7:-1}
MAX_SEQ_LENGTH=${8:-8192}
CONFIG_PATH=${9:-}
BUCKET_NUM=${10:-}

FFN_HIDDEN_SIZE=$(($HIDDEN_SIZE * 4))
if [ $NUM_LAYERS -eq 32 ] && [ $HIDDEN_SIZE -eq 4096 ] && [ $NUM_HEADS -eq 32 ]; then
    FFN_HIDDEN_SIZE=11008
    MODEL_SIZE=7B
elif [ $NUM_LAYERS -eq 40 ] && [ $HIDDEN_SIZE -eq 5120 ] && [ $NUM_HEADS -eq 40 ]; then
    FFN_HIDDEN_SIZE=13824
    MODEL_SIZE=13B
elif [ $NUM_LAYERS -eq 60 ] && [ $HIDDEN_SIZE -eq 6656 ] && [ $NUM_HEADS -eq 64 ]; then
    FFN_HIDDEN_SIZE=17920
    MODEL_SIZE=32B
elif [ $NUM_LAYERS -eq 80 ] && [ $HIDDEN_SIZE -eq 8192 ] && [ $NUM_HEADS -eq 64 ]; then
    FFN_HIDDEN_SIZE=28672
    MODEL_SIZE=70B
else
    MODEL_SIZE=UNKNOWN
fi

if [ -z ${HETU_DATA_DISPATCH} ]; then
    export HETU_DATA_DISPATCH=BALANCE
fi

export HETU_DATA_DISPATCH=TEST
export CUSTOM_DISTRIBUTION=FALSE
export BUCKET_PLAN=DYNAMIC

ROOT_FOLDER=data
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
TRAINER_CONFIG_PATH=trainer_config/${CONFIG_PATH}.json
PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}_1tasks_sp${SP}.csv
# PROFILE_FW_PATH=exp_result/profile/cost_model/profile_fw_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
# PROFILE_BW_PATH=exp_result/profile/cost_model/profile_bw_time_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv
MEMORY_PROFILE_PATH=exp_result/profile/memory/max_tokens_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_sp${SP}.csv

python3 test/try_static_planner.py \
--trainer_config_path $TRAINER_CONFIG_PATH \
--profile_path $PROFILE_PATH \
--max_tokens_path $MEMORY_PROFILE_PATH \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--sp $SP \
--model_type $MODEL_TYPE \
--max_seq_length $MAX_SEQ_LENGTH \
--min_seq_length 256 \
--num_gpus $NUM_GPUS \
--bucket_num $BUCKET_NUM
