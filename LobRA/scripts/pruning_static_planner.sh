MODEL_SIZE=${1:-'7B'}
NUM_GPUS=${2:-16}
TRAIN_TASK_NUM=${3:-6}
BUCKET_NUM=${4:-16}
MAX_SEQ_LENGTH=${5:-16384}
CONFIG_PATH=${6:-'exp_task6'}
MODEL_TYPE=${7:-llama}

if [ "${MODEL_SIZE}" = "7B" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
	FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "32B" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656
	FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70B" ]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192
	FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/32b/70b for test.
    exit 0
fi

export HETU_DATA_DISPATCH=TEST
export CUSTOM_DISTRIBUTION=FALSE
if [ -z ${BUCKET_PLAN} ]; then
    export BUCKET_PLAN=DYNAMIC
fi

ROOT_FOLDER=data
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
TRAINER_CONFIG_PATH=trainer_config/${CONFIG_PATH}.json
PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}.csv
MEMORY_PROFILE_PATH=exp_result/profile/memory/max_tokens_${MODEL_TYPE}_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv

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
--sp 1 \
--model_type $MODEL_TYPE \
--max_seq_length $MAX_SEQ_LENGTH \
--min_seq_length 256 \
--num_gpus $NUM_GPUS \
--bucket_num $BUCKET_NUM
