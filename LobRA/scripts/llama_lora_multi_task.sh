MODEL_SIZE=${1:-'7B'}
TRAIN_TASK_NUM=${2:-6}
MAX_TOKENS_LIST=${3:-16384}
DPS=${4:-2}
TPS=${5:-8}
PPS=${6:-1}
BUCKET_NUM=${7:-16}
MAX_SEQ_LENGTH=${8:-8192}
CONFIG_PATH=${9:-}
MIN_SEQ_LENGTH=${10:-256}
SP=${11:-1}
PROFILE_PATH=${12:-}
VOCAB_FILE=${13:-}
MERGE_FILE=${14:-}

# env
PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

export NCCL_DEBUG=VERSION
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export EVENT_TIMING=FALSE
export BUCKET_GRAD_BUFFER=LAYER
export GRAD_CONCAT_BUFFER=ON
export SPLIT_COLLECTIVE_STREAM=OFF

if [ -z $DP_BUCKET ]; then
    export DP_BUCKET=TRUE
fi

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

if [ -z $HETU_DATA_DISPATCH ]; then
    export HETU_DATA_DISPATCH=BALANCE
fi
export LORA_SPLIT_METHOD=SPLIT_B2

if [ $DP_BUCKET = "TRUE" ] || [ $BUCKET_NUM = 16 ]; then
    export HETU_MAX_SPLIT_SIZE_MB=10240
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
else 
    export HETU_MAX_SPLIT_SIZE_MB=200
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
fi

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

export HETU_MEMORY_PROFILE=WARN
# export HETU_MEMORY_LOG_FILE="memory_logs/run_multi_task_llama/memory_profile_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks_dps${DPS}_tps${TPS}_pps${PPS}_sp${SP}_tokens${MAX_TOKENS_LIST}"
export HETU_MEMORY_LOG_FILE=""

export PROFILE_DYNAMIC_PLANNER=FALSE
export PROFILE_E2E_COST=TRUE
export CUSTOM_DISTRIBUTION=FALSE
export COST_MODEL_ESTIMATE=FALSE
export GET_TOKENS=FALSE

IFS=',' read -r -a dps <<< "$DPS"
IFS=',' read -r -a tps <<< "$TPS"
IFS=',' read -r -a pps <<< "$PPS"
IFS=',' read -r -a max_tokens_list <<< "$MAX_TOKENS_LIST"
NUM_GPUS=0

if false;then
new_dps=()
new_tps=()
new_pps=()
new_max_tokens_list=()

for i in "${!dps[@]}"; do
    dp="${dps[i]}"
    tp="${tps[i]}"
    pp="${pps[i]}"
    max_tokens="${max_tokens_list[i]}"

    for ((j=0; j<dp; j++)); do
        new_dps+=("1")
        new_tps+=("$tp")
        new_pps+=("$pp")
        new_max_tokens_list+=("$max_tokens")
    done
done

dps=("${new_dps[@]}")
tps=("${new_tps[@]}")
pps=("${new_pps[@]}")

IFS=','
MAX_TOKENS_LIST="${new_max_tokens_list[*]}"
DPS="${dps[*]}"
TPS="${tps[*]}"
PPS="${pps[*]}"
unset IFS
fi

SPS=""
for i in $(seq 0 $(( ${#dps[@]} - 1 ))); do
    if [ $i -eq 0 ]; then
        SPS=${SP}
    else
        SPS=${SPS},${SP}
    fi
done

if [ ${#dps[@]} -ne ${#tps[@]} ]; then
    echo "error: dps length is not equal to tps length"
    exit 1
fi

if [ ${#dps[@]} -ne ${#pps[@]} ]; then
    echo "error: dps length is not equal to pps length"
    exit 1
fi

for i in $(seq 0 $(( ${#dps[@]} - 1 ))); do
    NUM_GPUS=$((NUM_GPUS + dps[i] * tps[i] * pps[i]))
done

STRATEGY_NUM=${#dps[@]}
SAVE_FOLDER=generated_ds_configs

python3 utils/ds_parallel_config.py \
    --num_layers ${NUM_LAYERS} \
    --dps ${DPS} \
    --tps ${TPS} \
    --pps ${PPS} \
    --sps ${SPS} \
    --save_folder ${SAVE_FOLDER}

DS_PARALLEL_CONFIGS=""
for i in $(seq 0 $((STRATEGY_NUM-1))); do
    DS_PARALLEL_CONFIG=${SAVE_FOLDER}/dp${dps[i]}_tp${tps[i]}_pp${pps[i]}_${SP}_lora_${i}.json
    if [ $i -eq 0 ]; then
        DS_PARALLEL_CONFIGS=${DS_PARALLEL_CONFIG}
    else
        DS_PARALLEL_CONFIGS=${DS_PARALLEL_CONFIGS}@${DS_PARALLEL_CONFIG}
    fi
done

ROOT_FOLDER=data
if [ -z $VOCAB_FILE ]; then
    VOCAB_FILE=${ROOT_FOLDER}/vocab.json
fi
if [ -z $MERGE_FILE ]; then
    MERGE_FILE=${ROOT_FOLDER}/merges.txt
fi
if [ -z $PROFILE_PATH ]; then
    PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}_1tasks_sp${SP}.csv
fi
TRAINER_CONFIG_PATH=trainer_config/${CONFIG_PATH}.json

if [ $NUM_GPUS -eq 16 ]; then
    mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np 16 \
    -H job-83e1033f-9636-44b3-bf8b-2b627707b95f-master-0:8,job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0:8 \
    -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
    -x EVENT_TIMING -x BUCKET_GRAD_BUFFER -x GRAD_CONCAT_BUFFER -x SPLIT_COLLECTIVE_STREAM -x GET_TOKENS -x DP_BUCKET \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_DEBUG -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x NCCL_IB_GID_INDEX=3 \
    -x CUSTOM_DISTRIBUTION -x HETU_DATA_DISPATCH -x PROFILE_DYNAMIC_PLANNER -x HETU_MEMORY_PROFILE -x PROFILE_E2E_COST -x LORA_SPLIT_METHOD \
    -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
    --output-filename logs/run_multi_task_llama/ds_parallel_${NUM_GPUS}_dps${DPS}_tps${TPS}_pps${PPS}_sp${SP}_tokens${MAX_TOKENS_LIST} --merge-stderr-to-stdout \
    python3 scripts/llama_lora_multi_task.py \
    --ds_parallel_config $DS_PARALLEL_CONFIGS \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --profile_path $PROFILE_PATH \
    --max_tokens $MAX_TOKENS_LIST \
    --vocab_file $VOCAB_FILE \
    --merge_file $MERGE_FILE \
    --vocab_size 30592 \
    --hidden_size $HIDDEN_SIZE \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_attention_heads $NUM_HEADS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --min_seq_length $MIN_SEQ_LENGTH \
    --bucket_num $BUCKET_NUM \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --hidden_act relu \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn \
    --use_two_node
else
    export HETU_LOCAL_HOSTNAME=worker-0
    mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
    --output-filename logs/run_multi_task_llama/ds_parallel_${NUM_GPUS}_dps${DPS}_tps${TPS}_pps${PPS}_sp${SP}_tokens${MAX_TOKENS} --merge-stderr-to-stdout \
    python3 scripts/llama_lora_multi_task.py \
    --ds_parallel_config $DS_PARALLEL_CONFIGS \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --profile_path $PROFILE_PATH \
    --max_tokens $MAX_TOKENS_LIST \
    --vocab_file $VOCAB_FILE \
    --merge_file $MERGE_FILE \
    --vocab_size 30592 \
    --hidden_size $HIDDEN_SIZE \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_attention_heads $NUM_HEADS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --min_seq_length $MIN_SEQ_LENGTH \
    --bucket_num $BUCKET_NUM \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --hidden_act relu \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn
fi
