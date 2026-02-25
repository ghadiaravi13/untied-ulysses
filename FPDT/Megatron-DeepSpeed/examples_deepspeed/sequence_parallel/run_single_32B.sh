#!/bin/bash
dir=$(pwd)

###############################################################################
#                         USER-CONFIGURABLE KNOBS                             #
###############################################################################

# Sequence length (can be overridden via first positional argument)
# Note: must be divisible by sp_size * sp_size * num_chunks
seq_len=131072
if [ -n "$1" ]; then
    seq_len="$1"
fi

# Training duration
STEPS=${STEPS:-10}
if [ ${seq_len} -ge 1048576 ]; then
    STEPS=3  # Reduce steps for very long sequences
fi

# Logging and evaluation
log_interval=1
eval_interval=100
eval_iters=0
save_interval=100

# Batch sizes
global_batch_size=1
batch_size=1

# Random seed
seed=1234

# Activation checkpointing (saves memory, reduces speed)
activation_checkpoint="true"

# Log optimizer states to tensorboard (uses extra GPU memory)
log_optimizer_state="false"

# Output directory
output_home="outputs/output_32B_FA3"

###############################################################################
#                           MODEL CONFIGURATION                               #
###############################################################################
# Qwen3 32B architecture
model_size=32.0
num_layers=64
hidden_size=5120
ffn_hidden_size=25600
num_attn_heads=64
num_key_value_heads=8
kv_channels=128

# Learning rate
lr=1.2e-4
min_lr=1.0e-6
init_std=0.009

###############################################################################
#                         PARALLELISM CONFIGURATION                           #
###############################################################################

# GPU configuration
num_gpus=8
num_gpus_pernode=8
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))

# Sequence parallelism (FPDT)
sp_size=8

# Model parallelism (must be 1 when SP > 1)
mp_size=1

# Pipeline parallelism (disabled for FPDT)
pp_size=1
no_pp="true"

# ZeRO stage (0=disabled, 3=full sharding)
zero_stage=3

# Data parallel size (computed)
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} / ${sp_size} ))

###############################################################################
#                      TRAINING DURATION (DERIVED)                            #
###############################################################################

train_tokens=$((${seq_len} * 10))
train_tokens_in_million=$((${train_tokens} / 1000000))
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))
exit_duration=30000000

###############################################################################
#                       LEARNING RATE SCHEDULE (DERIVED)                      #
###############################################################################

lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
lr_decay_tokens_in_million=${train_tokens_in_million}
lr_decay_tokens=$((${lr_decay_tokens_in_million} * 1000000))
lr_decay_style="cosine"

###############################################################################
#                          DATA AND TOKENIZER                                 #
###############################################################################

data_path="data/baichuan_mmap"
num_workers=0

# Download data if not present
if [ ! -f "data/baichuan_mmap.bin" ]; then
    wget -P data/ https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.bin
fi
if [ ! -f "data/baichuan_mmap.idx" ]; then
    wget -P data/ https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.idx
fi

# Download vocab files if not present
# vocab_path="gpt2-vocab.json"
# merge_path="gpt2-merges.txt"
# if [ ! -f "$vocab_path" ]; then
#     wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
# fi
# if [ ! -f "$merge_path" ]; then
#     wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
# fi

# Tokenizer (HuggingFace model ID or local path)
TOKENIZER_PATH=meta-llama/Meta-Llama-3-8B-Instruct

###############################################################################
#                            OUTPUT PATHS                                     #
###############################################################################

current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"

# Build job name for logging
prescale_grad="true"
if [[ $zero_stage -gt 0 ]]; then
    prescale_grad="false"
fi

jobname="seqlen_${seq_len}_llama_${model_size}B_tok${train_tokens_in_million}M"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_million}M_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}_z${zero_stage}_sp${sp_size}"
jobname="${jobname}_seed${seed}_rebase"

log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
tensorboard_path="${output_home}/tensorboard/${jobname}_${host}_${current_time}"

mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}

###############################################################################
#                           MEGATRON OPTIONS                                  #
###############################################################################

data_options=" \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --data-path ${data_path} \
    --data-impl mmap"

megatron_options=" \
    --disable-mem-efficient-ln \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --ds-sequence-parallel-fpdt \
    --ds-sequence-parallel-fpdt-chunk-size 32768 \
    --ds-sequence-parallel-fpdt-offloading \
    --ds-sequence-parallel-size ${sp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --num-key-value-heads ${num_key_value_heads} \
    --kv-channels ${kv_channels} \
    --ffn-hidden-size ${ffn_hidden_size} \
    --swiglu \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-iters ${STEPS} \
    --train-tokens ${train_tokens} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 50,25,25 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --weight-decay 0.1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --seed ${seed} \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v3 \
    --tensorboard-queue-size 1 \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --rotary-position-embeddings-theta 100000000 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --torch-profiler-enable \
    --tensorboard-dir ${tensorboard_path}"

# --train-samples ${train_samples} \

if [ "${activation_checkpoint}" = "true" ]; then
    megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
    megatron_options="${megatron_options} \
        --log-optimizer-states-to-tensorboard"
fi

###############################################################################
#                           DEEPSPEED OPTIONS                                 #
###############################################################################

config_json="ds_config_templates/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="ds_config_templates/ds_config_llama3_8B_TEMPLATE.json"

sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    > ${config_json}

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
    deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
    deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing \
        --checkpoint-in-cpu"
fi

###############################################################################
#                         CHECKPOINT RESUMPTION                               #
###############################################################################

# Handle checkpoint resumption for multi-node setups
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0

for (( node = 0; node <= num_node-1; node++ )); do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} : ${iteration} ))
    fi
done

if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

###############################################################################
#                              LAUNCH TRAINING                                #
###############################################################################

deepspeed ${dir}/../../pretrain_gpt.py \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options} \
    2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log
