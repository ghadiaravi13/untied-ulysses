#!/bin/bash
dir=$(pwd)

###############################################################################
#              MULTI-NODE LAUNCH (no passwordless SSH required)               #
###############################################################################
#
# Usage: run this script on EACH node independently, setting the environment
# variables below.  MASTER_ADDR and NODE_RANK are required; everything else
# has sensible defaults for a 4×8-GPU H100 setup.
#
# No passwordless SSH between nodes is needed.  DeepSpeed uses a hostfile
# (auto-generated) only for topology info; --no_ssh ensures no SSH is
# attempted.  NCCL handles cross-node communication directly.
#
#   On the master node (my-master-node):
#     MASTER_ADDR=my-master-node NODE_RANK=0 bash run_single_multinode_70B.sh
#
#   On the worker nodes (my-worker-node-N):
#     MASTER_ADDR=my-master-node NODE_RANK=N bash run_single_multinode_70B.sh
#
# Optionally override any variable via the environment, e.g.:
#     MASTER_PORT=29501 NUM_NODES=4 NUM_GPUS_PER_NODE=8 SP_SIZE=32 ...
#     NODE_ADDRESSES="host1,host2,host3,host4"  (comma-separated list of all nodes)
#
###############################################################################

# ---------- multi-node coordination ----------
MASTER_ADDR=${MASTER_ADDR:?"ERROR: MASTER_ADDR must be set (hostname or IP of rank-0 node)"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:?"ERROR: NODE_RANK must be set (0 for master, 1..N-1 for workers)"}
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

# All node hostnames, comma-separated.  Used to generate the hostfile that
# DeepSpeed requires for topology discovery.  No SSH is actually performed.
NODE_ADDRESSES=${NODE_ADDRESSES:?"ERROR: NODE_ADDRESSES must be set (comma-separated list of all nodes)"}

# ---------- generate a temporary hostfile ----------
# DeepSpeed's launcher requires a hostfile even with --no_ssh; it uses it
# purely to build the world-info topology (slots per host).  The --no_ssh
# and --no_ssh_check flags ensure no SSH connections are attempted.
HOSTFILE=$(mktemp /tmp/deepspeed_hostfile.XXXXXX)
IFS=',' read -ra _NODES <<< "${NODE_ADDRESSES}"
for _node in "${_NODES[@]}"; do
    echo "${_node} slots=${NUM_GPUS_PER_NODE}" >> "${HOSTFILE}"
done
echo "Generated hostfile at ${HOSTFILE}:"
cat "${HOSTFILE}"
trap "rm -f ${HOSTFILE}" EXIT

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
STEPS=${STEPS:-3}
if [ ${seq_len} -ge 1048576 ]; then
    STEPS=3  # Reduce steps for very long sequences
fi

# Disable pinned CPU memory for FPDT offloading at very long sequences
# to avoid host OOM from pinned (non-pageable) memory exhaustion.
fpdt_pin_memory_flag=""
if [ ${seq_len} -gt 4194304 ]; then
    fpdt_pin_memory_flag="--no-ds-sequence-parallel-fpdt-pin-memory"
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
output_home="outputs/output_70B_FA3_multinode"

###############################################################################
#                           MODEL CONFIGURATION                               #
###############################################################################
# Llama3 70B architecture
model_size=70.0
num_layers=80
hidden_size=8192
ffn_hidden_size=28672
num_attn_heads=64
num_key_value_heads=32 # need to keep it atleast as much as num_nodes
kv_channels=128

# Learning rate
lr=1.2e-4
min_lr=1.0e-6
init_std=0.009

###############################################################################
#                         PARALLELISM CONFIGURATION                           #
###############################################################################

# GPU configuration (derived from multi-node settings)
num_gpus=$(( ${NUM_NODES} * ${NUM_GPUS_PER_NODE} ))
num_gpus_pernode=${NUM_GPUS_PER_NODE}
num_node=${NUM_NODES}

# Sequence parallelism (FPDT) — all GPUs for maximum sequence length
sp_size=${SP_SIZE:-32}

# Model parallelism (must be 1 when SP > 1)
mp_size=1

# Pipeline parallelism (disabled for FPDT)
pp_size=1
no_pp="true"

# ZeRO stage (0=disabled, 3=full sharding)
zero_stage=3

# Data parallel size (computed)
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} / ${sp_size} ))

echo "================================================================"
echo " Multi-node FPDT launch"
echo "   MASTER_ADDR      = ${MASTER_ADDR}"
echo "   MASTER_PORT      = ${MASTER_PORT}"
echo "   NODE_RANK        = ${NODE_RANK}"
echo "   NUM_NODES        = ${NUM_NODES}"
echo "   NUM_GPUS_PER_NODE= ${NUM_GPUS_PER_NODE}"
echo "   Total GPUs       = ${num_gpus}"
echo "   SP size           = ${sp_size}"
echo "   DP size           = ${dp_size}"
echo "   seq_len           = ${seq_len}"
echo "================================================================"

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
    --ds-sequence-parallel-fpdt-chunk-size 65536 \
    --ds-sequence-parallel-fpdt-offloading \
    ${fpdt_pin_memory_flag} \
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

# Local-only checkpoint detection (no SSH required).
# Each node checks its own filesystem for the latest checkpoint.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0

if [ -f "$iteration_file" ]; then
    iteration=$(cat "$iteration_file")
fi

if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    echo "$iteration" > "$iteration_file"
    echo "$iteration_2" > "$iteration_file_2"
    echo "Resuming from checkpoint iteration ${iteration}"
fi

###############################################################################
#                              LAUNCH TRAINING                                #
###############################################################################

# DeepSpeed multi-node launch WITHOUT passwordless SSH.
#
# How it works:
#   --hostfile    : provides topology (which hosts, how many slots each).
#   --no_ssh      : tells DS not to SSH into remote hosts; each node only
#                   spawns its own local workers using --node_rank.
#   --no_ssh_check: skips the SSH-reachability pre-flight test.
#
# You run this script independently on every node.  NCCL coordinates
# across nodes via MASTER_ADDR:MASTER_PORT.
deepspeed \
    --hostfile ${HOSTFILE} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK} \
    --no_ssh \
    --no_ssh_check \
    ${dir}/../../pretrain_gpt.py \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options} \
    2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log
