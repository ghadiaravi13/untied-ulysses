#!/usr/bin/bash
# Single experiment runner for multinode context-parallelism configurations.
# Run this script on EACH node with appropriate RDZV_ENDPOINT set.
#
# USAGE (2-node example):
#   # On node 1 (master):
#   RDZV_ENDPOINT="node1-hostname:29500" bash run_single_multinode.sh
#
#   # On node 2:
#   RDZV_ENDPOINT="node1-hostname:29500" bash run_single_multinode.sh
#
# The script auto-coordinates via torchrun's rendezvous mechanism.

set -x  # Print commands as they are executed (removed -e to continue on failures)

################################################################################
# Paths (relative to repo root)
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ROOT_TMP="${REPO_ROOT}/tmp"
HF_CACHE_DIR="${ROOT_TMP}/HF_cache"
TRAIN_ROOT="${REPO_ROOT}/torchtitan"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

################################################################################
# Multinode configuration
################################################################################
NNODES=${NNODES:-"2"}
NPROC_PER_NODE=${NPROC_PER_NODE:-"8"}
RDZV_ID=${RDZV_ID:-"101"}
RDZV_BACKEND=${RDZV_BACKEND:-"c10d"}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"master-hostname:29500"}  # Set this to master node!
NODE_RANK=${NODE_RANK:-""}  # Set by torchrun if not specified

################################################################################
# Training configuration
################################################################################
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b_test_2d.toml"}
DUMP_FOLDER=${DUMP_FOLDER:-"${ROOT_TMP}/run_single_multinode_32b"}
BASE_TRACE_FOLDER="run_single_multinode"
MEM_BASE_TRACE_FOLDER="run_single_multinode"

CONTEXT_LENGTH=${CONTEXT_LENGTH:-"131072"}
PROFILING_SUFFIX=""
COMPILE=${COMPILE:-"false"}
FLAVOR=${FLAVOR:-"32B"}

# AC/Offloading parameters
OFFLOADING=${OFFLOADING:-"no"} # ["no", "UAO", "TAO"] Unsloth/Torchtune offloading
CHECKPOINTING=${CHECKPOINTING:-"full"} # ["none", "async_selective", "sync_selective", "async_", "full"]
FSDP_OFFLOADING=${FSDP_OFFLOADING:-"false"} # ["true", "false"] FSDP CPU Offloading

WANDB_PROJECT=${WANDB_PROJECT:-"run_single_multinode_32b"}

STEPS=${STEPS:-"3"}
BATCH_SIZE=${BATCH_SIZE:-"1"}

################################################################################
# Environment configuration
################################################################################
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export REQUESTS_TIMEOUT=1200
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_CACHE="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIN_MEMORY=${PIN_MEMORY:-"True"}
export INP_PIN_MEMORY=${INP_PIN_MEMORY:-"True"}

# no clear cuda cache
export CLEAR_CUDA_CACHE=${CLEAR_CUDA_CACHE:-"0"}
export MEMORY_SNAPSHOT_MAX_ENTRIES=${MEMORY_SNAPSHOT_MAX_ENTRIES:-"100000000"}
export WARMUP=${WARMUP:-"0"}

# NCCL config hacks
# export NCCL_DEBUG="INFO"
export NCCL_ALGO="RING"
export NCCL_IGNORE_CPU_AFFINITY="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_AR_THRESHOLD="0"
export NCCL_IB_PCI_RELAXED_ORDERING="1"
export NCCL_IB_SPLIT_DATA_ON_QPS="0"
export NCCL_IB_QPS_PER_CONNECTION="2"

# Ensure cache directory exists
mkdir -p "${HF_CACHE_DIR}"

# Logging configuration
export LOG_RANK=${LOG_RANK:-0}

# TorchFT configuration
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

################################################################################
# Experiment grid
################################################################################
# Format: "ulysses_degree ring_degree"
# For 2 nodes × 8 GPUs = 16 total GPUs
EXPERIMENTS=(
    "8 2"
    "1 16"
)

ATTN_IMPL_OPTIONS=(
    "upipe_fa3_offload_tiled_mlp"   # UpipeAttention (Untied Ulysses)
    "usp_fa3_offload_tiled_mlp"     # LongContextAttention (USP zigzag)
    "torch_ring_alltoall"           # Standard PyTorch attention
)

RING_COMM_HEADS_OPTIONS=(
    # "mha_kv"
    "gqa_kv"
)

################################################################################
# Helpers
################################################################################
get_script_path() {
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "${BASH_SOURCE[0]}"
    elif command -v realpath >/dev/null 2>&1; then
        realpath "${BASH_SOURCE[0]}"
    else
        echo "${BASH_SOURCE[0]}"
    fi
}

trace_folder_name() {
    local ulysses_degree=$1
    local ring_degree=$2
    local ring_comm_heads=$3
    local attn_impl=$4

    echo "${BASE_TRACE_FOLDER}_$((CONTEXT_LENGTH / 1024))k_ACS_${AC_LAYER_STRIDE}_BS_${BATCH_SIZE}_${ulysses_degree}u${ring_degree}r_${PROFILING_SUFFIX}_${ring_comm_heads}_${attn_impl}_ao_${OFFLOADING}_ac_${CHECKPOINTING}_fsdpoffl_${FSDP_OFFLOADING}_compile_${COMPILE}"
}

mem_trace_folder_name() {
    local ulysses_degree=$1
    local ring_degree=$2
    local ring_comm_heads=$3
    local attn_impl=$4

    echo "${MEM_BASE_TRACE_FOLDER}_$((CONTEXT_LENGTH / 1024))k_ACS_${AC_LAYER_STRIDE}_BS_${BATCH_SIZE}_${ulysses_degree}u${ring_degree}r_${PROFILING_SUFFIX}_${ring_comm_heads}_${attn_impl}_ao_${OFFLOADING}_ac_${CHECKPOINTING}_fsdpoffl_${FSDP_OFFLOADING}_compile_${COMPILE}"
}

should_skip_config() {
    local config=$1
    local attn_impl=$2

    # Pure ring (1uNr) only allowed for usp or torch (have ring support)
    if [[ "${config}" == "1 "* ]] && [[ "${attn_impl}" != *"usp"* ]] && [[ "${attn_impl}" != *"torch"* ]]; then
        return 0
    fi

    # torch_ring_alltoall only valid for 1 ulysses + some ring (1u Nr)
    if [ "${attn_impl}" = "torch_ring_alltoall" ] && [[ "${config}" != "1 "* ]]; then
        return 0
    fi

    return 1
}

build_overrides() {
    local ulysses_degree=$1
    local ring_degree=$2
    local trace_folder=$3
    local mem_trace_folder=$4
    local attn_impl=$5
    local ring_comm_heads=$6

    overrides=(
        --parallelism.context_parallel_ulysses_degree "${ulysses_degree}"
        --parallelism.context_parallel_degree "${ring_degree}"
        --profiling.save_traces_folder "${trace_folder}"
        --profiling.save_memory_snapshot_folder "${mem_trace_folder}"
        --job.dump_folder "${DUMP_FOLDER}"
        --model.attn_impl "${attn_impl}"
        --model.ring_comm_heads "${ring_comm_heads}"
        --activation_checkpoint.mode "${CHECKPOINTING}"
        --activation_checkpoint.offloading "${OFFLOADING}"
        --training.seq_len "${CONTEXT_LENGTH}"
        --training.steps "${STEPS}"
        --profiling.enable_profiling
        --profiling.enable_memory_snapshot
        --model.flavor "${FLAVOR}"
        --profiling.profile_freq "${STEPS}"
        --metrics.log_freq "1"
        --training.batch_size "${BATCH_SIZE}"
    )

    if [ "${FSDP_OFFLOADING}" = "true" ]; then
        overrides+=(--training.enable_cpu_offload)
    fi

    if [ "${COMPILE}" = "true" ]; then
        overrides+=(--training.compile)
    fi
}

run_experiment() {
    local ulysses_degree=$1
    local ring_degree=$2
    local attn_impl=$3
    local ring_comm_heads=$4
    local trace_folder=$5
    local mem_trace_folder=$6

    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    ${VENV_PYTHON} -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend="${RDZV_BACKEND}" \
        --rdzv_endpoint="${RDZV_ENDPOINT}" \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --job.config_file "${CONFIG_FILE}" \
        "${overrides[@]}"
}

################################################################################
# Derived settings
################################################################################
# Adjust PIN_MEMORY for very long sequences per node
if [ "$((CONTEXT_LENGTH / NNODES))" -gt 4194304 ]; then
    INP_PIN_MEMORY="False"
fi

total_experiments=$((${#EXPERIMENTS[@]} * ${#ATTN_IMPL_OPTIONS[@]} * ${#RING_COMM_HEADS_OPTIONS[@]}))

echo "Starting multinode experiment run with ${total_experiments} configurations..."
echo "Multinode Config:"
echo "  - Nodes: ${NNODES}"
echo "  - Processes per node: ${NPROC_PER_NODE}"
echo "  - Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "  - Rendezvous ID: ${RDZV_ID}"
echo "  - Rendezvous endpoint: ${RDZV_ENDPOINT}"
echo "WandB project: ${WANDB_PROJECT}"
echo "Parallelism configs: ${#EXPERIMENTS[@]}"
echo "Attention implementations: ${#ATTN_IMPL_OPTIONS[@]} (${ATTN_IMPL_OPTIONS[*]})"
echo "Ring comm heads: ${#RING_COMM_HEADS_OPTIONS[@]} (${RING_COMM_HEADS_OPTIONS[*]})"
echo "Activation Offloading: ${OFFLOADING}"
echo "Activation Checkpointing: ${CHECKPOINTING}"
echo "FSDP CPU Offloading: ${FSDP_OFFLOADING}"

experiment_counter=0
failed_experiments=0
failed_experiment_list=()
successful_experiments=0

################################################################################
# Main loop
################################################################################
for config in "${EXPERIMENTS[@]}"; do
    read -r ulysses_degree ring_degree <<< "${config}"

    for attn_impl in "${ATTN_IMPL_OPTIONS[@]}"; do
        for ring_comm_heads in "${RING_COMM_HEADS_OPTIONS[@]}"; do
            experiment_counter=$((experiment_counter + 1))

            trace_folder=$(trace_folder_name "${ulysses_degree}" "${ring_degree}" "${ring_comm_heads}" "${attn_impl}")
            mem_trace_folder=$(mem_trace_folder_name "${ulysses_degree}" "${ring_degree}" "${ring_comm_heads}" "${attn_impl}")

            # Skip already completed runs.
            if [ -d "${DUMP_FOLDER}/${trace_folder}/iteration_${STEPS}" ]; then
                echo "⏭️  SKIPPING experiment ${experiment_counter}/${total_experiments}: ${trace_folder}"
                echo "   Reason: Already completed (iteration_${STEPS} folder exists)"
                echo "----------------------------------------"
                continue
            fi

            if should_skip_config "${config}" "${attn_impl}"; then
                continue
            fi

            experiment_name="${trace_folder}"
            job_description="Multinode Context Parallel: ${experiment_name}"

            echo "========================================"
            echo "Running experiment ${experiment_counter}/${total_experiments}: ${experiment_name}"
            echo "Ulysses degree: ${ulysses_degree}"
            echo "Ring degree: ${ring_degree}"
            echo "Attention implementation: ${attn_impl}"
            echo "Ring comm heads: ${ring_comm_heads}"
            echo "Trace folder: ${trace_folder}"
            echo "Memory trace folder: ${mem_trace_folder}"
            echo "Job description: ${job_description}"
            echo "========================================"

            mkdir -p "${DUMP_FOLDER}/${trace_folder}"
            cp -f "$(get_script_path)" "${DUMP_FOLDER}/${trace_folder}/run_script.sh"

            export WANDB_PROJECT="${WANDB_PROJECT}"
            export WANDB_NAME="${experiment_name}"
            export WANDB_TAGS="context_parallel,multinode,ulysses_${ulysses_degree},ring_${ring_degree},seq_$((CONTEXT_LENGTH / 1024))k,ACS_${AC_LAYER_STRIDE},BS_${BATCH_SIZE},${attn_impl},${ring_comm_heads}"

            build_overrides "${ulysses_degree}" "${ring_degree}" "${trace_folder}" "${mem_trace_folder}" "${attn_impl}" "${ring_comm_heads}"

            if PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
               cd "${TRAIN_ROOT}" && \
               run_experiment "${ulysses_degree}" "${ring_degree}" "${attn_impl}" "${ring_comm_heads}" "${trace_folder}" "${mem_trace_folder}"; then
                echo "✅ SUCCESSFULLY completed experiment ${experiment_counter}/${total_experiments}: ${experiment_name}"
                successful_experiments=$((successful_experiments + 1))
            else
                echo "❌ FAILED experiment ${experiment_counter}/${total_experiments}: ${experiment_name}"
                echo "   Error occurred during execution. Continuing with next experiment..."
                failed_experiments=$((failed_experiments + 1))
                failed_experiment_list+=("${experiment_name}")
            fi

            echo "----------------------------------------"
            echo "Progress: ${experiment_counter}/${total_experiments} completed (✅ ${successful_experiments} successful, ❌ ${failed_experiments} failed)"
            sleep 30
        done
    done
done

################################################################################
# Summary
################################################################################
echo ""
echo "========================================="
echo "🎉 MULTINODE EXPERIMENT BATCH COMPLETED!"
echo "========================================="
echo "Total experiments: ${total_experiments}"
echo "Successful: ${successful_experiments}"
echo "Failed: ${failed_experiments}"

if [ ${failed_experiments} -gt 0 ]; then
    echo ""
    echo "❌ Failed experiments:"
    for failed_exp in "${failed_experiment_list[@]}"; do
        echo "  - ${failed_exp}"
    done
fi

echo ""
echo "✅ Trace folders created for successful experiments:"
for config in "${EXPERIMENTS[@]}"; do
    read -r ulysses_degree ring_degree <<< "${config}"
    for attn_impl in "${ATTN_IMPL_OPTIONS[@]}"; do
        for ring_comm_heads in "${RING_COMM_HEADS_OPTIONS[@]}"; do
            trace_folder="${BASE_TRACE_FOLDER}_$((CONTEXT_LENGTH / 1024))k_ACS_${AC_LAYER_STRIDE}_BS_${BATCH_SIZE}_${ulysses_degree}u${ring_degree}r_${PROFILING_SUFFIX}_${ring_comm_heads}_${attn_impl}"
            echo "  - ${trace_folder}"
        done
    done
done
echo ""
echo "Check your WandB dashboard at: https://wandb.ai/[your-username]/${WANDB_PROJECT}"

# Exit with appropriate code
if [ ${failed_experiments} -gt 0 ]; then
    echo "⚠️  Some experiments failed. Check the logs above for details."
    exit 1
else
    echo "🎉 All experiments completed successfully!"
    exit 0
fi
