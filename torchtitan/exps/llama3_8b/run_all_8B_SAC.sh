#!/usr/bin/bash
# Full experiment runner for paper benchmarks
# Runs all methods across all sequence lengths

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
NNODES=${NNODES:-"1"}
NPROC_PER_NODE=${NPROC_PER_NODE:-"8"}
RDZV_ID=${RDZV_ID:-"101"}
RDZV_BACKEND=${RDZV_BACKEND:-"c10d"}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-$(hostname -s)}
NODE_RANK=${NODE_RANK:-""}  # Set by torchrun if not specified

################################################################################
# Training configuration
################################################################################
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b_test_2d.toml"}
DUMP_FOLDER=${DUMP_FOLDER:-"${ROOT_TMP}/run_all_8b_SAC"}
BASE_TRACE_FOLDER="run_all_8b_SAC"
MEM_BASE_TRACE_FOLDER="run_all_8b_SAC"

PROFILING_SUFFIX=""
COMPILE=${COMPILE:-"false"}
FLAVOR=${FLAVOR:-"8B"}

# AC/Offloading parameters
OFFLOADING=${OFFLOADING:-"cpu"} # ["no", "UAO", "cpu" "TAO"] Unsloth/Torchtune offloading
CHECKPOINTING=${CHECKPOINTING:-"async_selective"} # ["none", "async_selective", "sync_selective", "async_", "full"]
FSDP_OFFLOADING=${FSDP_OFFLOADING:-"false"} # ["true", "false"] FSDP CPU Offloading

WANDB_PROJECT=${WANDB_PROJECT:-"run_all_8b_SAC"}

STEPS=${STEPS:-"5"}
# Overridden each iteration when paired with SEQLENS[i]; default only if arrays are unused.
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

# see Untied-Ulysses/torchtitan/patch_torch_files/checkpoint_wrapper.py
# setting this to K will skip CPU offloading of the input to every Kth layer
export AC_LAYER_STRIDE=${AC_LAYER_STRIDE:-"1000"}

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
# Benchmark matrices
################################################################################
# Sequence lengths from the paper table (in tokens)
SEQLENS=(
    131072    # 128K
    262144    # 256K
    524288    # 512K
    1048576   # 1M
    # 2097152   # 2M
    # 3145728   # 3M
    # 4194304   # 4M
    # 5242880   # 5M
)

# Aligned with SEQLENS by index: batch * seqlen ≈ 3_145_728 tokens (~3M) per step.
BATCH_SIZES=(
    1
    1
    1
    1
)

if [ "${#SEQLENS[@]}" -ne "${#BATCH_SIZES[@]}" ]; then
    echo "Error: SEQLENS (${#SEQLENS[@]} entries) and BATCH_SIZES (${#BATCH_SIZES[@]} entries) must have the same length."
    exit 1
fi

# Methods: "name attn_impl ulysses ring"
METHODS=(
    "upipe_SAC upipe_fa3_offload_tiled_mlp 8 1"
    # "usp-ulysses usp_fa3_offload_tiled_mlp 8 1"
    # "usp-ring usp_fa3_offload_tiled_mlp 1 8"
    # "torch torch_ring_alltoall 1 8"
)

RING_COMM_HEADS="gqa_kv"

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
    local method_name=$1
    local seqlen_k=$2

    echo "${BASE_TRACE_FOLDER}_${method_name}_${seqlen_k}k_BS_${BATCH_SIZE}_ao_${OFFLOADING}_ac_${CHECKPOINTING}"
}

build_overrides() {
    local ulysses_degree=$1
    local ring_degree=$2
    local trace_folder=$3
    local mem_trace_folder=$4
    local attn_impl=$5
    local seqlen=$6

    overrides=(
        --parallelism.context_parallel_ulysses_degree "${ulysses_degree}"
        --parallelism.context_parallel_degree "${ring_degree}"
        --profiling.save_traces_folder "${trace_folder}"
        --profiling.save_memory_snapshot_folder "${mem_trace_folder}"
        --job.dump_folder "${DUMP_FOLDER}"
        --model.attn_impl "${attn_impl}"
        --model.ring_comm_heads "${RING_COMM_HEADS}"
        --activation_checkpoint.mode "${CHECKPOINTING}"
        --activation_checkpoint.offloading "${OFFLOADING}"
        --training.seq_len "${seqlen}"
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
# Main
################################################################################
total_experiments=$((${#METHODS[@]} * ${#SEQLENS[@]}))

echo "========================================="
echo "Starting paper benchmark experiments"
echo "========================================="
echo "Multinode Config:"
echo "  - Nodes: ${NNODES}"
echo "  - Processes per node: ${NPROC_PER_NODE}"
echo "  - Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "  - Rendezvous endpoint: ${RDZV_ENDPOINT}"
echo ""
echo "Methods: ${#METHODS[@]}"
for method_config in "${METHODS[@]}"; do
    read -r method_name attn_impl ulysses ring <<< "${method_config}"
    echo "  - ${method_name}: ${attn_impl} (${ulysses}u${ring}r)"
done
echo ""
echo "Sequence lengths (paired batch → ~3M tokens/step): ${#SEQLENS[@]}"
for ((si=0; si<${#SEQLENS[@]}; si++)); do
    s="${SEQLENS[si]}"
    b="${BATCH_SIZES[si]}"
    echo "  - $((s / 1024))K  batch=${b}  tokens/step=$((s * b))"
done
echo ""
echo "Total experiments: ${total_experiments}"
echo "========================================="

experiment_counter=0
failed_experiments=0
failed_experiment_list=()
successful_experiments=0
skipped_experiments=0

# Results table
declare -A results_table

################################################################################
# Main loop: iterate methods x seqlens
################################################################################
for method_config in "${METHODS[@]}"; do
    read -r method_name attn_impl ulysses_degree ring_degree <<< "${method_config}"

    for ((si=0; si<${#SEQLENS[@]}; si++)); do
        seqlen=${SEQLENS[si]}
        BATCH_SIZE=${BATCH_SIZES[si]}
        seqlen_k=$((seqlen / 1024))
        experiment_counter=$((experiment_counter + 1))

        # Adjust PIN_MEMORY when total tokens per step (batch * seq) is very large
        if [ $((BATCH_SIZE * seqlen)) -gt 4194304 ]; then
            export INP_PIN_MEMORY="False"
        else
            export INP_PIN_MEMORY="True"
        fi

        trace_folder=$(trace_folder_name "${method_name}" "${seqlen_k}")
        mem_trace_folder="${trace_folder}"

        # Skip already completed runs
        if [ -d "${DUMP_FOLDER}/${trace_folder}/iteration_${STEPS}" ]; then
            echo "⏭️  SKIPPING [${experiment_counter}/${total_experiments}]: ${method_name} @ ${seqlen_k}K batch=${BATCH_SIZE}"
            echo "   Reason: Already completed (iteration_${STEPS} folder exists)"
            echo "----------------------------------------"
            skipped_experiments=$((skipped_experiments + 1))
            results_table["${method_name},${seqlen_k}"]="SKIP"
            continue
        fi

        experiment_name="${method_name}_${seqlen_k}k"

        echo ""
        echo "========================================"
        echo "[${experiment_counter}/${total_experiments}] ${method_name} @ ${seqlen_k}K batch=${BATCH_SIZE}"
        echo "========================================"
        echo "  attn_impl: ${attn_impl}"
        echo "  ulysses: ${ulysses_degree}, ring: ${ring_degree}"
        echo "  seqlen: ${seqlen}"
        echo "  batch_size: ${BATCH_SIZE} (tokens/step: $((seqlen * BATCH_SIZE)))"
        echo "  trace_folder: ${trace_folder}"
        echo "========================================"

        mkdir -p "${DUMP_FOLDER}/${trace_folder}"
        cp -f "$(get_script_path)" "${DUMP_FOLDER}/${trace_folder}/run_script.sh"

        export WANDB_PROJECT="${WANDB_PROJECT}"
        export WANDB_NAME="${experiment_name}"
        export WANDB_TAGS="benchmark,${method_name},seq_${seqlen_k}k,bs_${BATCH_SIZE},${ulysses_degree}u${ring_degree}r"

        build_overrides "${ulysses_degree}" "${ring_degree}" "${trace_folder}" "${mem_trace_folder}" "${attn_impl}" "${seqlen}"

        if PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
           cd "${TRAIN_ROOT}" && \
           run_experiment; then
            echo "✅ SUCCESS: ${method_name} @ ${seqlen_k}K batch=${BATCH_SIZE}"
            successful_experiments=$((successful_experiments + 1))
            results_table["${method_name},${seqlen_k}"]="OK"
        else
            echo "❌ FAILED: ${method_name} @ ${seqlen_k}K batch=${BATCH_SIZE} (likely OOM)"
            failed_experiments=$((failed_experiments + 1))
            failed_experiment_list+=("${experiment_name}")
            results_table["${method_name},${seqlen_k}"]="FAIL"
        fi

        echo "----------------------------------------"
        echo "Progress: ${experiment_counter}/${total_experiments} (✅ ${successful_experiments} | ❌ ${failed_experiments} | ⏭️ ${skipped_experiments})"
        sleep 10
    done
done

################################################################################
# Summary
################################################################################
echo ""
echo "========================================="
echo "🎉 BENCHMARK SUITE COMPLETED!"
echo "========================================="
echo "Total experiments: ${total_experiments}"
echo "Successful: ${successful_experiments}"
echo "Failed: ${failed_experiments}"
echo "Skipped: ${skipped_experiments}"

# Print results table
echo ""
echo "Results Matrix:"
echo "==============="
printf "%-15s" "Method"
for seqlen in "${SEQLENS[@]}"; do
    printf "%-8s" "$((seqlen / 1024))K"
done
echo ""
printf "%-15s" "---------------"
for seqlen in "${SEQLENS[@]}"; do
    printf "%-8s" "--------"
done
echo ""

for method_config in "${METHODS[@]}"; do
    read -r method_name _ _ _ <<< "${method_config}"
    printf "%-15s" "${method_name}"
    for seqlen in "${SEQLENS[@]}"; do
        seqlen_k=$((seqlen / 1024))
        result="${results_table[${method_name},${seqlen_k}]:-N/A}"
        printf "%-8s" "${result}"
    done
    echo ""
done

if [ ${failed_experiments} -gt 0 ]; then
    echo ""
    echo "❌ Failed experiments:"
    for failed_exp in "${failed_experiment_list[@]}"; do
        echo "  - ${failed_exp}"
    done
fi

echo ""
echo "Output folder: ${DUMP_FOLDER}"
echo "WandB: https://wandb.ai/[your-username]/${WANDB_PROJECT}"

# Exit with appropriate code
if [ ${failed_experiments} -gt 0 ]; then
    echo "⚠️  Some experiments failed (likely OOM at longer sequences)."
    exit 1
else
    echo "🎉 All experiments completed successfully!"
    exit 0
fi
