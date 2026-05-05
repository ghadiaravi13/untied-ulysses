#!/usr/bin/env bash
###############################################################################
# nvlink_internode_bench.sh
# -----------------------------------------------------------------------------
# Multi-node NCCL benchmark across 4x (8x H100) using torchrun's c10d
# rendezvous (no SSH / no MPI bootstrap needed). Run this script on EVERY node
# with the same RDZV_ENDPOINT pointing at the master.
#
# USAGE  (4-node, master = dev-rghadia-1f9iv6):
#   # On every node:
#   RDZV_ENDPOINT="dev-rghadia-1f9iv6:29500" \
#       bash nvlink_internode_bench.sh
#
# Optional env knobs:
#   NNODES               default 4
#   NPROC_PER_NODE       default 8
#   RDZV_ID              default 9001
#   ITERS                default 50
#   WARMUP               default 20
#   PROFILE              "iso_train" (default) | "default"  | "both"
#   SIZES_BYTES          space-separated overrides for collective sweep sizes
#   LOG_DIR              output directory
###############################################################################

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PY_BENCH="${SCRIPT_DIR}/nvlink_internode_bench.py"

if [ ! -x "${VENV_PYTHON}" ] && [ ! -f "${VENV_PYTHON}" ]; then
    echo "ERROR: venv python not found at ${VENV_PYTHON}" >&2
    exit 2
fi
if [ ! -f "${PY_BENCH}" ]; then
    echo "ERROR: missing python harness ${PY_BENCH}" >&2
    exit 2
fi

###############################################################################
# Multinode config (mirror of run_all_multinode_70B_BS.sh)
###############################################################################
NNODES=${NNODES:-"4"}
NPROC_PER_NODE=${NPROC_PER_NODE:-"8"}
RDZV_ID=${RDZV_ID:-"9001"}
RDZV_BACKEND=${RDZV_BACKEND:-"c10d"}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"dev-rghadia-1f9iv6:29500"}

ITERS=${ITERS:-50}
WARMUP=${WARMUP:-20}
PROFILE=${PROFILE:-"iso_train"}   # iso_train | default | both

HOSTN="$(hostname -s)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs/internode_${TS}"}
mkdir -p "${LOG_DIR}"

###############################################################################
# NCCL env -- ISO-TRAIN (verbatim from run_all_multinode_70B_BS.sh)
###############################################################################
apply_iso_train_env() {
    export NCCL_ALGO="RING"
    export NCCL_IGNORE_CPU_AFFINITY="1"
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export NCCL_IB_AR_THRESHOLD="0"
    export NCCL_IB_PCI_RELAXED_ORDERING="1"
    export NCCL_IB_SPLIT_DATA_ON_QPS="0"
    export NCCL_IB_QPS_PER_CONNECTION="2"
}

apply_default_env() {
    unset NCCL_ALGO NCCL_IGNORE_CPU_AFFINITY \
          NCCL_IB_AR_THRESHOLD NCCL_IB_PCI_RELAXED_ORDERING \
          NCCL_IB_SPLIT_DATA_ON_QPS NCCL_IB_QPS_PER_CONNECTION
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
}

# Optional: turn on NCCL_DEBUG=INFO for first-rank visibility into transport
# selection (set NCCL_DEBUG=INFO via env to enable; off by default to keep
# logs small).
: "${NCCL_DEBUG_SUBSYS:=INIT,NET,GRAPH}"
export NCCL_DEBUG_SUBSYS

run_one_profile() {
    local profile=$1
    local profile_log_dir="${LOG_DIR}/${profile}_${HOSTN}"
    mkdir -p "${profile_log_dir}"

    if [ "${profile}" = "iso_train" ]; then
        apply_iso_train_env
    else
        apply_default_env
    fi

    echo
    echo "==================================================================="
    echo " Internode NCCL bench  profile=${profile}"
    echo "  host=${HOSTN}  nnodes=${NNODES}  nproc/node=${NPROC_PER_NODE}"
    echo "  rdzv_endpoint=${RDZV_ENDPOINT}"
    echo "  log_dir=${profile_log_dir}"
    echo "==================================================================="
    echo "----- NCCL env -----"
    env | grep -E '^(NCCL_|CUDA_DEVICE_ORDER)' | sort | tee "${profile_log_dir}/_env.txt"
    echo "--------------------"

    "${VENV_PYTHON}" -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend="${RDZV_BACKEND}" \
        --rdzv_endpoint="${RDZV_ENDPOINT}" \
        --role bench \
        --tee 3 \
        "${PY_BENCH}" \
        --iters "${ITERS}" \
        --warmup "${WARMUP}" \
        --log-dir "${profile_log_dir}" \
        --profile-tag "${profile}" \
        ${SIZES_BYTES:+--sizes ${SIZES_BYTES}}
}

case "${PROFILE}" in
    iso_train) run_one_profile iso_train ;;
    default)   run_one_profile default ;;
    both)      run_one_profile iso_train; run_one_profile default ;;
    *) echo "Unknown PROFILE=${PROFILE} (expected iso_train|default|both)" >&2; exit 2 ;;
esac

echo
echo "==================================================================="
echo " DONE on ${HOSTN}.  Logs under ${LOG_DIR}"
echo " (each rank wrote a JSON; rank-0 also wrote summary tables)"
echo "==================================================================="
