#!/usr/bin/env bash
###############################################################################
# nvlink_intranode_bench.sh
# -----------------------------------------------------------------------------
# Single-node NVLink / NVSwitch benchmark for 8x H100. Uses the prebuilt
# NCCL tests at /usr/local/bin/{sendrecv,all_reduce,alltoall,reduce_scatter,
# all_gather}_perf.
#
# Probes:
#   1. Pairwise P2P NVLink BW for all 28 GPU pairs (sendrecv_perf, g=2)
#   2. 8-GPU all-reduce sweep (8 B -> 8 GiB, x2)
#   3. 8-GPU all-to-all  sweep (1 MiB -> 1 GiB, x2)   <- maps to Ulysses CP
#   4. 8-GPU reduce-scatter / all-gather sweep        <- maps to FSDP
#
# Two configurations are run for each collective sweep:
#     a) ISO-TRAIN  : same NCCL env as run_all_multinode_70B_BS.sh
#     b) NCCL-DEFAULT: stock NCCL defaults  (peak hardware)
# so you can see whether the training-time NCCL tuning is leaving
# bandwidth on the table.
#
# Usage:
#     bash nvlink_intranode_bench.sh
#     LOG_DIR=/tmp/foo bash nvlink_intranode_bench.sh
#     ITERS=100 WARMUP=20 bash nvlink_intranode_bench.sh
###############################################################################

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTN="$(hostname -s)"
TS="$(date +%Y%m%d_%H%M%S)"

LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs/intranode_${HOSTN}_${TS}"}
mkdir -p "${LOG_DIR}"

NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-"/usr/local/bin"}
SENDRECV_BIN="${NCCL_TESTS_DIR}/sendrecv_perf"
ALLRED_BIN="${NCCL_TESTS_DIR}/all_reduce_perf"
ALLTOALL_BIN="${NCCL_TESTS_DIR}/alltoall_perf"
RDSCAT_BIN="${NCCL_TESTS_DIR}/reduce_scatter_perf"
ALLGAT_BIN="${NCCL_TESTS_DIR}/all_gather_perf"

for b in "${SENDRECV_BIN}" "${ALLRED_BIN}" "${ALLTOALL_BIN}" "${RDSCAT_BIN}" "${ALLGAT_BIN}"; do
    if [ ! -x "$b" ]; then
        echo "ERROR: missing nccl-tests binary: $b" >&2
        exit 2
    fi
done

ITERS=${ITERS:-50}
WARMUP=${WARMUP:-20}

###############################################################################
# Profile A: ISO-TRAIN (must match run_all_multinode_70B_BS.sh exactly)
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

# For *intranode* runs we additionally force NCCL to take the NVLink path
# (no SHM / no IB) so the numbers cleanly characterize NVSwitch.
force_nvlink_only() {
    export NCCL_P2P_DISABLE=0
    export NCCL_P2P_LEVEL=NVL
    export NCCL_SHM_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_NET_DISABLE=1
}

print_env_block() {
    echo "----- nccl env -----"
    env | grep -E '^(NCCL_|CUDA_DEVICE_ORDER)' | sort
    echo "--------------------"
}

###############################################################################
# 1. Pairwise P2P NVLink BW  (28 unique pairs, ISO-TRAIN env)
###############################################################################
pairwise_section() {
    apply_iso_train_env
    force_nvlink_only

    local outdir="${LOG_DIR}/pairwise_p2p"
    mkdir -p "${outdir}"
    local summary="${outdir}/_summary.csv"
    echo "gpu_a,gpu_b,size_bytes,algbw_GBps,busbw_GBps" > "${summary}"

    echo
    echo "============================================================"
    echo " 1) Pairwise P2P NVLink BW   (sendrecv_perf, g=2, ISO-TRAIN)"
    echo "============================================================"
    print_env_block | tee "${outdir}/_env.txt"

    for i in 0 1 2 3 4 5 6 7; do
        for j in $(seq $((i + 1)) 7); do
            local log="${outdir}/p2p_${i}_${j}.log"
            CUDA_VISIBLE_DEVICES="${i},${j}" \
                "${SENDRECV_BIN}" -g 2 -b 8 -e 1G -f 2 \
                                  -n "${ITERS}" -w "${WARMUP}" -c 0 \
                > "${log}" 2>&1
            if [ $? -ne 0 ]; then
                echo "  pair (${i},${j}): FAILED  see ${log}"
                continue
            fi
            # Parse the largest-size row (last numeric data line) and emit CSV.
            awk -v a="${i}" -v b="${j}" '
                /^# *Out of bounds/ {next}
                /^[[:space:]]*[0-9]/ {
                    size=$1; algbw=$7; busbw=$8;
                    print a","b","size","algbw","busbw;
                }' "${log}" >> "${summary}"
            local peak
            peak=$(awk '/^[[:space:]]*[0-9]/{print $8}' "${log}" | sort -g | tail -1)
            printf "  pair (%d,%d):  peak busBW = %s GB/s\n" "${i}" "${j}" "${peak}"
        done
    done

    echo
    echo "Pairwise summary CSV: ${summary}"
}

###############################################################################
# 2-4. Collective sweeps over 8 GPUs, run twice: ISO-TRAIN + NCCL-DEFAULT
###############################################################################
run_collective() {
    local label=$1   ;# pretty name
    local tag=$2     ;# short tag for filenames
    local bin=$3     ;# binary path
    local minb=$4
    local maxb=$5
    local profile=$6 ;# "iso_train" | "default"

    local outdir="${LOG_DIR}/${tag}_${profile}"
    mkdir -p "${outdir}"
    local log="${outdir}/run.log"

    if [ "${profile}" = "iso_train" ]; then
        apply_iso_train_env
    else
        apply_default_env
    fi
    force_nvlink_only

    echo
    echo "------------------------------------------------------------"
    echo " ${label}   profile=${profile}   (g=8)"
    echo "------------------------------------------------------------"
    print_env_block | tee "${outdir}/_env.txt"

    "${bin}" -g 8 -b "${minb}" -e "${maxb}" -f 2 \
             -n "${ITERS}" -w "${WARMUP}" -c 0 \
             > "${log}" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED (rc=${rc}); see ${log}"
        return
    fi

    # Pretty mini-summary: largest-size in/out-of-place busBW
    awk '
        /^# *Out of bounds/ {next}
        /^[[:space:]]*[0-9]/ {
            size=$1;
            oop_alg=$7; oop_bus=$8;
            ip_alg=$11; ip_bus=$12;
            printf("  size=%-12s OOP busBW=%6s GB/s   IP busBW=%6s GB/s\n",
                   size, oop_bus, ip_bus);
        }' "${log}"
}

collective_sweeps() {
    for profile in iso_train default; do
        echo
        echo "############################################################"
        echo "## Collective sweeps  -- profile=${profile}"
        echo "############################################################"

        run_collective "AllReduce"      "all_reduce"     "${ALLRED_BIN}"   8     8G  "${profile}"
        run_collective "AllToAll"       "alltoall"       "${ALLTOALL_BIN}" 1M    1G  "${profile}"
        run_collective "ReduceScatter"  "reduce_scatter" "${RDSCAT_BIN}"   8     8G  "${profile}"
        run_collective "AllGather"      "all_gather"     "${ALLGAT_BIN}"   8     8G  "${profile}"
    done
}

###############################################################################
# Main
###############################################################################
echo "==================================================================="
echo " NVLink intranode benchmark  on $(hostname)  @ $(date)"
echo " Log dir: ${LOG_DIR}"
echo " GPUs:"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | sed 's/^/   /'
echo "==================================================================="

pairwise_section
collective_sweeps

echo
echo "==================================================================="
echo " DONE.  All logs under: ${LOG_DIR}"
echo "==================================================================="
