#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/run_single_multinode_70B.sh"

if [ ! -x "$LAUNCH_SCRIPT" ]; then
    chmod +x "$LAUNCH_SCRIPT"
fi

# Sequence lengths to sweep (tokens)
SEQLENS=(131072 262144 524288 1048576 2097152 3145728 4194304 5242880)

# ---------- multi-node coordination ----------
MASTER_ADDR=${MASTER_ADDR:?"ERROR: MASTER_ADDR must be set (hostname or IP of rank-0 node)"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:?"ERROR: NODE_RANK must be set (0 for master, 1..N-1 for workers)"}
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

# All node hostnames, comma-separated.  Used to generate the hostfile that
# DeepSpeed requires for topology discovery.  No SSH is actually performed.
NODE_ADDRESSES=${NODE_ADDRESSES:?"ERROR: NODE_ADDRESSES must be set (comma-separated list of all nodes)"}

for SL in "${SEQLENS[@]}"; do
  echo "===== Running Llama3-70B FPDT training with seq_len=${SL} ====="
  # Run non-interactively and stream logs; each run produces its own timestamped log file as per the launch script
  MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" NODE_RANK="${NODE_RANK}" NUM_NODES="${NUM_NODES}" NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE}" NODE_ADDRESSES="${NODE_ADDRESSES}" "$LAUNCH_SCRIPT" "$SL"
done

echo "All runs completed."
