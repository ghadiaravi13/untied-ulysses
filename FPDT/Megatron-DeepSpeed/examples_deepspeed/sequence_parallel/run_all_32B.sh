#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/run_single_32B.sh"

if [ ! -x "$LAUNCH_SCRIPT" ]; then
    chmod +x "$LAUNCH_SCRIPT"
fi

# Sequence lengths to sweep (tokens)
SEQLENS=(131072 262144 524288 1048576 2097152 3145728 4194304)

for SL in "${SEQLENS[@]}"; do
  echo "===== Running Qwen3 FPDT training with seq_len=${SL} ====="
  # Run non-interactively and stream logs; each run produces its own timestamped log file as per the launch script
  "$LAUNCH_SCRIPT" "$SL"
done

echo "All runs completed."

