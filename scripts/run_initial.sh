#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <experiment_data_dir> <initial_idx_csv> <target_prompt> [extra_args...]"
    echo "Example: $0 samples/isaacsim_seq_16 0,1,2,3,4,5,6,7 \"a red cup\""
    exit 1
fi

EXPERIMENT_DATA_DIR="$1"
INITIAL_IDX_CSV="$2"
TARGET_PROMPT="$3"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a INITIAL_IDX <<< "${INITIAL_IDX_CSV}"

python -u -m clutt3rseg.initial_segmenter \
    --experiment_data_dir "${EXPERIMENT_DATA_DIR}" \
    --voxel_size 0.005 \
    --initial_idx "${INITIAL_IDX[@]}" \
    --target_prompt "${TARGET_PROMPT}" \
    "$@"
