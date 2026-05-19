#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <experiment_data_dir> <update_fid> <update_num> <prompt> [extra_args...]"
    echo "Example: $0 samples/isaacsim_seq_16 8 1 \"a red cup\""
    exit 1
fi

EXPERIMENT_DATA_DIR="$1"
UPDATE_FID="$2"
UPDATE_NUM="$3"
PROMPT="$4"
shift 4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -u -m clutt3rseg.update_segmenter \
    --experiment_data_dir "${EXPERIMENT_DATA_DIR}" \
    --update_fid "${UPDATE_FID}" \
    --update_num "${UPDATE_NUM}" \
    --prompt "${PROMPT}" \
    "$@"
