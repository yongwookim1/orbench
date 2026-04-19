#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/shieldgemma-2b}"
DATA_DIR="${SCRIPT_DIR}/or-bench"
OUTPUT_DIR="${SCRIPT_DIR}/results/shieldgemma_2b"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

python3 "${SCRIPT_DIR}/evaluate_guard_models_orbench.py" \
  --model "${MODEL_PATH}" \
  --model-type shieldgemma \
  --input-files \
    "${DATA_DIR}/or-bench-80k.csv" \
    "${DATA_DIR}/or-bench-hard-1k.csv" \
    "${DATA_DIR}/or-bench-toxic.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  "$@"
