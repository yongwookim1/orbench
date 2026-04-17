#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/Llama-Guard-4-12B"
DATA_DIR="${SCRIPT_DIR}/or-bench"
OUTPUT_DIR="${SCRIPT_DIR}/results/llama_guard_4_12b"

python "${SCRIPT_DIR}/evaluate_guard_models_orbench.py" \
  --model "${MODEL_PATH}" \
  --model-type llama-guard-4 \
  --input-files \
    "${DATA_DIR}/or-bench-80k.csv" \
    "${DATA_DIR}/or-bench-hard-1k.csv" \
    "${DATA_DIR}/or-bench-toxic.csv" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
