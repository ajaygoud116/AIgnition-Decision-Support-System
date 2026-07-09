#!/usr/bin/env bash
set -euo pipefail

# AIgnition Forecasting Pipeline
# Hackathon submission entry point.
#
# Usage: ./run.sh [DATA_DIR] [MODEL_PATH] [OUTPUT_PATH]
#
# Arguments (all optional, with defaults):
#   DATA_DIR     Folder containing input CSV files (default: ./data)
#   MODEL_PATH   Path to pickled model (default: ./pickle/model.pkl)
#   OUTPUT_PATH  Where to write predictions CSV (default: ./output/predictions.csv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:-${SCRIPT_DIR}/data}"
MODEL_PATH="${2:-${SCRIPT_DIR}/pickle/model.pkl}"
OUTPUT_PATH="${3:-${SCRIPT_DIR}/output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python -m src.pipeline.main \
    --data-dir "$DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --output-dir "$(dirname "$OUTPUT_PATH")"

echo "Done. Predictions written to $OUTPUT_PATH"
