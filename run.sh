#!/usr/bin/env bash
# AIgnition Forecasting Pipeline — Hackathon submission entry point.
#
# Usage:   ./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH
# Example: ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
#
# DATA_DIR    Folder containing input CSV files
# MODEL_PATH  Path to pickled model
# OUTPUT_PATH File path where predictions CSV is written (pipeline generates
#             forecasts.csv internally then copies it here)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:?Usage: $0 DATA_DIR MODEL_PATH OUTPUT_PATH}"
MODEL_PATH="${2:?Usage: $0 DATA_DIR MODEL_PATH OUTPUT_PATH}"
OUTPUT_PATH="${3:?Usage: $0 DATA_DIR MODEL_PATH OUTPUT_PATH}"

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python -m src.pipeline.main \
    --data-dir "$DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR"

# The pipeline writes forecasts.csv internally; copy to the requested path
cp "$OUTPUT_DIR/forecasts.csv" "$OUTPUT_PATH"
echo "Done. Predictions written to $OUTPUT_PATH"
