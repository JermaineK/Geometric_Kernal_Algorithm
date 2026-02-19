#!/usr/bin/env bash
set -euo pipefail

RAW_PATH="${1:-data/raw/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet}"
DATASET_DIR="${2:-dataset/weather_minipilot}"
RESULTS_DIR="${3:-results/weather_minipilot}"

python examples/weather_minipilot/prepare_minipilot.py \
  --input "${RAW_PATH}" \
  --out "${DATASET_DIR}" \
  --max-pairs 600

gka validate "${DATASET_DIR}"
gka run "${DATASET_DIR}" \
  --domain weather \
  --config examples/weather_minipilot/config.yaml \
  --out "${RESULTS_DIR}" \
  --dump-intermediates

gka report --in "${RESULTS_DIR}" --out "${RESULTS_DIR}/report.html"
gka audit "${RESULTS_DIR}"
