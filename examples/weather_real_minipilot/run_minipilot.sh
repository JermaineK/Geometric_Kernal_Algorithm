#!/usr/bin/env bash
set -euo pipefail

RAW_PATH="${1:-data/raw/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet}"
PREPARED_ROOT="${2:-data/prepared/weather_v1}"
TILES_DATASET="${3:-data/tiles/weather_real_v1}"
RESULTS_DIR="${4:-results/weather_real_minipilot}"
SPLIT_DATE="${5:-2025-04-01}"
BUFFER_HOURS="${6:-72}"
IBTRACS_PATH="${7:-}"

python examples/weather_real_minipilot/prepare_weather_v1.py \
  --input "${RAW_PATH}" \
  --out "${PREPARED_ROOT}" \
  --lon0-sweep 145 150 155 160

IBTRACS_ARGS=()
if [[ -n "${IBTRACS_PATH}" ]]; then
  IBTRACS_ARGS+=(--ibtracs-csv "${IBTRACS_PATH}")
fi

python examples/weather_real_minipilot/build_tiles.py \
  --prepared-root "${PREPARED_ROOT}" \
  --out "${TILES_DATASET}" \
  --cohort all \
  --centers hybrid \
  --distance-time-tolerance-hours 3.0 \
  --distance-far-km 1000 \
  --distance-far-loose-km 2000 \
  --scan-batch-rows 120000 \
  --background-pool-max-rows 300000 \
  --background-max-batches-per-lead 0 \
  --vortex-max-fragments-per-lead 0 \
  --control-mode matched_background \
  --allow-nonexact-controls \
  --require-physical-match \
  --physical-speed-bin-ms 2.0 \
  --physical-zeta-bin 1.0e-5 \
  --match-lat-bin-deg 2.0 \
  --match-lon-bin-deg 2.0 \
  --anomaly-mode lat_hour \
  --anomaly-lat-bin-deg 1.0 \
  --anomaly-lon-bin-deg 2.0 \
  --anomaly-commute-rmse-max 4.0 \
  --anomaly-commute-corr-min 0.05 \
  --anomaly-min-bin-samples 24 \
  --anomaly-min-covered-frac 0.70 \
  --polar-enable \
  --polar-r-bins 64 \
  --polar-theta-bins 128 \
  --polar-pitches -1.5 -1.0 -0.7 0.7 1.0 1.5 \
  --min-distinct-scales-per-case 4 \
  --min-controls-per-lead-bucket 2 \
  --min-far-nonstorm-per-lead-bucket 2 \
  --adaptive-control-coverage \
  --lead-buckets 24 120 240 none \
  --max-events-per-lead 12 \
  "${IBTRACS_ARGS[@]}"

gka validate "${TILES_DATASET}"

gka run "${TILES_DATASET}" \
  --domain weather \
  --config examples/weather_real_minipilot/pipeline_config.yaml \
  --out "${RESULTS_DIR}" \
  --dump-intermediates

gka report --in "${RESULTS_DIR}" --out "${RESULTS_DIR}/report.html"
gka audit "${RESULTS_DIR}"

python examples/weather_real_minipilot/evaluate_minipilot.py \
  --dataset "${TILES_DATASET}" \
  --config examples/weather_real_minipilot/pipeline_config.yaml \
  --out "${RESULTS_DIR}/evaluation.json" \
  --split-date "${SPLIT_DATE}" \
  --time-buffer-hours "${BUFFER_HOURS}" \
  --min-train-cases 15 \
  --min-test-cases 15 \
  --min-event-test-cases 10 \
  --min-event-test-per-lead 0 \
  --min-far-nonstorm-test-cases 6 \
  --min-far-nonstorm-test-per-lead 1 \
  --axis-sensitivity-json "${PREPARED_ROOT}/lon0_sensitivity.json" \
  --axis-std-threshold 0.02 \
  --parity-claim-mode auto \
  --parity-confound-max-ratio 0.6 \
  --parity-event-minus-far-min 0.15 \
  --parity-far-max 0.35 \
  --anomaly-agreement-mean-min 0.85 \
  --anomaly-agreement-min-min 0.80 \
  --anomaly-mode-agreement-min 0.70 \
  --mode-invariant-event-minus-far-min 0.15 \
  --mode-invariant-far-max 0.35 \
  --angular-witness-margin-min 0.10 \
  --angular-witness-null-drop-min 0.30 \
  --angular-witness-null-abs-drop-min 0.05 \
  --null-collapse-min-drop 0.50 \
  --null-collapse-min-abs-drop 0.15 \
  --knee-weather-delta-bic-min 6.0 \
  --knee-weather-resid-improvement-min 0.75 \
  --knee-weather-slope-delta-min 0.20 \
  --knee-weather-min-consistent-modes 2 \
  --knee-weather-knee-l-rel-tol 0.35 \
  --enable-time-frequency-knee \
  --tf-knee-bic-delta-min 6.0 \
  --slowtick-delta-min 0.05 \
  --slowtick-p-max 0.10 \
  --ibtracs-strict-use-flags
