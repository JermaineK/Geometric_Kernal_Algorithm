# Weather Real Minipilot

This example turns the 70M-row weather parquet into a lead-aware, mirror-aware, vortex-centered GKA workflow.

## What it does

1. Stream raw parquet row groups and write prepared partitions:
   - `data/prepared/weather_v1/lead=<bucket>/date=<YYYY-MM-DD>/part-*.parquet`
   - `data/prepared/weather_v1/mirror_audit.json`
   - `data/prepared/weather_v1/lon0_sensitivity.json` (optional, when `--lon0-sweep` is set)
2. Discover event centers via vortex detection + tracking (or fallback labels):
   - candidates from vorticity + Okubo-Weiss
   - tracked into storm-like paths and converted to `case_id`
   - outputs: `vortex_candidates.parquet`, `storm_tracks.parquet`
3. Derive mirror channels about longitude `150.0`:
   - vector transform after reflection: `u -> -u`, `v -> v`
4. Add parity channels:
   - `u_even/u_odd`, `v_even/v_odd`, `eta_parity`
5. Build storm-centered tile aggregates into canonical GKA dataset:
   - `data/tiles/weather_real_v1/dataset.yaml`
   - `data/tiles/weather_real_v1/samples.parquet`
   - `data/tiles/weather_real_v1/polar_features.parquet`
   - optional `data/tiles/weather_real_v1/ibtracs_match.json` when `--ibtracs-csv` is provided
   - supports `--cohort all|events|background`
   - supports `--anomaly-mode lat_hour|lat_day|none` to remove mean-flow before parity aggregation
   - supports `--event-source vortex_or_labels|vortex|labels|ibtracs`
   - supports storm-centered polar diagnostics via `--polar-enable`
   - supports matched-background control mode (`--control-mode matched_background`)
   - uses pyarrow streaming controls for large corpora:
     - `--scan-batch-rows`
     - `--background-pool-max-rows`
     - `--background-max-batches-per-lead`
6. Run GKA + generate report + blocked-split evaluation.

## Run end-to-end

```bash
bash examples/weather_real_minipilot/run_minipilot.sh
```

Or run stepwise:

```bash
python examples/weather_real_minipilot/prepare_weather_v1.py \
  --input data/raw/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \
  --out data/prepared/weather_v1 \
  --lon0-sweep 145 150 155 160
```

```bash
python examples/weather_real_minipilot/build_tiles.py \
  --prepared-root data/prepared/weather_v1 \
  --out data/tiles/weather_real_v1 \
  --cohort all \
  --event-source vortex_or_labels \
  --control-mode matched_background \
  --polar-enable \
  --anomaly-mode lat_hour \
  --lead-buckets 24 120 240 none \
  --max-events-per-lead 12
```

```bash
gka validate data/tiles/weather_real_v1
gka run data/tiles/weather_real_v1 --domain weather \
  --config examples/weather_real_minipilot/pipeline_config.yaml \
  --out results/weather_real_minipilot --dump-intermediates
gka report --in results/weather_real_minipilot --out results/weather_real_minipilot/report.html
gka audit results/weather_real_minipilot
```

```bash
python examples/weather_real_minipilot/evaluate_minipilot.py \
  --dataset data/tiles/weather_real_v1 \
  --config examples/weather_real_minipilot/pipeline_config.yaml \
  --out results/weather_real_minipilot/evaluation.json \
  --split-date 2025-04-01 \
  --time-buffer-hours 72 \
  --enable-time-frequency-knee \
  --slowtick-delta-min 0.05 \
  --slowtick-p-max 0.10
```

## Leakage controls included

- lead-bucket partitioning and case separation by lead
- blocked time split with exclusion buffer around the boundary
- split audit in output (`split_audit`: case overlap, storm-id overlap when available, time-buffer violations)
- null checks:
  - direction randomization (L/R swap)
  - spatial shuffle control
  - time permutation within lead bucket
  - fake mirror pairing (wrong mirror pairing surrogate)
  - latitude mirror pairing null
  - circular lon-shift pairing null
- stratified evaluation slices:
  - `all`
  - `events`
  - `near_storm_only`
  - `far_nonstorm`
