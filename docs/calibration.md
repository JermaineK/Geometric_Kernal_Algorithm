# Calibration vs Scoring

GKA now treats threshold tuning and scoring as separate stages.

## Why this split exists

- `calibration` learns thresholds from calibration data.
- `scoring` applies frozen thresholds to new data.

This prevents leakage where thresholds are changed while claiming blind performance.

## Calibration flow

1. Generate or collect `parameter_runs.json`.
2. Fit thresholds:
   - `gka calibrate --parameter-runs tests/stress/outputs/robustness/parameter_sweep/parameter_runs.json --calibration-out tests/stress/outputs/calibration.json`
3. Archive `calibration.json` with run artifacts.

`calibration.json` includes:

- schema version
- generation timestamp
- source file path + hash
- objective settings (`false_positive_rate_max`, `beta`)
- frozen thresholds

## Scoring flow

1. Run scoring with no threshold edits:
   - `gka score --parameter-runs tests/stress/outputs/robustness/parameter_sweep/parameter_runs.json --calibration tests/stress/outputs/calibration.json --out tests/stress/outputs/score_report.json`
2. Review reported confusion and rates.

## Guardrail

`gka calibrate` refuses to run calibrate+score in one invocation unless
`--allow-calibrate-and-score` is explicitly passed.
