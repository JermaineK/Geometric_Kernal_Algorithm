# Weather Stress Suite

Domain-transfer stress checks for real-weather tile datasets.

These tests are designed to ensure GKA does not report coherent knees/parity
under generic geophysical structure that is not event-driven.

## Scenarios

- `diurnal_cycle_only_no_knee`
- `topography_mask_only`
- `latitudinal_gradient_only`
- `random_storm_times_real_fields`

## Run

```bash
python tests/weather_stress/run_weather_stress.py \
  --dataset data/tiles/weather_real_v2_all \
  --config examples/weather_real_minipilot/pipeline_config.yaml \
  --out tests/weather_stress/outputs
```

## Outputs

- `tests/weather_stress/outputs/suite_results.json`
- `tests/weather_stress/outputs/suite_summary.md`

