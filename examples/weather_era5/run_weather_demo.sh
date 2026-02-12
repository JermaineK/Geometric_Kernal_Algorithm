#!/usr/bin/env bash
set -euo pipefail
python examples/weather_era5/prepare_weather_dataset.py --out examples/weather_era5/dataset
gka validate examples/weather_era5/dataset
gka run examples/weather_era5/dataset --domain weather --config examples/weather_era5/weather_config.yaml --out examples/weather_era5/results --null 20
gka report examples/weather_era5/results --out examples/weather_era5/results/report.html
