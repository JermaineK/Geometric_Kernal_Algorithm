#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python examples/weather_era5/prepare_weather_dataset.py --out .tmp/weather_dataset
gka validate .tmp/weather_dataset
gka run .tmp/weather_dataset --domain weather --out .tmp/weather_results --null 5
gka report .tmp/weather_results --out .tmp/weather_results/report.html

echo "Smoke test complete"
