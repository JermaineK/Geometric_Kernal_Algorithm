#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python examples/weather_era5/prepare_weather_dataset.py --out .tmp/repro_dataset
gka run .tmp/repro_dataset --domain weather --out .tmp/repro_run_a --seed 123 --null 0
gka run .tmp/repro_dataset --domain weather --out .tmp/repro_run_b --seed 123 --null 0

HASH_A="$(python - <<'PY'
from gka.utils.hash import file_sha256
print(file_sha256('.tmp/repro_run_a/results.parquet'))
PY
)"
HASH_B="$(python - <<'PY'
from gka.utils.hash import file_sha256
print(file_sha256('.tmp/repro_run_b/results.parquet'))
PY
)"

if [[ "$HASH_A" != "$HASH_B" ]]; then
  echo "Repro audit failed: results hashes differ"
  exit 1
fi

echo "Repro audit passed"
