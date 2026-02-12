#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: export_results_bundle.sh <results_dir> <bundle_path.tar.gz>"
  exit 1
fi

RESULTS_DIR="$1"
OUT_FILE="$2"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Results directory not found: $RESULTS_DIR"
  exit 1
fi

tar -czf "$OUT_FILE" -C "$RESULTS_DIR" .
echo "Exported bundle: $OUT_FILE"
