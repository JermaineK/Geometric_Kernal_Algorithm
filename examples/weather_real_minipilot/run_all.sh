#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-vortex_polar}"

if [[ "${MODE}" == "vortex_polar" ]]; then
  shift || true
  "${SCRIPT_DIR}/run_minipilot_vortex_polar.sh" "$@"
  exit 0
fi

if [[ "${MODE}" == "base" ]]; then
  shift || true
  "${SCRIPT_DIR}/run_minipilot.sh" "$@"
  exit 0
fi

echo "Unknown mode: ${MODE}"
echo "Usage: ${0} [vortex_polar|base] [args...]"
exit 2
