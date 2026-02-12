#!/usr/bin/env bash
set -euo pipefail
python examples/em_resonator/prepare_em_dataset.py --out examples/em_resonator/dataset
gka validate examples/em_resonator/dataset
gka run examples/em_resonator/dataset --domain em_resonator --config examples/em_resonator/em_config.yaml --out examples/em_resonator/results --null 15
gka report examples/em_resonator/results --out examples/em_resonator/results/report.html
