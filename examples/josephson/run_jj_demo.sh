#!/usr/bin/env bash
set -euo pipefail
python examples/josephson/prepare_jj_dataset.py --out examples/josephson/dataset
gka validate examples/josephson/dataset
gka run examples/josephson/dataset --domain josephson --config examples/josephson/jj_config.yaml --out examples/josephson/results --null 20
gka report examples/josephson/results --out examples/josephson/results/report.html
