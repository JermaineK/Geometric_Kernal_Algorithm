#!/usr/bin/env bash
set -euo pipefail
python examples/oam_images/prepare_oam_dataset.py --out examples/oam_images/dataset
gka validate examples/oam_images/dataset
gka run examples/oam_images/dataset --domain oam --config examples/oam_images/oam_config.yaml --out examples/oam_images/results --null 10
gka report examples/oam_images/results --out examples/oam_images/results/report.html
