from __future__ import annotations

import argparse
from pathlib import Path
import sys

SYN_PATH = Path(__file__).resolve().parents[1] / "synthetic"
if str(SYN_PATH) not in sys.path:
    sys.path.insert(0, str(SYN_PATH))

from generate_synthetic import generate_dataset, load_config, write_generated_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one stress-case dataset")
    parser.add_argument("--config", required=True, help="Stress config YAML")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--run-index", type=int, default=0, help="Run index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    generated = generate_dataset(cfg, seed=int(args.seed), run_index=int(args.run_index))
    write_generated_dataset(args.outdir, generated)
    print(f"Wrote stress dataset to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
