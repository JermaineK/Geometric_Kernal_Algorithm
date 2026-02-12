"""Implementation of `gka prepare`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gka.data.io import prepare_dataset_layout, write_dataset_spec, write_samples


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare", help="Prepare canonical dataset layout")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--in", dest="input_path", required=True, help="Raw input CSV/Parquet path")
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("--id", dest="dataset_id", default=None, help="Dataset id override")
    parser.set_defaults(func=cmd_prepare)


def cmd_prepare(args: argparse.Namespace) -> int:
    in_path = Path(args.input_path)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    samples = _load_input_table(in_path)
    _check_required_columns(samples)

    prepare_dataset_layout(out_path)
    write_samples(samples, out_path)

    dataset_id = args.dataset_id or out_path.name
    spec = {
        "schema_version": 1,
        "domain": args.domain,
        "id": dataset_id,
        "description": f"Prepared from {in_path}",
        "units": {"time": "index", "L": "arb", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"] if "O" in samples.columns else ["O_path"],
        },
        "analysis": {
            "knee": {"method": "segmented", "rho": 1.5},
            "scaling": {"method": "wls", "min_points": 4, "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.1},
        },
    }
    write_dataset_spec(spec, out_path)

    print(f"Prepared canonical dataset at {out_path}")
    print("Generated files:")
    print(f"- {out_path / 'dataset.yaml'}")
    print(f"- {out_path / 'samples.parquet'}")
    return 0


def _load_input_table(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.is_file() and path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.is_dir():
        parquet = path / "samples.parquet"
        csv = path / "samples.csv"
        if parquet.exists():
            return pd.read_parquet(parquet)
        if csv.exists():
            return pd.read_csv(csv)
        raise ValueError(
            f"Input directory {path} must contain samples.parquet or samples.csv for prepare"
        )

    raise ValueError("Unsupported input format. Use CSV or Parquet")


def _check_required_columns(df: pd.DataFrame) -> None:
    required = {"case_id", "t", "L", "hand"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(
            "Input table missing required columns "
            f"{missing}. Required: case_id,t,L,hand plus O or O_path"
        )
    if "O" not in df.columns and "O_path" not in df.columns:
        raise ValueError("Input table must contain O or O_path column")
