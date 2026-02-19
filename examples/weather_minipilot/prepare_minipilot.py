from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a tiny weather minipilot dataset")
    parser.add_argument("--input", required=True, help="Path to source parquet table")
    parser.add_argument("--out", required=True, help="Output canonical dataset directory")
    parser.add_argument("--max-pairs", type=int, default=600, help="Maximum paired rows to keep")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed")
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    src = Path(ns.input)
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    col_case = _pick(df, ["case_id", "storm_id", "id", "track_id", "group_id"])
    col_time = _pick(df, ["t", "time", "timestamp", "valid_time", "datetime"])
    col_size = _pick(df, ["L", "scale", "radius_km", "patch_size", "size"])
    col_hand = _pick(df, ["hand", "handedness", "chirality", "lr"])
    col_obs = _pick_observable(df, excluded={col_case, col_time, col_size, col_hand})

    mapped = pd.DataFrame(
        {
            "case_id": df[col_case],
            "t": df[col_time],
            "L": pd.to_numeric(df[col_size], errors="coerce"),
            "hand": _normalize_hand(df[col_hand]),
            "O": pd.to_numeric(df[col_obs], errors="coerce"),
        }
    )
    if "omega" in df.columns:
        mapped["omega"] = pd.to_numeric(df["omega"], errors="coerce")

    mapped = mapped.dropna(subset=["case_id", "t", "L", "hand", "O"])
    mapped = mapped[mapped["hand"].isin(["L", "R"])].copy()
    if mapped.empty:
        raise ValueError("No usable rows after cleaning and hand normalization")

    key_cols = ["case_id", "t", "L"] + (["omega"] if "omega" in mapped.columns else [])
    counts = mapped.groupby(key_cols, dropna=False)["hand"].nunique()
    good_keys = counts[counts == 2].index
    pairs = mapped.set_index(key_cols).loc[good_keys].reset_index()
    if pairs.empty:
        raise ValueError("No complete L/R pairs found in source table")

    rng = np.random.default_rng(int(ns.seed))
    unique_keys = pairs[key_cols].drop_duplicates().reset_index(drop=True)
    if unique_keys.shape[0] > int(ns.max_pairs):
        pick_idx = np.sort(rng.choice(unique_keys.index.to_numpy(), size=int(ns.max_pairs), replace=False))
        keep = unique_keys.loc[pick_idx]
        pairs = pairs.merge(keep, on=key_cols, how="inner")

    pairs = pairs.sort_values(key_cols + ["hand"]).reset_index(drop=True)
    pairs.to_parquet(out_dir / "samples.parquet", index=False)

    dataset_yaml = {
        "schema_version": 1,
        "domain": "weather",
        "id": "weather_minipilot_v1",
        "description": "Tiny weather pilot extracted from source parquet",
        "units": {"time": "index", "L": "arbitrary", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"source_hand_column": str(col_hand)}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "analysis": {
            "knee": {"method": "segmented", "rho": 1.5},
            "scaling": {"method": "wls", "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.15},
        },
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    print(f"Prepared minipilot dataset: {out_dir}")
    print(f"Rows: {pairs.shape[0]}  paired keys: {pairs[key_cols].drop_duplicates().shape[0]}")
    print(
        f"Mapping: case={col_case} time={col_time} size={col_size} hand={col_hand} observable={col_obs}"
    )
    return 0


def _pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find any of required columns: {candidates}")


def _pick_observable(df: pd.DataFrame, excluded: set[str]) -> str:
    if "O" in df.columns:
        return "O"
    numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric observable column found")
    preferred = ["eta", "odd_response", "vorticity", "signal", "value"]
    for col in preferred:
        if col in numeric:
            return col
    return numeric[0]


def _normalize_hand(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "l": "L",
        "left": "L",
        "-1": "L",
        "0": "L",
        "r": "R",
        "right": "R",
        "1": "R",
    }
    return s.map(mapping)


if __name__ == "__main__":
    raise SystemExit(main())
