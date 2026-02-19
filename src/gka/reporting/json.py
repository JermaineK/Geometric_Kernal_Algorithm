"""Structured report payload generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gka.utils.hash import file_sha256


def build_report_payload(results_dir: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    results_path = root / "results.parquet"
    metadata_path = root / "run_metadata.json"
    cfg_path = root / "config_resolved.yaml"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    df = pd.read_parquet(results_path)
    if df.empty:
        raise ValueError(f"Results are empty: {results_path}")
    row = df.iloc[0].to_dict()

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    payload = {
        "run_dir": str(root.resolve()),
        "artifacts": {
            "results_parquet": str(results_path.resolve()),
            "run_metadata_json": str(metadata_path.resolve()) if metadata_path.exists() else None,
            "config_resolved_yaml": str(cfg_path.resolve()) if cfg_path.exists() else None,
        },
        "reproducibility": {
            "dataset_hash": metadata.get("dataset_hash"),
            "config_hash": metadata.get("config_hash")
            or (file_sha256(cfg_path) if cfg_path.exists() else None),
            "git_commit": metadata.get("git_commit"),
            "rng_seed": metadata.get("random_seed"),
            "timestamp_utc": metadata.get("timestamp_utc"),
        },
        "summary": {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
        },
        "invariants": _coerce_scalars(row),
        "stage_context": metadata.get("stage_context", {}),
    }
    return payload


def write_report_json(payload: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_scalars(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (str, bool)) or value is None:
            out[key] = value
            continue
        try:
            out[key] = float(value)
        except Exception:
            out[key] = str(value)
    return out
