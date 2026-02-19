"""Implementation of `gka audit`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("audit", help="Audit decision rationale for a run directory")
    parser.add_argument("run_dir", help="Run output directory containing results.parquet")
    parser.add_argument("--json", action="store_true", help="Emit audit payload as JSON")
    parser.set_defaults(func=cmd_audit)


def cmd_audit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.parquet"
    metadata_path = run_dir / "run_metadata.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    df = pd.read_parquet(results_path)
    if df.empty:
        raise ValueError(f"Results file is empty: {results_path}")
    row = df.iloc[0]

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    payload = _build_audit_payload(row, metadata)
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Run dir: {run_dir}")
    print(f"Dataset: {payload.get('dataset_path')}")
    print(f"Domain: {payload.get('domain')}")
    print(f"Knee decision: {payload['knee']['decision']}")
    print(f"  confidence={payload['knee']['confidence']:.4f} delta_bic={payload['knee']['delta_bic']:.4f}")
    if payload["knee"].get("reasons"):
        print(f"  reasons={';'.join(payload['knee']['reasons'])}")
    print(f"Parity decision: {payload['parity']['decision']}")
    print(
        "  "
        f"p_perm={payload['parity']['p_perm']:.4g} "
        f"p_dir={payload['parity']['p_dir']:.4g} "
        f"P_lock={payload['parity']['P_lock']:.4g}"
    )
    print("Stability diagnostics:")
    print(
        "  "
        f"gamma={payload['stability']['gamma']:.6g} "
        f"Delta_b={payload['stability']['Delta_b']:.6g} "
        f"S_at_mu_k={payload['stability']['S_at_mu_k']:.6g} "
        f"band={payload['stability']['band_class']} "
        f"eigen={payload['stability']['eigen_band']} "
        f"margin={payload['stability']['stability_margin']:.6g}"
    )
    return 0


def _as_float(row: pd.Series, key: str, default: float = float("nan")) -> float:
    try:
        v = float(row.get(key, default))
        return v
    except Exception:
        return default


def _as_bool(row: pd.Series, key: str, default: bool = False) -> bool:
    v = row.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return bool(v)


def _build_audit_payload(row: pd.Series, metadata: dict[str, Any]) -> dict[str, Any]:
    rejected = str(row.get("knee_rejected_because", "") or "")
    reasons = [r for r in rejected.split(";") if r]
    knee_detected = _as_bool(row, "knee_detected", False)
    parity_pass = _as_bool(row, "parity_signal_pass", False)

    return {
        "dataset_path": metadata.get("dataset_path"),
        "domain": metadata.get("domain"),
        "timestamp_utc": metadata.get("timestamp_utc"),
        "knee": {
            "decision": "accepted" if knee_detected else "rejected",
            "confidence": _as_float(row, "knee_confidence", 0.0),
            "delta_bic": _as_float(row, "knee_delta_bic", float("nan")),
            "L_k": _as_float(row, "L_k", float("nan")),
            "reasons": reasons,
        },
        "parity": {
            "decision": "odd" if parity_pass else "null",
            "mirror_stat": _as_float(row, "parity_mirror_stat", float("nan")),
            "p_perm": _as_float(row, "parity_p_perm", 1.0),
            "p_dir": _as_float(row, "parity_p_dir", 1.0),
            "P_lock": _as_float(row, "P_lock", float("nan")),
        },
        "stability": {
            "gamma": _as_float(row, "gamma", float("nan")),
            "Delta_b": _as_float(row, "Delta_hat", float("nan")),
            "S_at_mu_k": _as_float(row, "S_at_mu_k", float("nan")),
            "W_mu": _as_float(row, "W_mu", float("nan")),
            "band_class": str(row.get("band_class_hat", "unknown")),
            "class": str(row.get("stability_class", "unknown")),
            "eigen_band": str(row.get("eigen_band", "unknown")),
            "stability_margin": _as_float(row, "stability_margin", float("nan")),
        },
    }
