"""Implementation of `gka diagnose`."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from gka.core.pipeline import run_pipeline


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("diagnose", help="Run diagnostics and emit compact summary JSON")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--domain", default=None, help="Domain adapter (optional if dataset manifest has domain)")
    parser.add_argument("--config", default=None, help="Optional run config YAML")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    parser.set_defaults(func=cmd_diagnose)


def cmd_diagnose(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="gka_diagnose_") as tmpdir:
        result = run_pipeline(
            dataset_path=args.data,
            domain=args.domain,
            out_dir=tmpdir,
            config_path=args.config,
            null_n=0,
            allow_missing=False,
            seed=42,
            argv=["gka", "diagnose", "--data", args.data],
        )

        df = result.summary
        if df.empty:
            raise RuntimeError("Pipeline produced empty results; cannot diagnose")
        row = df.iloc[0]

        payload = {
            "gamma_hat": _safe_float(row.get("gamma")),
            "Delta_b": _safe_float(row.get("Delta_hat")),
            "omega_k_hat": _safe_float(row.get("omega_k_hat")),
            "tau_s_hat": _safe_float(row.get("tau_s_hat")),
            "S_at_mu_k": _safe_float(row.get("S_at_mu_k")),
            "W_mu": _safe_float(row.get("W_mu")),
            "R_align": _safe_float(row.get("R_align")),
            "band_class_hat": row.get("band_class_hat"),
            "knee_detected": bool(row.get("knee_detected", False)),
            "knee_rejected_because": row.get("knee_rejected_because"),
        }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Diagnosis written to {out_path}")
    print(text)
    return 0


def _safe_float(v):
    try:
        if v is None:
            return None
        fv = float(v)
    except Exception:
        return None
    if pd.isna(fv):
        return None
    return fv
