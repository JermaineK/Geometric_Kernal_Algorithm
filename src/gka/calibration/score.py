"""Scoring with frozen calibration thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gka.calibration.schema import validate_calibration_payload
from gka.utils.hash import mapping_sha256


def score_parameter_runs(
    parameter_runs_path: str | Path,
    calibration_path: str | Path,
) -> dict[str, Any]:
    runs_path = Path(parameter_runs_path)
    cal_path = Path(calibration_path)
    if not runs_path.exists():
        raise FileNotFoundError(f"Parameter runs file not found: {runs_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")

    runs_df = pd.read_json(runs_path)
    calibration = validate_calibration_payload(json.loads(cal_path.read_text(encoding="utf-8")))

    thresholds = calibration["thresholds"]
    forbidden_min = float(thresholds["middle_score_band"]["forbidden_min"])

    pred = (
        (
            (pd.to_numeric(runs_df.get("knee_p"), errors="coerce") >= float(thresholds["knee_p_min"]))
            & (
                pd.to_numeric(runs_df.get("knee_strength"), errors="coerce")
                >= float(thresholds["knee_strength_min"])
            )
            & (
                pd.to_numeric(runs_df.get("knee_delta_bic"), errors="coerce")
                >= float(thresholds["knee_delta_bic_min"])
            )
        )
        | (pd.to_numeric(runs_df.get("middle_score"), errors="coerce") >= forbidden_min)
    ).fillna(False)

    y_true = runs_df["has_knee_true"].astype(bool)
    tp = int(np.sum(pred & y_true))
    tn = int(np.sum((~pred) & (~y_true)))
    fp = int(np.sum(pred & (~y_true)))
    fn = int(np.sum((~pred) & y_true))

    p = tp + fp
    r = tp + fn
    n_neg = fp + tn
    precision = float(tp / p) if p else 0.0
    recall = float(tp / r) if r else 0.0
    fp_rate = float(fp / n_neg) if n_neg else 0.0
    fn_rate = float(fn / r) if r else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    return {
        "schema_version": 1,
        "calibration_path": str(cal_path.resolve()),
        "parameter_runs_path": str(runs_path.resolve()),
        "calibration_hash": mapping_sha256(calibration),
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_rows": int(runs_df.shape[0])},
        "metrics": {
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def write_score_report(payload: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
