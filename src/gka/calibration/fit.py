"""Calibration fitting entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gka.calibrate.fit_thresholds import fit_thresholds_from_robustness
from gka.calibration.schema import (
    CALIBRATION_SCHEMA_VERSION,
    validate_calibration_payload,
)
from gka.utils.hash import file_sha256


def fit_calibration_from_parameter_runs(
    parameter_runs_path: str | Path,
    target_fp_max: float,
    objective_beta: float,
) -> dict[str, Any]:
    path = Path(parameter_runs_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter runs file not found: {path}")
    df = pd.read_json(path)
    thresholds = fit_thresholds_from_robustness(
        parameter_runs=df,
        target_fp_max=float(target_fp_max),
        objective_beta=float(objective_beta),
    )
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "parameter_runs_path": str(path.resolve()),
            "parameter_runs_sha256": file_sha256(path),
            "n_rows": int(df.shape[0]),
        },
        "objective": {
            "false_positive_rate_max": float(target_fp_max),
            "beta": float(objective_beta),
        },
        "thresholds": thresholds,
    }
    return validate_calibration_payload(payload)


def write_calibration(payload: dict[str, Any], out_path: str | Path) -> None:
    import json

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_calibration_payload(payload)
    out.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
