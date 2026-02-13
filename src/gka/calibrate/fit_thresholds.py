"""Fit diagnostic thresholds from robustness outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def fit_thresholds_from_robustness(
    parameter_runs: pd.DataFrame,
    target_fp_max: float = 0.10,
    objective_beta: float = 1.0,
) -> dict[str, Any]:
    df = parameter_runs.copy()
    for col in ("knee_p", "knee_strength", "knee_delta_bic", "middle_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y_true = df["has_knee_true"].astype(bool).to_numpy()

    p_vals = _grid(df.get("knee_p"), default=np.array([0.25, 0.35, 0.45, 0.55]))
    s_vals = _grid(df.get("knee_strength"), default=np.array([0.0, 0.5, 1.0, 1.5]))
    b_vals = _grid(df.get("knee_delta_bic"), default=np.array([4.0, 6.0, 8.0, 10.0]))
    m_vals = _grid(df.get("middle_score"), default=np.array([0.45, 0.55, 0.65, 0.75]))

    best: dict[str, Any] | None = None
    for p_min in p_vals:
        for s_min in s_vals:
            for b_min in b_vals:
                for m_thr in m_vals:
                    pred_pos = (
                        ((df.get("knee_p", 0.0) >= p_min) & (df.get("knee_strength", 0.0) >= s_min) & (df.get("knee_delta_bic", 0.0) >= b_min))
                        | (df.get("middle_score", 0.0) >= m_thr)
                    ).fillna(False).to_numpy()

                    fp = _rate(pred_pos & ~y_true, ~y_true)
                    fn = _rate((~pred_pos) & y_true, y_true)
                    prec = _precision(pred_pos, y_true)
                    rec = _recall(pred_pos, y_true)
                    fbeta = _f_beta(prec, rec, beta=objective_beta)
                    if fp > target_fp_max:
                        continue
                    candidate = {
                        "knee_p_min": float(p_min),
                        "knee_strength_min": float(s_min),
                        "knee_delta_bic_min": float(b_min),
                        "middle_score_band": {"forbidden_min": float(m_thr)},
                        "metrics": {
                            "false_positive_rate": fp,
                            "false_negative_rate": fn,
                            "precision": prec,
                            "recall": rec,
                            "f_beta": fbeta,
                        },
                    }
                    if best is None or candidate["metrics"]["f_beta"] > best["metrics"]["f_beta"]:
                        best = candidate

    if best is None:
        best = {
            "knee_p_min": 0.45,
            "knee_strength_min": 0.5,
            "knee_delta_bic_min": 8.0,
            "middle_score_band": {"forbidden_min": 0.6},
            "metrics": {
                "false_positive_rate": float("nan"),
                "false_negative_rate": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f_beta": float("nan"),
            },
        }
    return best


def write_threshold_yaml(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit knee thresholds from robustness parameter runs")
    parser.add_argument("--parameter-runs", required=True, help="Path to parameter_runs.json")
    parser.add_argument("--out", required=True, help="Output YAML path")
    parser.add_argument("--fp-max", type=float, default=0.10, help="Maximum allowed false positive rate")
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta objective weight")
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    runs = pd.read_json(ns.parameter_runs)
    best = fit_thresholds_from_robustness(
        parameter_runs=runs,
        target_fp_max=float(ns.fp_max),
        objective_beta=float(ns.beta),
    )
    payload = {
        "generated_from": str(Path(ns.parameter_runs)),
        "objective": {"beta": float(ns.beta), "fp_max": float(ns.fp_max)},
        "thresholds": best,
    }
    write_threshold_yaml(payload, Path(ns.out))
    print(f"Wrote calibrated thresholds to {ns.out}")
    return 0


def _grid(series: pd.Series | np.ndarray | None, default: np.ndarray) -> np.ndarray:
    if series is None:
        return default
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return default
    qs = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    out = np.unique(np.concatenate([default, qs]))
    return np.sort(out)


def _rate(num_mask: np.ndarray, den_mask: np.ndarray) -> float:
    den = int(np.sum(den_mask))
    if den <= 0:
        return 0.0
    return float(np.sum(num_mask) / den)


def _precision(pred_pos: np.ndarray, y_true: np.ndarray) -> float:
    tp = int(np.sum(pred_pos & y_true))
    pp = int(np.sum(pred_pos))
    if pp <= 0:
        return 0.0
    return float(tp / pp)


def _recall(pred_pos: np.ndarray, y_true: np.ndarray) -> float:
    tp = int(np.sum(pred_pos & y_true))
    p = int(np.sum(y_true))
    if p <= 0:
        return 0.0
    return float(tp / p)


def _f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 1e-12:
        return 0.0
    return float((1.0 + b2) * precision * recall / denom)


if __name__ == "__main__":
    raise SystemExit(main())
