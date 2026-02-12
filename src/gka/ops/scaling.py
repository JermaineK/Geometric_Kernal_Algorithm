"""Scaling fit stage."""

from __future__ import annotations

import numpy as np

from gka.core.types import ScalingOutputs
from gka.stats.bootstrap import bootstrap_slope_ci
from gka.stats.regression import fit_theil_sen, fit_wls
from gka.utils.safe_math import safe_log


def fit_scaling(
    eta: np.ndarray,
    L: np.ndarray,
    exclude_band: tuple[float, float] | None,
    weights: np.ndarray | None,
    method: str,
    min_sizes: int = 4,
    bootstrap_n: int = 1000,
    rng: np.random.Generator | None = None,
) -> ScalingOutputs:
    eta_arr = np.asarray(eta, dtype=float)
    L_arr = np.asarray(L, dtype=float)

    if eta_arr.size != L_arr.size:
        raise ValueError("eta and L must have the same length")

    mask = np.ones_like(L_arr, dtype=bool)
    if exclude_band is not None:
        lo, hi = exclude_band
        mask &= ~((L_arr >= lo) & (L_arr <= hi))

    x = safe_log(L_arr[mask])
    y = safe_log(np.abs(eta_arr[mask]) + 1e-12)

    if x.size < min_sizes:
        raise ValueError(
            f"Not enough points for scaling fit after exclusions ({x.size} < {min_sizes})"
        )

    if method == "wls":
        w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)[mask]
        slope, intercept, residuals = fit_wls(x, y, w)
    elif method == "theil_sen":
        slope, intercept, residuals = fit_theil_sen(x, y)
    else:
        raise ValueError(f"Unsupported scaling method '{method}'. Use wls|theil_sen")

    ci_lo, ci_hi = bootstrap_slope_ci(x, y, method=method, n=bootstrap_n, rng=rng)
    drift = _drift_metric(x, residuals)
    gamma = float(slope)
    return ScalingOutputs(
        gamma=gamma,
        Delta_hat=float(2.0 - gamma),
        ci=(float(ci_lo), float(ci_hi)),
        drift=float(drift),
        n_points=int(x.size),
    )


def _drift_metric(x: np.ndarray, residuals: np.ndarray) -> float:
    if x.size < 4:
        return float(np.mean(np.abs(residuals)))
    split = x.size // 2
    left = residuals[:split]
    right = residuals[split:]
    return float(np.abs(left.mean() - right.mean()))
