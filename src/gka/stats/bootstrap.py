"""Bootstrap routines."""

from __future__ import annotations

import numpy as np

from gka.data.resample import resample_blocks
from gka.stats.regression import fit_theil_sen, fit_wls


def bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "wls",
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have equal length")
    if x_arr.size < 3:
        raise ValueError("bootstrap_slope_ci requires at least 3 points")

    rng = rng or np.random.default_rng(0)
    slopes = np.empty(n, dtype=float)

    for i in range(n):
        idx = rng.integers(0, x_arr.size, size=x_arr.size)
        xb = x_arr[idx]
        yb = y_arr[idx]
        if method == "wls":
            slope, _, _ = fit_wls(xb, yb, np.ones_like(xb))
        elif method == "theil_sen":
            slope, _, _ = fit_theil_sen(xb, yb)
        else:
            raise ValueError(f"Unsupported method '{method}' for bootstrap")
        slopes[i] = slope

    lo = float(np.quantile(slopes, alpha / 2.0))
    hi = float(np.quantile(slopes, 1.0 - alpha / 2.0))
    return lo, hi


def block_bootstrap_series(
    values: np.ndarray,
    n: int,
    block_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    rng = rng or np.random.default_rng(0)
    out = np.empty((n, arr.size), dtype=float)
    for i in range(n):
        out[i] = resample_blocks(arr, block_size=block_size, rng=rng)
    return out
