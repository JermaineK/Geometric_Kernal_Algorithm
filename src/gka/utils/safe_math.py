"""Safe math helpers for finite and stable calculations."""

from __future__ import annotations

import numpy as np


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log(np.clip(arr, eps, None))


def safe_div(num: np.ndarray | float, den: np.ndarray | float, eps: float = 1e-12):
    den_arr = np.asarray(den, dtype=float)
    return np.asarray(num, dtype=float) / np.where(np.abs(den_arr) < eps, eps, den_arr)


def finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else fallback
