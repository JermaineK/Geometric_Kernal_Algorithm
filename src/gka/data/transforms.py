"""Data transform helpers used by adapters and ops."""

from __future__ import annotations

import numpy as np


def zscore(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=0)
    if std < eps:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def minmax(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = arr.min()
    hi = arr.max()
    scale = hi - lo
    if scale < eps:
        return np.zeros_like(arr)
    return (arr - lo) / scale


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")
