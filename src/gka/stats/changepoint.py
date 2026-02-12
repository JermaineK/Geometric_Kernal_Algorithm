"""Changepoint helpers."""

from __future__ import annotations

import numpy as np


def segmented_split_index(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have equal length")
    if x_arr.size < 5:
        raise ValueError("segmented split requires at least 5 points")

    total_sse = _fit_sse(x_arr, y_arr)
    best_idx = 2
    best_sse = np.inf

    for idx in range(2, x_arr.size - 2):
        sse_left = _fit_sse(x_arr[:idx], y_arr[:idx])
        sse_right = _fit_sse(x_arr[idx:], y_arr[idx:])
        sse = sse_left + sse_right
        if sse < best_sse:
            best_sse = sse
            best_idx = idx

    quality = float(1.0 - best_sse / np.maximum(total_sse, 1e-12))
    return best_idx, max(0.0, min(quality, 1.0))


def _fit_sse(x: np.ndarray, y: np.ndarray) -> float:
    coeffs = np.polyfit(x, y, deg=1)
    pred = np.polyval(coeffs, x)
    return float(np.sum((y - pred) ** 2))
