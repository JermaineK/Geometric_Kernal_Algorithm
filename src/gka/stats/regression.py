"""Regression helpers used by scaling diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes


def fit_wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    if x_arr.size != y_arr.size or x_arr.size != w_arr.size:
        raise ValueError("x, y, and w must have equal length")
    if np.any(w_arr <= 0):
        raise ValueError("All weights must be positive")

    X = np.vstack([x_arr, np.ones_like(x_arr)]).T
    W = np.diag(w_arr)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_arr)
    slope, intercept = float(beta[0]), float(beta[1])
    residuals = y_arr - (slope * x_arr + intercept)
    return slope, intercept, residuals


def fit_theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    slope, intercept, _, _ = theilslopes(y_arr, x_arr)
    residuals = y_arr - (slope * x_arr + intercept)
    return float(slope), float(intercept), residuals
