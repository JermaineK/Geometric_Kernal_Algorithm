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


def fit_huber(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    delta: float = 1.5,
    max_iter: int = 32,
    tol: float = 1e-7,
) -> tuple[float, float, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have equal length")
    if x_arr.size < 2:
        raise ValueError("fit_huber requires at least 2 points")

    if w is None:
        base_w = np.ones_like(x_arr, dtype=float)
    else:
        base_w = np.asarray(w, dtype=float)
        if base_w.size != x_arr.size:
            raise ValueError("weights must match x/y length")
        if np.any(base_w <= 0):
            raise ValueError("All weights must be positive")

    X = np.vstack([x_arr, np.ones_like(x_arr)]).T
    beta = _solve_weighted_ls(X, y_arr, base_w)

    for _ in range(max_iter):
        resid = y_arr - (X @ beta)
        scale = float(np.median(np.abs(resid)))
        scale = max(scale, 1e-9)
        u = np.abs(resid) / (delta * scale)
        huber_w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        w_eff = base_w * huber_w
        new_beta = _solve_weighted_ls(X, y_arr, w_eff)
        if float(np.max(np.abs(new_beta - beta))) <= tol:
            beta = new_beta
            break
        beta = new_beta

    slope, intercept = float(beta[0]), float(beta[1])
    residuals = y_arr - (slope * x_arr + intercept)
    return slope, intercept, residuals


def _solve_weighted_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sqrt_w = np.sqrt(np.asarray(w, dtype=float))
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return np.asarray(beta, dtype=float)
