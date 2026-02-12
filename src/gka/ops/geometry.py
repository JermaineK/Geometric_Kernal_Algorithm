"""Geometry stage for curvature/torsion-like diagnostics."""

from __future__ import annotations

import numpy as np

from gka.core.types import GeometryOutputs


def compute_geometry(X: np.ndarray, eps: float = 1e-12) -> GeometryOutputs:
    arr = np.asarray(X, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
    if arr.size < 3:
        raise ValueError("compute_geometry requires at least 3 samples")

    d1 = np.gradient(arr)
    d2 = np.gradient(d1)
    kappa = np.abs(d2) / np.power(1.0 + d1 * d1, 1.5)
    tau = np.gradient(kappa)
    R = 1.0 / np.maximum(kappa, eps)
    stacked = np.vstack([kappa, tau, R])
    R_cov = np.cov(stacked)
    return GeometryOutputs(kappa=kappa, tau=tau, R=R, R_cov=R_cov)
