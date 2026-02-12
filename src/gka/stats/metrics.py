"""General metric helpers."""

from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("Inputs must have the same shape")
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError("Inputs must have equal length")
    if aa.size < 2:
        return 0.0
    c = np.corrcoef(aa, bb)[0, 1]
    return float(0.0 if np.isnan(c) else c)
