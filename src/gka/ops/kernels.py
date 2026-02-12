"""Kernel utilities used by domain adapters and transforms."""

from __future__ import annotations

import numpy as np


def mirror_reflect(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.flip(np.asarray(arr), axis=axis)


def spiral_kernel(size: int = 7, handedness: str = "L") -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("spiral kernel size must be odd")
    yy, xx = np.mgrid[-size // 2 : size // 2 + 1, -size // 2 : size // 2 + 1]
    angle = np.arctan2(yy, xx)
    radius = np.sqrt(xx * xx + yy * yy)
    base = np.sin(angle + radius)
    if handedness.upper() == "R":
        base = np.fliplr(base)
    return base


def apply_kernel(field: np.ndarray, kernel: np.ndarray) -> float:
    f = np.asarray(field, dtype=float)
    k = np.asarray(kernel, dtype=float)
    if f.shape != k.shape:
        raise ValueError("Field and kernel must have equal shape")
    return float(np.sum(f * k))
