"""Impedance alignment checks."""

from __future__ import annotations

import numpy as np

from gka.core.types import ImpedanceOutputs


def impedance_alignment(
    omega_k: np.ndarray | float | None,
    L: np.ndarray | float | None,
    cm_or_v: np.ndarray | float | None,
    a: np.ndarray | float | None,
    tolerance: float = 0.1,
) -> ImpedanceOutputs:
    if omega_k is None or L is None or cm_or_v is None:
        return ImpedanceOutputs(ratio=None, passed=None, tolerance=tolerance)

    omega_arr = np.asarray(omega_k, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    c_arr = np.asarray(cm_or_v, dtype=float)

    if np.any(np.abs(c_arr) < 1e-12):
        raise ValueError("cm_or_v is zero; cannot compute impedance alignment ratio")

    ratio_series = (omega_arr * L_arr) / (2.0 * np.pi * c_arr)
    if a is not None:
        a_arr = np.asarray(a, dtype=float)
        ratio_series = ratio_series * np.where(np.abs(L_arr) > 1e-12, a_arr / L_arr, 1.0)

    ratio = float(np.nanmedian(ratio_series))
    passed = bool(np.abs(ratio - 1.0) <= tolerance)
    return ImpedanceOutputs(ratio=ratio, passed=passed, tolerance=tolerance)


def _representative_value(v: np.ndarray | float) -> float:
    arr = np.asarray(v, dtype=float)
    return float(np.nanmedian(arr))
