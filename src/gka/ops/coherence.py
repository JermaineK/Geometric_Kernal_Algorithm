"""Coherence metrics for parity locking assessment."""

from __future__ import annotations

import numpy as np

from gka.core.types import CoherenceOutputs


def compute_coherence_metrics(
    X_plus: np.ndarray,
    X_minus: np.ndarray,
    signed_parity: np.ndarray | None = None,
    segment_labels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> CoherenceOutputs:
    xp = np.asarray(X_plus, dtype=float).reshape(-1)
    xm = np.asarray(X_minus, dtype=float).reshape(-1)
    if xp.size != xm.size:
        raise ValueError("X_plus and X_minus must have the same size")

    denom = np.linalg.norm(xp) * np.linalg.norm(xm) + eps
    A = float(np.dot(xp, xm) / denom)
    shear_num = np.std(np.diff(xm)) if xm.size > 1 else 0.0
    shear_den = np.std(xp) + eps
    V_shear = float(shear_num / shear_den)
    F = float(np.exp(-V_shear))
    if signed_parity is None:
        P_lock = float(np.clip(((A + 1.0) * 0.5) * F, 0.0, 1.0))
        return CoherenceOutputs(A=A, F=F, P_lock=P_lock)

    signed = np.asarray(signed_parity, dtype=float).reshape(-1)
    if signed.size == 0:
        return CoherenceOutputs(A=A, F=F, P_lock=0.0)
    signs = np.sign(signed)
    non_zero = signs[np.abs(signs) > 0]
    if non_zero.size == 0:
        return CoherenceOutputs(A=A, F=F, P_lock=0.0)

    base_lock = float(np.abs(np.mean(non_zero)))
    seg_stability = 1.0
    if segment_labels is not None:
        labels = np.asarray(segment_labels)
        if labels.size == non_zero.size:
            unique = np.unique(labels)
            if unique.size > 1:
                seg_means = np.array([np.mean(non_zero[labels == u]) for u in unique], dtype=float)
                seg_stability = float(np.clip(1.0 - np.std(seg_means), 0.0, 1.0))

    P_lock = float(np.clip(base_lock * seg_stability, 0.0, 1.0))
    return CoherenceOutputs(A=A, F=F, P_lock=P_lock)
