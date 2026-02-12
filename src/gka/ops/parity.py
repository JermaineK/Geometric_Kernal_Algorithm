"""Parity decomposition operators."""

from __future__ import annotations

import numpy as np

from gka.core.types import MirrorSpec, ParityOutputs


def parity_split(X: np.ndarray, mirror_spec: MirrorSpec, eps: float = 1e-12) -> ParityOutputs:
    arr = np.asarray(X, dtype=float)
    pair_index = np.asarray(mirror_spec.pair_index, dtype=int)

    if arr.shape[0] != pair_index.shape[0]:
        raise ValueError("Mirror mapping length must match X length")

    idx = np.arange(arr.shape[0], dtype=int)
    if not np.array_equal(pair_index[pair_index], idx):
        raise ValueError("Mirror mapping is not involutive: expected M(M(i)) == i")

    partner = arr[pair_index]
    X_plus = 0.5 * (arr + partner)
    X_minus = 0.5 * (arr - partner)

    E_plus = float(np.mean(np.square(X_plus)))
    E_minus = float(np.mean(np.square(X_minus)))
    eta = np.abs(X_minus) / (np.abs(X_plus) + eps)
    return ParityOutputs(X_plus=X_plus, X_minus=X_minus, E_plus=E_plus, E_minus=E_minus, eta=eta)
