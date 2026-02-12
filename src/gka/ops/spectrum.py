"""Spectral separation and ridge extraction."""

from __future__ import annotations

import numpy as np
from scipy import signal

from gka.core.types import TickOutputs


def extract_ticks(
    X: np.ndarray,
    method: str = "welch",
    fs: float = 1.0,
    top_k: int = 3,
) -> TickOutputs:
    arr = np.asarray(X, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
    if arr.size < 8:
        return TickOutputs(
            Omega_candidates=np.array([], dtype=float),
            omega_band=None,
            R_Omega=0.0,
            ridge_strength=0.0,
        )

    if method == "welch":
        freq, power = signal.welch(arr, fs=fs, nperseg=min(256, arr.size))
        return _summarize_spectrum(freq, power, top_k)

    if method == "cwt":
        widths = np.arange(1, min(64, max(4, arr.size // 3)))
        cwt = signal.cwt(arr, signal.ricker, widths)
        power = np.abs(cwt).mean(axis=1)
        freq = fs / widths
        return _summarize_spectrum(freq, power, top_k)

    raise ValueError(f"Unsupported ticks method '{method}'. Use welch|cwt")


def _summarize_spectrum(freq: np.ndarray, power: np.ndarray, top_k: int) -> TickOutputs:
    if power.size == 0:
        return TickOutputs(
            Omega_candidates=np.array([], dtype=float),
            omega_band=None,
            R_Omega=0.0,
            ridge_strength=0.0,
        )

    mask = freq > 0
    freq = freq[mask]
    power = power[mask]
    if power.size == 0:
        return TickOutputs(
            Omega_candidates=np.array([], dtype=float),
            omega_band=None,
            R_Omega=0.0,
            ridge_strength=0.0,
        )

    idx = np.argsort(power)[::-1]
    top_idx = idx[: min(top_k, idx.size)]
    omega_candidates = 2.0 * np.pi * freq[top_idx]
    strongest = int(top_idx[0])
    omega0 = 2.0 * np.pi * freq[strongest]
    band = (0.9 * omega0, 1.1 * omega0)
    r_omega = float(power[strongest] / np.maximum(power.sum(), 1e-12))
    ridge = float(power[strongest] / np.maximum(np.median(power), 1e-12))
    return TickOutputs(
        Omega_candidates=np.sort(omega_candidates),
        omega_band=band,
        R_Omega=r_omega,
        ridge_strength=ridge,
    )
