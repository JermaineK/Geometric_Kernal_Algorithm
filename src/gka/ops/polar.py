from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gka.weather.polar import summarize_polar_features


def compute_polar_metrics(
    frame: pd.DataFrame,
    *,
    lat0: float,
    lon0: float,
    u_col: str = "u10",
    v_col: str = "v10",
    n_r: int = 64,
    n_theta: int = 128,
    r_max_km: float | None = None,
    pitch_values: list[float] | tuple[float, ...] = (-1.5, -1.0, -0.7, 0.7, 1.0, 1.5),
) -> dict[str, Any]:
    """Compute compact polar diagnostics for one tile/time frame."""

    return summarize_polar_features(
        frame,
        lat0=float(lat0),
        lon0=float(lon0),
        u_col=str(u_col),
        v_col=str(v_col),
        n_r=int(n_r),
        n_theta=int(n_theta),
        r_max_km=r_max_km,
        pitch_values=pitch_values,
    )


def compute_polar_parity_metrics(
    frame: pd.DataFrame,
    *,
    lat0: float,
    lon0: float,
    n_r: int = 64,
    n_theta: int = 128,
    r_max_km: float | None = None,
    pitch_values: list[float] | tuple[float, ...] = (-1.5, -1.0, -0.7, 0.7, 1.0, 1.5),
) -> dict[str, float]:
    """Compute mirror-odd parity metrics in polar space from original + mirrored winds."""

    orig = compute_polar_metrics(
        frame,
        lat0=lat0,
        lon0=lon0,
        u_col="u10",
        v_col="v10",
        n_r=n_r,
        n_theta=n_theta,
        r_max_km=r_max_km,
        pitch_values=pitch_values,
    )
    mir = compute_polar_metrics(
        frame,
        lat0=lat0,
        lon0=lon0,
        u_col="u10_mirror",
        v_col="v10_mirror",
        n_r=n_r,
        n_theta=n_theta,
        r_max_km=r_max_km,
        pitch_values=pitch_values,
    )

    left_o = float(orig.get("left_response", 0.0))
    right_o = float(orig.get("right_response", 0.0))
    left_m = float(mir.get("left_response", 0.0))
    right_m = float(mir.get("right_response", 0.0))

    odd_left = 0.5 * (left_o - left_m)
    odd_right = 0.5 * (right_o - right_m)
    even_left = 0.5 * (left_o + left_m)
    even_right = 0.5 * (right_o + right_m)

    odd_energy = float(odd_left * odd_left + odd_right * odd_right)
    even_energy = float(even_left * even_left + even_right * even_right)
    odd_ratio = float(odd_energy / (odd_energy + even_energy + 1e-12))

    chirality_o = float(orig.get("chirality", 0.0))
    chirality_m = float(mir.get("chirality", 0.0))
    eta = float((chirality_o - chirality_m) / (abs(chirality_o) + abs(chirality_m) + 1e-12))

    return {
        "polar_odd_energy_ratio": odd_ratio,
        "eta_parity_polar": eta,
        "chirality_orig": chirality_o,
        "chirality_mirror": chirality_m,
        "spiral_score_orig": float(orig.get("spiral_score", np.nan)),
        "spiral_score_mirror": float(mir.get("spiral_score", np.nan)),
    }
