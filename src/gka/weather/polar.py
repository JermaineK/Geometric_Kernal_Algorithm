from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def summarize_polar_features(
    frame: pd.DataFrame,
    *,
    lat0: float,
    lon0: float,
    u_col: str = "u10",
    v_col: str = "v10",
    n_r: int = 64,
    n_theta: int = 128,
    r_max_km: float | None = None,
    pitch_values: Iterable[float] = (-1.5, -1.0, -0.7, 0.7, 1.0, 1.5),
    mode_values: Iterable[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Compute compact storm-frame polar diagnostics for one tile/time slice."""

    if frame.empty:
        return _empty_polar_summary()
    if u_col not in frame.columns or v_col not in frame.columns:
        return _empty_polar_summary()
    if "lat" not in frame.columns or "lon" not in frame.columns:
        return _empty_polar_summary()

    lat_vals = np.sort(pd.to_numeric(frame["lat"], errors="coerce").dropna().unique().astype(float))
    lon_vals = np.sort(pd.to_numeric(frame["lon"], errors="coerce").dropna().unique().astype(float))
    if lat_vals.size < 3 or lon_vals.size < 3:
        return _empty_polar_summary()

    u_grid = _grid_from_frame(frame, "lat", "lon", u_col, lat_vals, lon_vals)
    v_grid = _grid_from_frame(frame, "lat", "lon", v_col, lat_vals, lon_vals)
    if u_grid is None or v_grid is None:
        return _empty_polar_summary()

    x_axis, y_axis = _xy_axes_km(lat_vals, lon_vals, lat0=float(lat0), lon0=float(lon0))
    if r_max_km is None:
        r_max_km = float(min(np.nanmax(np.abs(x_axis)), np.nanmax(np.abs(y_axis))))
    r_max_km = float(max(r_max_km, 20.0))

    r = np.linspace(0.0, r_max_km, int(max(n_r, 16)))
    theta = np.linspace(0.0, 2.0 * np.pi, int(max(n_theta, 32)), endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    u_pol = _interp_regular_grid(y_axis, x_axis, u_grid, yy, xx)
    v_pol = _interp_regular_grid(y_axis, x_axis, v_grid, yy, xx)

    # Velocity decomposition in local polar basis.
    v_r = u_pol * np.cos(tt) + v_pol * np.sin(tt)
    v_t = -u_pol * np.sin(tt) + v_pol * np.cos(tt)
    vt_abs = np.abs(v_t)

    mode_power: dict[int, float] = {}
    mode_phase_consistency: dict[int, float] = {}
    for m in mode_values:
        m_int = int(m)
        amp_r = np.nanmean(v_t * np.exp(-1j * m_int * tt), axis=1)
        power = np.nanmean(np.abs(amp_r))
        mode_power[m_int] = float(power) if np.isfinite(power) else 0.0
        valid = np.abs(amp_r) > 1e-12
        if np.any(valid):
            ph = amp_r[valid] / np.abs(amp_r[valid])
            mode_phase_consistency[m_int] = float(np.abs(np.nanmean(ph)))
        else:
            mode_phase_consistency[m_int] = 0.0

    if mode_power:
        dominant_m = int(max(mode_power, key=lambda k: mode_power[k]))
    else:
        dominant_m = 1

    pitch_scores: dict[float, float] = {}
    for b in pitch_values:
        b_val = float(b)
        if abs(b_val) < 1e-6:
            continue
        pitch_scores[b_val] = float(_log_spiral_response(vt_abs, rr, tt, pitch=b_val))
    left = [v for k, v in pitch_scores.items() if k < 0]
    right = [v for k, v in pitch_scores.items() if k > 0]
    left_resp = float(max(left)) if left else 0.0
    right_resp = float(max(right)) if right else 0.0
    chirality = float(left_resp - right_resp)
    spiral_score = float(max(left_resp, right_resp))

    return {
        "polar_valid": True,
        "left_response": left_resp,
        "right_response": right_resp,
        "chirality": chirality,
        "spiral_score": spiral_score,
        "dominant_m": int(dominant_m),
        "dominant_m_power": float(mode_power.get(dominant_m, 0.0)),
        "phase_consistency": float(mode_phase_consistency.get(dominant_m, 0.0)),
        "vr_mean_abs": float(np.nanmean(np.abs(v_r))),
        "vt_mean_abs": float(np.nanmean(vt_abs)),
        "n_r": int(r.size),
        "n_theta": int(theta.size),
        "r_max_km": float(r_max_km),
    }


def _log_spiral_response(field: np.ndarray, rr: np.ndarray, tt: np.ndarray, *, pitch: float) -> float:
    valid = np.isfinite(field) & np.isfinite(rr) & np.isfinite(tt) & (rr > 0.0)
    if not np.any(valid):
        return 0.0
    r = rr[valid]
    th = tt[valid]
    x = np.log(np.maximum(r, 1e-6))
    phase = th - (x / float(pitch))
    f = field[valid]
    f = f - np.nanmean(f)
    std = float(np.nanstd(f))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    f = f / std
    comp = np.nanmean(f * np.exp(-1j * phase))
    return float(np.abs(comp))


def _xy_axes_km(lat_vals: np.ndarray, lon_vals: np.ndarray, *, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    x = (lon_vals - float(lon0)) * 111.32 * np.cos(np.deg2rad(float(lat0)))
    y = (lat_vals - float(lat0)) * 111.32
    return x.astype(float), y.astype(float)


def _interp_regular_grid(
    y_axis: np.ndarray,
    x_axis: np.ndarray,
    field: np.ndarray,
    yq: np.ndarray,
    xq: np.ndarray,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (y_axis, x_axis),
        field,
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([yq.reshape(-1), xq.reshape(-1)])
    vals = interp(pts)
    out = vals.reshape(yq.shape)
    if np.any(np.isnan(out)):
        fill = float(np.nanmedian(out[np.isfinite(out)])) if np.any(np.isfinite(out)) else 0.0
        out = np.where(np.isfinite(out), out, fill)
    return out


def _grid_from_frame(
    frame: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> np.ndarray | None:
    pivot = frame.pivot_table(
        index=lat_col,
        columns=lon_col,
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=lat_vals, columns=lon_vals)
    arr = pivot.to_numpy(dtype=float)
    if arr.shape != (lat_vals.size, lon_vals.size):
        return None
    if not np.any(np.isfinite(arr)):
        return None
    if np.any(~np.isfinite(arr)):
        fill = float(np.nanmedian(arr[np.isfinite(arr)]))
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _empty_polar_summary() -> dict[str, Any]:
    return {
        "polar_valid": False,
        "left_response": 0.0,
        "right_response": 0.0,
        "chirality": 0.0,
        "spiral_score": 0.0,
        "dominant_m": 1,
        "dominant_m_power": 0.0,
        "phase_consistency": 0.0,
        "vr_mean_abs": 0.0,
        "vt_mean_abs": 0.0,
        "n_r": 0,
        "n_theta": 0,
        "r_max_km": 0.0,
    }

