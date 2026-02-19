from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class VortexDetectConfig:
    sigma_cells: float = 1.0
    zeta_percentile: float = 99.0
    min_separation_km: float = 180.0
    ow_threshold: float = 0.0
    max_candidates_per_time: int = 16
    neighborhood_cells: int = 1


def detect_vortex_candidates(
    frame: pd.DataFrame,
    *,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    u_col: str = "u10",
    v_col: str = "v10",
    mslp_col: str | None = None,
    lead_bucket: str | None = None,
    config: VortexDetectConfig | None = None,
) -> pd.DataFrame:
    """Detect storm-like vortex cores from gridded wind slices."""

    cfg = config or VortexDetectConfig()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "lat0",
                "lon0",
                "zeta_peak",
                "ow_median",
                "score",
                "mslp_min",
                "candidate_rank",
                "grid_i",
                "grid_j",
                "lead_bucket",
            ]
        )

    df = frame.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col, u_col, v_col])
    if df.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, Any]] = []
    for t_val, grp in df.groupby(time_col, sort=True, dropna=False):
        recs = _detect_candidates_one_time(
            grp,
            lat_col=lat_col,
            lon_col=lon_col,
            u_col=u_col,
            v_col=v_col,
            mslp_col=mslp_col,
            cfg=cfg,
        )
        for rank, rec in enumerate(recs):
            rec["candidate_rank"] = int(rank + 1)
            rec["time"] = pd.Timestamp(t_val)
            rec["lead_bucket"] = str(lead_bucket) if lead_bucket is not None else None
            out_rows.append(rec)
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows)
    out = out.sort_values(["time", "candidate_rank"]).reset_index(drop=True)
    return out


def _detect_candidates_one_time(
    grp: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    u_col: str,
    v_col: str,
    mslp_col: str | None,
    cfg: VortexDetectConfig,
) -> list[dict[str, Any]]:
    lat_vals = np.sort(pd.to_numeric(grp[lat_col], errors="coerce").dropna().unique().astype(float))
    lon_vals = np.sort(pd.to_numeric(grp[lon_col], errors="coerce").dropna().unique().astype(float))
    if lat_vals.size < 3 or lon_vals.size < 3:
        return []

    u_grid = _grid_from_frame(grp, lat_col=lat_col, lon_col=lon_col, val_col=u_col, lat_vals=lat_vals, lon_vals=lon_vals)
    v_grid = _grid_from_frame(grp, lat_col=lat_col, lon_col=lon_col, val_col=v_col, lat_vals=lat_vals, lon_vals=lon_vals)
    if u_grid is None or v_grid is None:
        return []

    zeta, gradients = compute_relative_vorticity(u_grid, v_grid, lat_vals, lon_vals)
    ow = compute_okubo_weiss(gradients)
    zeta_smooth = gaussian_filter(zeta, sigma=float(cfg.sigma_cells), mode="nearest")
    abs_z = np.abs(zeta_smooth)
    finite_mask = np.isfinite(abs_z)
    if not np.any(finite_mask):
        return []
    thresh = float(np.nanpercentile(abs_z[finite_mask], float(cfg.zeta_percentile)))
    if not np.isfinite(thresh) or thresh <= 0:
        return []

    mean_dx_km, mean_dy_km = _mean_grid_spacing_km(lat_vals, lon_vals)
    cell_km = float(np.sqrt(max(mean_dx_km, 1e-6) * max(mean_dy_km, 1e-6)))
    sep_cells = max(1, int(np.ceil(float(cfg.min_separation_km) / max(cell_km, 1e-6))))
    max_filt = maximum_filter(abs_z, size=int(2 * sep_cells + 1), mode="nearest")
    cand_mask = finite_mask & (abs_z >= thresh) & (abs_z >= max_filt - 1e-14)
    idx = np.argwhere(cand_mask)
    if idx.size == 0:
        return []

    mslp_grid: np.ndarray | None = None
    if mslp_col and mslp_col in grp.columns:
        mslp_grid = _grid_from_frame(
            grp,
            lat_col=lat_col,
            lon_col=lon_col,
            val_col=mslp_col,
            lat_vals=lat_vals,
            lon_vals=lon_vals,
        )

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    ow_scale = float(np.nanstd(ow[np.isfinite(ow)])) if np.any(np.isfinite(ow)) else 1.0
    if not np.isfinite(ow_scale) or ow_scale <= 1e-12:
        ow_scale = 1.0
    for (iy, ix) in idx:
        y0, y1 = max(0, iy - int(cfg.neighborhood_cells)), min(ow.shape[0], iy + int(cfg.neighborhood_cells) + 1)
        x0, x1 = max(0, ix - int(cfg.neighborhood_cells)), min(ow.shape[1], ix + int(cfg.neighborhood_cells) + 1)
        ow_med = float(np.nanmedian(ow[y0:y1, x0:x1]))
        if not np.isfinite(ow_med):
            continue
        if ow_med > float(cfg.ow_threshold):
            continue

        lat0 = float(lat_vals[iy])
        lon0 = float(lon_vals[ix])
        z_pk = float(zeta_smooth[iy, ix])
        mslp_min = None
        if mslp_grid is not None:
            window = mslp_grid[y0:y1, x0:x1]
            if np.any(np.isfinite(window)):
                mslp_min = float(np.nanmin(window))

        score = (abs(z_pk) / max(thresh, 1e-12)) + max(0.0, -ow_med / ow_scale)
        rows.append(
            {
                "lat0": lat0,
                "lon0": lon0,
                "zeta_peak": z_pk,
                "ow_median": ow_med,
                "score": float(score),
                "mslp_min": mslp_min,
                "grid_i": int(iy),
                "grid_j": int(ix),
            }
        )
        scores.append(float(score))

    if not rows:
        return []
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    keep = order[: int(max(cfg.max_candidates_per_time, 1))]
    return [rows[int(i)] for i in keep]


def compute_relative_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute relative vorticity with spherical metric corrections."""

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shape")
    if u.ndim != 2:
        raise ValueError("u and v must be 2D arrays")

    lat_rad = np.deg2rad(lat_vals)
    lon_rad = np.deg2rad(lon_vals)
    dlat = np.gradient(lat_rad)
    dlon = np.gradient(lon_rad)
    dy = EARTH_RADIUS_M * dlat[:, None]
    dx = EARTH_RADIUS_M * np.cos(lat_rad)[:, None] * dlon[None, :]
    dx = np.where(np.abs(dx) > 1e-9, dx, np.nan)
    dy = np.where(np.abs(dy) > 1e-9, dy, np.nan)

    du_dy = np.gradient(u, axis=0) / dy
    du_dx = np.gradient(u, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    dv_dx = np.gradient(v, axis=1) / dx
    zeta = dv_dx - du_dy
    gradients = {"du_dx": du_dx, "du_dy": du_dy, "dv_dx": dv_dx, "dv_dy": dv_dy}
    return zeta, gradients


def compute_okubo_weiss(gradients: dict[str, np.ndarray]) -> np.ndarray:
    du_dx = np.asarray(gradients["du_dx"], dtype=float)
    du_dy = np.asarray(gradients["du_dy"], dtype=float)
    dv_dx = np.asarray(gradients["dv_dx"], dtype=float)
    dv_dy = np.asarray(gradients["dv_dy"], dtype=float)
    zeta = dv_dx - du_dy
    s_n = du_dx - dv_dy
    s_s = dv_dx + du_dy
    return np.square(s_n) + np.square(s_s) - np.square(zeta)


def _grid_from_frame(
    frame: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    val_col: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> np.ndarray | None:
    if val_col not in frame.columns:
        return None
    pivot = frame.pivot_table(
        index=lat_col,
        columns=lon_col,
        values=val_col,
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


def _mean_grid_spacing_km(lat_vals: np.ndarray, lon_vals: np.ndarray) -> tuple[float, float]:
    if lat_vals.size < 2 or lon_vals.size < 2:
        return 1.0, 1.0
    dlat = np.median(np.abs(np.diff(np.asarray(lat_vals, dtype=float))))
    dlon = np.median(np.abs(np.diff(np.asarray(lon_vals, dtype=float))))
    lat0 = float(np.nanmean(np.asarray(lat_vals, dtype=float)))
    dy = 111.32 * dlat
    dx = 111.32 * np.cos(np.deg2rad(lat0)) * dlon
    return float(max(dx, 1e-6)), float(max(dy, 1e-6))

