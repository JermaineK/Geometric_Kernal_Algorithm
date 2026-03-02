from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

WIND_CANDIDATES = [
    "USA_WIND",
    "WMO_WIND",
    "BOM_WIND",
    "TOKYO_WIND",
    "CMA_WIND",
    "WIND",
]
PRES_CANDIDATES = [
    "USA_PRES",
    "WMO_PRES",
    "BOM_PRES",
    "TOKYO_PRES",
    "CMA_PRES",
    "PRES",
]


def load_ibtracs_points(
    csv_path: str | Path,
    *,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load IBTrACS positions with normalized columns.

    Notes:
    - Handles the common IBTrACS units/header row.
    - Uses best-available wind/pressure columns with per-row fallback.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS file not found: {path}")

    # Probe schema once.
    preview = pd.read_csv(path, nrows=64, low_memory=False, dtype=str)
    time_col = _pick_col(preview.columns, ["ISO_TIME", "iso_time", "time"])
    lat_col = _pick_col(preview.columns, ["LAT", "lat", "latitude"])
    lon_col = _pick_col(preview.columns, ["LON", "lon", "longitude"])
    sid_col = _pick_col(preview.columns, ["SID", "sid", "storm_id"])
    name_col = _pick_col(preview.columns, ["NAME", "name"])
    basin_col = _pick_col(preview.columns, ["BASIN", "basin"])
    season_col = _pick_col(preview.columns, ["SEASON", "season", "year"])

    if time_col is None or lat_col is None or lon_col is None or sid_col is None:
        raise ValueError(
            "IBTrACS CSV missing required columns. Need at least one of "
            "{ISO_TIME,time}, {LAT,lat}, {LON,lon}, {SID,sid}."
        )

    wind_cols = [c for c in (_pick_col(preview.columns, [k]) for k in WIND_CANDIDATES) if c is not None]
    pres_cols = [c for c in (_pick_col(preview.columns, [k]) for k in PRES_CANDIDATES) if c is not None]

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=max(int(chunksize), 1), low_memory=False, dtype=str):
        c = _drop_units_rows(chunk, time_col=time_col, sid_col=sid_col)
        if c.empty:
            continue

        wind, wind_src = _coalesce_numeric(c, wind_cols)
        pres, pres_src = _coalesce_numeric(c, pres_cols)
        out = pd.DataFrame(
            {
                "time": pd.to_datetime(c[time_col], errors="coerce", utc=True, format="mixed").dt.tz_convert(None),
                "lat0": pd.to_numeric(c[lat_col], errors="coerce"),
                "lon0": pd.to_numeric(c[lon_col], errors="coerce"),
                "storm_id": c[sid_col].astype(str),
                "name": c[name_col].astype(str) if name_col else None,
                "basin": c[basin_col].astype(str) if basin_col else None,
                "season": pd.to_numeric(c[season_col], errors="coerce") if season_col else None,
                "wind": wind,
                "pres": pres,
                "source_wind_col": wind_src,
                "source_pres_col": pres_src,
            }
        )
        out = out.dropna(subset=["time", "lat0", "lon0", "storm_id"]).copy()
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "time",
                "lat0",
                "lon0",
                "storm_id",
                "name",
                "basin",
                "season",
                "wind",
                "pres",
                "source_wind_col",
                "source_pres_col",
            ]
        )

    out = pd.concat(frames, ignore_index=True)

    # Normalize longitude into [0, 360) for consistent domain filtering.
    out["lon0"] = np.mod(out["lon0"].to_numpy(dtype=float), 360.0)

    if lon_min is not None and lon_max is not None:
        lo = float(lon_min) % 360.0
        hi = float(lon_max) % 360.0
        lon = out["lon0"].to_numpy(dtype=float)
        if lo <= hi:
            mask = (lon >= lo) & (lon <= hi)
        else:
            mask = (lon >= lo) | (lon <= hi)
        out = out.loc[mask].copy()
    if lat_min is not None:
        out = out.loc[pd.to_numeric(out["lat0"], errors="coerce") >= float(lat_min)].copy()
    if lat_max is not None:
        out = out.loc[pd.to_numeric(out["lat0"], errors="coerce") <= float(lat_max)].copy()
    if time_min is not None:
        out = out.loc[pd.to_datetime(out["time"], errors="coerce") >= pd.Timestamp(time_min)].copy()
    if time_max is not None:
        out = out.loc[pd.to_datetime(out["time"], errors="coerce") <= pd.Timestamp(time_max)].copy()

    out = out.sort_values(["storm_id", "time"]).reset_index(drop=True)
    return out


def interpolate_ibtracs_hourly(
    points: pd.DataFrame,
    *,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Interpolate IBTrACS storm points to hourly cadence."""

    if points.empty:
        return pd.DataFrame(columns=list(points.columns))
    req = {"storm_id", "time"}
    if not req.issubset(points.columns):
        missing = sorted(req - set(points.columns))
        raise ValueError(f"IBTrACS points missing required columns: {missing}")

    lat_col = "lat0" if "lat0" in points.columns else ("lat" if "lat" in points.columns else None)
    lon_col = "lon0" if "lon0" in points.columns else ("lon" if "lon" in points.columns else None)
    if lat_col is None or lon_col is None:
        raise ValueError("IBTrACS points need lat/lon columns (lat0/lon0 or lat/lon).")

    frame = points.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["storm_id", "time", lat_col, lon_col]).copy()
    if frame.empty:
        return pd.DataFrame(columns=list(points.columns))

    t_min = pd.Timestamp(time_min) if time_min is not None else None
    t_max = pd.Timestamp(time_max) if time_max is not None else None

    rows: list[dict[str, Any]] = []
    for storm_id, grp in frame.groupby("storm_id", sort=False):
        g = grp.sort_values("time").drop_duplicates(subset=["time"], keep="first").copy()
        if g.empty:
            continue

        src_t = pd.to_datetime(g["time"]).astype("int64").to_numpy(dtype=float) / 1e9
        lat_src = pd.to_numeric(g[lat_col], errors="coerce").to_numpy(dtype=float)
        lon_src = pd.to_numeric(g[lon_col], errors="coerce").to_numpy(dtype=float)

        t0 = pd.Timestamp(g["time"].min()).floor("h")
        t1 = pd.Timestamp(g["time"].max()).ceil("h")
        if t_min is not None:
            t0 = max(t0, t_min)
        if t_max is not None:
            t1 = min(t1, t_max)
        if t1 < t0:
            continue

        target = pd.date_range(start=t0, end=t1, freq="1h")
        trg_t = pd.to_datetime(target).astype("int64").to_numpy(dtype=float) / 1e9

        lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon_src)))
        lat_itp = np.interp(trg_t, src_t, lat_src)
        lon_itp = np.mod(np.interp(trg_t, src_t, lon_unwrapped), 360.0)
        dt_nearest_h = _nearest_time_delta_hours(trg_t=trg_t, src_t=src_t)
        interp_flag = np.where(dt_nearest_h <= (1.0 / 60.0), "exact", "interpolated")

        wind_src = (
            pd.to_numeric(g["wind"], errors="coerce").to_numpy(dtype=float)
            if "wind" in g.columns
            else np.full(src_t.shape, np.nan, dtype=float)
        )
        pres_src = (
            pd.to_numeric(g["pres"], errors="coerce").to_numpy(dtype=float)
            if "pres" in g.columns
            else np.full(src_t.shape, np.nan, dtype=float)
        )
        wind_itp = np.interp(trg_t, src_t, wind_src) if np.sum(np.isfinite(wind_src)) >= 2 else np.full(trg_t.shape, np.nan)
        pres_itp = np.interp(trg_t, src_t, pres_src) if np.sum(np.isfinite(pres_src)) >= 2 else np.full(trg_t.shape, np.nan)
        speed_kmh = np.full(trg_t.shape, np.nan, dtype=float)
        if trg_t.size >= 2:
            step_h = np.diff(trg_t) / 3600.0
            step_h = np.where(step_h > 0.0, step_h, np.nan)
            step_km = haversine_km_vec(
                lat_itp[:-1],
                lon_itp[:-1],
                lat_itp[1:],
                lon_itp[1:],
            )
            step_speed = step_km / step_h
            speed_kmh[1:] = step_speed
            speed_kmh[0] = step_speed[0] if step_speed.size > 0 else np.nan

        name = str(g["name"].dropna().iloc[0]) if "name" in g.columns and g["name"].notna().any() else None
        basin = str(g["basin"].dropna().iloc[0]) if "basin" in g.columns and g["basin"].notna().any() else None
        season = pd.to_numeric(g["season"], errors="coerce").dropna()
        season_v = int(season.iloc[0]) if not season.empty else None
        wind_src_col = (
            str(g["source_wind_col"].dropna().iloc[0])
            if "source_wind_col" in g.columns and g["source_wind_col"].notna().any()
            else None
        )
        pres_src_col = (
            str(g["source_pres_col"].dropna().iloc[0])
            if "source_pres_col" in g.columns and g["source_pres_col"].notna().any()
            else None
        )

        for i, t in enumerate(target):
            dt_fix_h = float(dt_nearest_h[i]) if np.isfinite(dt_nearest_h[i]) else np.nan
            spd_kmh = float(speed_kmh[i]) if np.isfinite(speed_kmh[i]) else np.nan
            center_uncert_km = (
                float(min(500.0, max(0.0, 0.5 * spd_kmh * dt_fix_h)))
                if np.isfinite(spd_kmh) and np.isfinite(dt_fix_h)
                else np.nan
            )
            rows.append(
                {
                    "storm_id": str(storm_id),
                    "name": name,
                    "season": season_v,
                    "basin": basin,
                    "time": pd.Timestamp(t),
                    "lat0": float(lat_itp[i]),
                    "lon0": float(lon_itp[i]),
                    "wind": float(wind_itp[i]) if np.isfinite(wind_itp[i]) else np.nan,
                    "pres": float(pres_itp[i]) if np.isfinite(pres_itp[i]) else np.nan,
                    "source_wind_col": wind_src_col,
                    "source_pres_col": pres_src_col,
                    "dt_to_nearest_ib_hours": dt_fix_h,
                    "interp_flag": str(interp_flag[i]),
                    "speed_kmh": spd_kmh,
                    "speed_qc_flag": bool(np.isfinite(spd_kmh) and (spd_kmh > 160.0)),
                    "center_uncert_km": center_uncert_km,
                }
            )

    if not rows:
        return pd.DataFrame(columns=list(points.columns))
    return pd.DataFrame(rows).sort_values(["storm_id", "time"]).reset_index(drop=True)


def prepare_ibtracs_catalog(
    csv_path: str | Path,
    *,
    out_path: str | Path,
    hourly_out_path: str | Path | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Build filtered + hourly IBTrACS parquet catalogs."""

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    points = load_ibtracs_points(
        csv_path,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_min=time_min,
        time_max=time_max,
    )
    points.to_parquet(out_p, index=False)

    hourly = pd.DataFrame()
    hourly_path = Path(hourly_out_path) if hourly_out_path is not None else None
    if hourly_path is not None:
        hourly_path.parent.mkdir(parents=True, exist_ok=True)
        hourly = interpolate_ibtracs_hourly(points, time_min=time_min, time_max=time_max)
        hourly.to_parquet(hourly_path, index=False)

    return {
        "catalog_path": str(out_p.resolve()),
        "hourly_path": str(hourly_path.resolve()) if hourly_path is not None else None,
        "n_points": int(points.shape[0]),
        "n_hourly_points": int(hourly.shape[0]),
        "n_storms": int(points["storm_id"].nunique()) if not points.empty else 0,
    }


def match_tracks_to_ibtracs(
    tracks: pd.DataFrame,
    ib_points: pd.DataFrame,
    *,
    max_time_hours: float = 6.0,
    max_distance_km: float = 300.0,
) -> dict[str, Any]:
    """Nearest-neighbor spatiotemporal match between discovered tracks and IBTrACS points."""

    if tracks.empty or ib_points.empty:
        return {
            "n_tracks": int(tracks["track_id"].nunique()) if "track_id" in tracks.columns else 0,
            "n_matched_tracks": 0,
            "match_rate": 0.0,
            "median_distance_km": None,
            "median_abs_time_offset_h": None,
            "matches": [],
        }

    tdf = tracks.copy()
    idf = ib_points.copy()
    ib_lat_col = "lat0" if "lat0" in idf.columns else ("lat" if "lat" in idf.columns else None)
    ib_lon_col = "lon0" if "lon0" in idf.columns else ("lon" if "lon" in idf.columns else None)
    if ib_lat_col is None or ib_lon_col is None:
        raise ValueError("IBTrACS points require lat/lon columns (lat0/lon0 or lat/lon).")

    tdf["time"] = pd.to_datetime(tdf["time"], errors="coerce")
    idf["time"] = pd.to_datetime(idf["time"], errors="coerce")
    tdf = tdf.dropna(subset=["time", "lat0", "lon0", "track_id"])
    idf = idf.dropna(subset=["time", ib_lat_col, ib_lon_col, "storm_id"])
    if tdf.empty or idf.empty:
        return {
            "n_tracks": int(tdf["track_id"].nunique()) if "track_id" in tdf.columns else 0,
            "n_matched_tracks": 0,
            "match_rate": 0.0,
            "median_distance_km": None,
            "median_abs_time_offset_h": None,
            "matches": [],
        }

    matches: list[dict[str, Any]] = []
    for tid, grp in tdf.groupby("track_id", sort=False):
        best: dict[str, Any] | None = None
        for _, row in grp.iterrows():
            dt_h = np.abs((idf["time"] - pd.Timestamp(row["time"])).dt.total_seconds() / 3600.0)
            cand = idf.loc[dt_h <= float(max_time_hours)].copy()
            if cand.empty:
                continue
            cand["dist_km"] = haversine_km_vec(
                float(row["lat0"]),
                float(row["lon0"]),
                cand[ib_lat_col].to_numpy(dtype=float),
                cand[ib_lon_col].to_numpy(dtype=float),
            )
            cand["dt_h"] = np.abs((cand["time"] - pd.Timestamp(row["time"])).dt.total_seconds() / 3600.0)
            cand = cand.loc[pd.to_numeric(cand["dist_km"], errors="coerce") <= float(max_distance_km)]
            if cand.empty:
                continue
            cand = cand.sort_values(["dist_km", "dt_h"]).reset_index(drop=True)
            top = cand.iloc[0]
            rec = {
                "track_id": int(tid),
                "storm_id": str(top["storm_id"]),
                "distance_km": float(top["dist_km"]),
                "abs_time_offset_h": float(top["dt_h"]),
                "track_time": str(pd.Timestamp(row["time"])),
                "ib_time": str(pd.Timestamp(top["time"])),
            }
            if best is None:
                best = rec
            elif (rec["distance_km"], rec["abs_time_offset_h"]) < (best["distance_km"], best["abs_time_offset_h"]):
                best = rec
        if best is not None:
            matches.append(best)

    n_tracks = int(tdf["track_id"].nunique())
    n_matched = int(len({int(m["track_id"]) for m in matches}))
    dist = [float(m["distance_km"]) for m in matches if np.isfinite(float(m["distance_km"]))]
    dt = [float(m["abs_time_offset_h"]) for m in matches if np.isfinite(float(m["abs_time_offset_h"]))]
    return {
        "n_tracks": n_tracks,
        "n_matched_tracks": n_matched,
        "match_rate": float(n_matched / max(n_tracks, 1)),
        "median_distance_km": float(np.median(dist)) if dist else None,
        "median_abs_time_offset_h": float(np.median(dt)) if dt else None,
        "matches": matches,
    }


def _pick_col(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        key = str(c).lower()
        if key in lookup:
            return lookup[key]
    return None


def _drop_units_rows(chunk: pd.DataFrame, *, time_col: str, sid_col: str) -> pd.DataFrame:
    out = chunk.copy()
    if out.empty:
        return out
    tvals = out[time_col].astype(str).str.strip().str.upper()
    svals = out[sid_col].astype(str).str.strip().str.upper()
    bad = tvals.isin({"ISO_TIME", "TIME", "YYYY-MM-DD HH:MM:SS"}) | svals.isin({"SID", "STORM_ID", ""})
    return out.loc[~bad].copy()


def _coalesce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> tuple[pd.Series, pd.Series]:
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return (
            pd.Series(np.nan, index=df.index, dtype=float),
            pd.Series([None] * len(df), index=df.index, dtype=object),
        )

    values = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col in use_cols:
        v = pd.to_numeric(df[col], errors="coerce")
        mask = values.isna() & v.notna()
        if mask.any():
            values.loc[mask] = v.loc[mask]
            source.loc[mask] = str(col)
    return values, source


def haversine_km_vec(lat1: float | np.ndarray, lon1: float | np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371.0
    p1 = np.deg2rad(np.asarray(lat1, dtype=float))
    p2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None)))
    return r * c


def _nearest_time_delta_hours(*, trg_t: np.ndarray, src_t: np.ndarray) -> np.ndarray:
    if trg_t.size == 0:
        return np.array([], dtype=float)
    if src_t.size == 0:
        return np.full(trg_t.shape, np.nan, dtype=float)
    idx = np.searchsorted(src_t, trg_t, side="left")
    prev_idx = np.clip(idx - 1, 0, src_t.size - 1)
    next_idx = np.clip(idx, 0, src_t.size - 1)
    dt_prev = np.abs(trg_t - src_t[prev_idx]) / 3600.0
    dt_next = np.abs(src_t[next_idx] - trg_t) / 3600.0
    return np.minimum(dt_prev, dt_next)
