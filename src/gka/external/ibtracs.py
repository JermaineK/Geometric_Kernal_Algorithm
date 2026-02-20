from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WIND_CANDIDATES = (
    "USA_WIND",
    "WMO_WIND",
    "BOM_WIND",
    "TOKYO_WIND",
    "CMA_WIND",
    "WIND",
)
PRES_CANDIDATES = (
    "USA_PRES",
    "WMO_PRES",
    "BOM_PRES",
    "TOKYO_PRES",
    "CMA_PRES",
    "PRES",
)


@dataclass(frozen=True)
class IBTracsTrackIndex:
    points: pd.DataFrame
    by_time: dict[pd.Timestamp, pd.DataFrame]
    lon_convention: str


def normalize_lon_180(values: Any) -> Any:
    arr = np.asarray(values, dtype=float)
    wrapped = ((arr + 180.0) % 360.0) - 180.0
    if np.isscalar(values):
        return float(wrapped.item())
    return wrapped


def haversine_km_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371.0
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2, dtype=float) - float(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None)))
    return r * c


def read_ibtracs(
    csv_path: str | Path,
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read IBTrACS CSV and normalize to canonical columns.

    Output columns:
    `sid, time_utc, lat, lon, name, basin, season, wind, pres, source_wind_col, source_pres_col`
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS file not found: {path}")

    preview = pd.read_csv(path, nrows=64, low_memory=False, dtype=str)
    time_col = _pick_col(preview.columns, ("ISO_TIME", "iso_time", "time"))
    lat_col = _pick_col(preview.columns, ("LAT", "lat", "latitude"))
    lon_col = _pick_col(preview.columns, ("LON", "lon", "longitude"))
    sid_col = _pick_col(preview.columns, ("SID", "sid", "storm_id"))
    name_col = _pick_col(preview.columns, ("NAME", "name"))
    basin_col = _pick_col(preview.columns, ("BASIN", "basin", "subbasin"))
    season_col = _pick_col(preview.columns, ("SEASON", "season", "year"))

    if time_col is None or lat_col is None or lon_col is None or sid_col is None:
        raise ValueError("IBTrACS CSV missing one or more required columns: SID, ISO_TIME, LAT, LON")

    wind_cols = [c for c in (_pick_col(preview.columns, (k,)) for k in WIND_CANDIDATES) if c is not None]
    pres_cols = [c for c in (_pick_col(preview.columns, (k,)) for k in PRES_CANDIDATES) if c is not None]

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=max(1, int(chunksize)), low_memory=False, dtype=str):
        c = _drop_non_data_rows(chunk, sid_col=sid_col, time_col=time_col, lat_col=lat_col, lon_col=lon_col)
        if c.empty:
            continue

        t = pd.to_datetime(c[time_col], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        lat = pd.to_numeric(c[lat_col], errors="coerce")
        lon = pd.to_numeric(c[lon_col], errors="coerce")
        sid = c[sid_col].astype(str).str.strip()

        wind, wind_src = _coalesce_numeric(c, wind_cols)
        pres, pres_src = _coalesce_numeric(c, pres_cols)
        out = pd.DataFrame(
            {
                "sid": sid,
                "time_utc": t,
                "lat": lat,
                "lon": normalize_lon_180(lon.to_numpy(dtype=float)),
                "name": c[name_col].astype(str) if name_col else None,
                "basin": c[basin_col].astype(str) if basin_col else None,
                "season": pd.to_numeric(c[season_col], errors="coerce") if season_col else None,
                "wind": wind,
                "pres": pres,
                "source_wind_col": wind_src,
                "source_pres_col": pres_src,
            }
        )
        out = out.dropna(subset=["sid", "time_utc", "lat", "lon"]).copy()
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "sid",
                "time_utc",
                "lat",
                "lon",
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
    out = out.sort_values(["sid", "time_utc"]).reset_index(drop=True)
    return out


def build_track_index(
    points: pd.DataFrame,
    *,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    basins: list[str] | None = None,
) -> IBTracsTrackIndex:
    """Build time-bucketed IBTrACS index for nearest-fix lookups."""

    if points is None or points.empty:
        return IBTracsTrackIndex(points=pd.DataFrame(), by_time={}, lon_convention="[-180,180)")
    req = {"sid", "time_utc", "lat", "lon"}
    missing = sorted(req - set(points.columns))
    if missing:
        raise ValueError(f"IBTrACS points missing required columns: {missing}")

    frame = points.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = normalize_lon_180(pd.to_numeric(frame["lon"], errors="coerce").to_numpy(dtype=float))
    frame = frame.dropna(subset=["sid", "time_utc", "lat", "lon"]).copy()
    if frame.empty:
        return IBTracsTrackIndex(points=pd.DataFrame(), by_time={}, lon_convention="[-180,180)")

    if time_min is not None:
        frame = frame.loc[frame["time_utc"] >= pd.Timestamp(time_min)].copy()
    if time_max is not None:
        frame = frame.loc[frame["time_utc"] <= pd.Timestamp(time_max)].copy()
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = [float(v) for v in bbox]
        lon_min = float(normalize_lon_180(lon_min))
        lon_max = float(normalize_lon_180(lon_max))
        frame = frame.loc[(frame["lat"] >= lat_min) & (frame["lat"] <= lat_max)].copy()
        lon = frame["lon"].to_numpy(dtype=float)
        if lon_min <= lon_max:
            lmask = (lon >= lon_min) & (lon <= lon_max)
        else:
            lmask = (lon >= lon_min) | (lon <= lon_max)
        frame = frame.loc[lmask].copy()
    if basins:
        basin_set = {str(v).strip().upper() for v in basins if str(v).strip()}
        if basin_set and "basin" in frame.columns:
            frame = frame.loc[frame["basin"].astype(str).str.upper().isin(basin_set)].copy()

    frame["time_hour"] = pd.to_datetime(frame["time_utc"], errors="coerce").dt.floor("h")
    by_time = {
        pd.Timestamp(t): g[["sid", "time_utc", "lat", "lon", *[c for c in ("name", "basin", "wind", "pres") if c in g.columns]]].copy()
        for t, g in frame.groupby("time_hour", sort=False)
    }
    frame = frame.drop(columns=["time_hour"], errors="ignore").reset_index(drop=True)
    return IBTracsTrackIndex(points=frame, by_time=by_time, lon_convention="[-180,180)")


def nearest_ibtracs_fix(
    index: IBTracsTrackIndex,
    *,
    time: pd.Timestamp,
    lat: float,
    lon: float,
    dt_max_hours: float = 3.0,
    r_max_km: float | None = None,
) -> dict[str, Any] | None:
    if index is None or index.points.empty or not index.by_time:
        return None
    if pd.isna(time):
        return None

    t0 = pd.Timestamp(time).floor("h")
    offsets = int(np.ceil(max(float(dt_max_hours), 0.0)))
    candidates: list[pd.DataFrame] = []
    for h in range(-offsets, offsets + 1):
        tk = t0 + pd.Timedelta(hours=h)
        g = index.by_time.get(tk)
        if g is not None and not g.empty:
            candidates.append(g)
    if not candidates:
        return None
    cand = pd.concat(candidates, ignore_index=True)
    if cand.empty:
        return None
    lon_q = float(normalize_lon_180(lon))
    dist = haversine_km_vec(float(lat), lon_q, cand["lat"].to_numpy(dtype=float), normalize_lon_180(cand["lon"].to_numpy(dtype=float)))
    dt = np.abs((pd.to_datetime(cand["time_utc"], errors="coerce") - pd.Timestamp(time)).dt.total_seconds().to_numpy(dtype=float) / 3600.0)
    finite = np.isfinite(dist) & np.isfinite(dt) & (dt <= float(dt_max_hours))
    if r_max_km is not None:
        finite = finite & (dist <= float(r_max_km))
    if not np.any(finite):
        return None
    idx = int(np.argmin(np.where(finite, dist, np.inf)))
    row = cand.iloc[idx]
    return {
        "sid": str(row["sid"]),
        "name": str(row["name"]) if "name" in cand.columns and pd.notna(row.get("name")) else None,
        "basin": str(row["basin"]) if "basin" in cand.columns and pd.notna(row.get("basin")) else None,
        "t_fix": pd.Timestamp(row["time_utc"]),
        "dt_hours": float(dt[idx]),
        "dist_km": float(dist[idx]),
        "lat": float(row["lat"]),
        "lon": float(normalize_lon_180(float(row["lon"]))),
        "wind": float(row["wind"]) if "wind" in cand.columns and pd.notna(row.get("wind")) else None,
        "pres": float(row["pres"]) if "pres" in cand.columns and pd.notna(row.get("pres")) else None,
    }


def _pick_col(columns: pd.Index, candidates: tuple[str, ...] | list[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        k = str(c).lower()
        if k in lookup:
            return lookup[k]
    return None


def _drop_non_data_rows(
    chunk: pd.DataFrame,
    *,
    sid_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    out = chunk.copy()
    if out.empty:
        return out
    sid = out[sid_col].astype(str).str.strip()
    sid_u = sid.str.upper()
    time_v = out[time_col].astype(str).str.strip().str.upper()
    lat_v = out[lat_col].astype(str).str.strip().str.upper()
    lon_v = out[lon_col].astype(str).str.strip().str.upper()

    # Units/header lines and malformed rows.
    bad = (
        sid_u.isin({"", "SID", "STORM_ID", "YEAR"})
        | time_v.isin({"", "ISO_TIME", "TIME", "YYYY-MM-DD HH:MM:SS"})
        | lat_v.isin({"", "LAT", "DEGREES_NORTH"})
        | lon_v.isin({"", "LON", "DEGREES_EAST"})
    )
    # Drop obvious non-storm metadata labels leaked into SID.
    bad = bad | sid_u.str.contains("UNITS", regex=False, na=False)
    return out.loc[~bad].copy()


def _coalesce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return (
            pd.Series(np.nan, index=df.index, dtype=float),
            pd.Series([None] * len(df), index=df.index, dtype=object),
        )
    vals = pd.Series(np.nan, index=df.index, dtype=float)
    src = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col in use_cols:
        num = pd.to_numeric(df[col], errors="coerce")
        mask = vals.isna() & num.notna()
        if mask.any():
            vals.loc[mask] = num.loc[mask]
            src.loc[mask] = str(col)
    return vals, src
