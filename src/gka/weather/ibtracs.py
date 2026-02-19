from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_ibtracs_points(
    csv_path: str | Path,
    *,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load IBTrACS positions with normalized columns."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    time_col = _pick_col(df.columns, ["ISO_TIME", "iso_time", "time"])
    lat_col = _pick_col(df.columns, ["LAT", "lat", "latitude"])
    lon_col = _pick_col(df.columns, ["LON", "lon", "longitude"])
    sid_col = _pick_col(df.columns, ["SID", "sid", "storm_id"])
    name_col = _pick_col(df.columns, ["NAME", "name"])
    basin_col = _pick_col(df.columns, ["BASIN", "basin"])
    season_col = _pick_col(df.columns, ["SEASON", "season", "year"])

    if time_col is None or lat_col is None or lon_col is None or sid_col is None:
        raise ValueError(
            "IBTrACS CSV missing required columns. Need at least one of "
            "{ISO_TIME,time}, {LAT,lat}, {LON,lon}, {SID,sid}."
        )

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col], errors="coerce", utc=True, format="mixed").dt.tz_convert(None),
            "lat0": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon0": pd.to_numeric(df[lon_col], errors="coerce"),
            "storm_id": df[sid_col].astype(str),
            "name": df[name_col].astype(str) if name_col else None,
            "basin": df[basin_col].astype(str) if basin_col else None,
            "season": pd.to_numeric(df[season_col], errors="coerce") if season_col else None,
        }
    )
    out = out.dropna(subset=["time", "lat0", "lon0", "storm_id"]).copy()

    # Normalize longitude into [0, 360) if required.
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
    tdf["time"] = pd.to_datetime(tdf["time"], errors="coerce")
    idf["time"] = pd.to_datetime(idf["time"], errors="coerce")
    tdf = tdf.dropna(subset=["time", "lat0", "lon0", "track_id"])
    idf = idf.dropna(subset=["time", "lat0", "lon0", "storm_id"])
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
                cand["lat0"].to_numpy(dtype=float),
                cand["lon0"].to_numpy(dtype=float),
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
            else:
                if (rec["distance_km"], rec["abs_time_offset_h"]) < (
                    best["distance_km"],
                    best["abs_time_offset_h"],
                ):
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


def haversine_km_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371.0
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2, dtype=float) - float(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None)))
    return r * c
