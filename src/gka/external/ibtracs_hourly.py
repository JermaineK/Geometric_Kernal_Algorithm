from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gka.weather.ibtracs import interpolate_ibtracs_hourly, load_ibtracs_points


def build_ibtracs_hourly_tracks(
    csv_path: str | Path,
    *,
    out_path: str | Path,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    time_min: pd.Timestamp | None = None,
    time_max: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Build an hourly IBTrACS parquet table with interpolation uncertainty fields.

    Output contains at least:
    - `storm_id`, `time`, `lat0`, `lon0`
    - `dt_to_nearest_ib_hours`
    - `interp_flag`
    - `speed_kmh`, `speed_qc_flag`
    """

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    points = load_ibtracs_points(
        csv_path,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_min=time_min,
        time_max=time_max,
    )
    hourly = interpolate_ibtracs_hourly(points, time_min=time_min, time_max=time_max)
    hourly.to_parquet(out, index=False)
    return {
        "path": str(out.resolve()),
        "n_rows": int(hourly.shape[0]),
        "n_storms": int(hourly["storm_id"].nunique()) if not hourly.empty and "storm_id" in hourly.columns else 0,
        "time_min": str(pd.to_datetime(hourly["time"], errors="coerce").min()) if not hourly.empty and "time" in hourly.columns else None,
        "time_max": str(pd.to_datetime(hourly["time"], errors="coerce").max()) if not hourly.empty and "time" in hourly.columns else None,
    }
