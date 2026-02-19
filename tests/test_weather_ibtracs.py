from __future__ import annotations

import pandas as pd

from gka.weather.ibtracs import interpolate_ibtracs_hourly, load_ibtracs_points


def test_load_ibtracs_points_normalizes_columns(tmp_path) -> None:
    csv = tmp_path / "ib.csv"
    csv.write_text(
        "SID,ISO_TIME,LAT,LON,NAME,USA_WIND,WMO_WIND,USA_PRES\n"
        ",ISO_TIME,degrees_north,degrees_east,text,kt,kt,hPa\n"
        "AL012025,2025-03-01 00:00:00,-15.0,150.0,TEST\n"
        "AL012025,2025-03-01 06:00:00,-15.5,151.0,TEST,45,,980\n",
        encoding="utf-8",
    )
    out = load_ibtracs_points(
        csv,
        lon_min=140.0,
        lon_max=170.0,
        time_min=pd.Timestamp("2025-03-01 00:00:00"),
        time_max=pd.Timestamp("2025-03-31 00:00:00"),
    )
    assert not out.empty
    assert set(["time", "lat0", "lon0", "storm_id", "wind", "pres", "source_wind_col", "source_pres_col"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["time"])
    assert float(out["lon0"].min()) >= 140.0
    assert float(out["lon0"].max()) <= 170.0
    assert int(out.shape[0]) == 2
    assert out["source_wind_col"].dropna().iloc[0] in {"USA_WIND", "WMO_WIND"}


def test_interpolate_ibtracs_hourly_builds_hourly_track() -> None:
    points = pd.DataFrame(
        {
            "storm_id": ["AL012025", "AL012025"],
            "time": [pd.Timestamp("2025-03-01 00:00:00"), pd.Timestamp("2025-03-01 06:00:00")],
            "lat0": [-15.0, -16.2],
            "lon0": [150.0, 151.2],
            "wind": [40.0, 50.0],
            "pres": [990.0, 980.0],
            "name": ["TEST", "TEST"],
            "basin": ["SI", "SI"],
            "season": [2025, 2025],
        }
    )
    out = interpolate_ibtracs_hourly(
        points,
        time_min=pd.Timestamp("2025-03-01 00:00:00"),
        time_max=pd.Timestamp("2025-03-01 06:00:00"),
    )
    assert not out.empty
    assert out["time"].nunique() == 7
    assert float(out["lat0"].min()) <= -15.0
    assert float(out["lat0"].max()) >= -16.2
