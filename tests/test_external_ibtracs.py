from __future__ import annotations

import pandas as pd

from gka.external.ibtracs import build_track_index, nearest_ibtracs_fix, read_ibtracs


def test_read_ibtracs_drops_units_rows(tmp_path) -> None:
    csv = tmp_path / "ibtracs.csv"
    csv.write_text(
        "SID,ISO_TIME,LAT,LON,NAME,BASIN,USA_WIND,USA_PRES\n"
        "Year,ISO_TIME,degrees_north,degrees_east,text,text,kt,hPa\n"
        "SI012025,2025-03-01 00:00:00,-15.0,150.0,ALICE,SI,40,995\n"
        "SI012025,2025-03-01 06:00:00,-15.4,151.0,ALICE,SI,45,990\n",
        encoding="utf-8",
    )
    out = read_ibtracs(csv)
    assert not out.empty
    assert int(out.shape[0]) == 2
    assert "Year" not in set(out["sid"].astype(str))
    assert pd.api.types.is_datetime64_any_dtype(out["time_utc"])
    assert pd.api.types.is_float_dtype(out["lat"])
    assert pd.api.types.is_float_dtype(out["lon"])
    assert float(out["lon"].min()) >= -180.0
    assert float(out["lon"].max()) <= 180.0


def test_nearest_ibtracs_fix_returns_match_within_dt_and_distance() -> None:
    pts = pd.DataFrame(
        {
            "sid": ["SI012025", "SI012025"],
            "time_utc": pd.to_datetime(["2025-03-01 00:00:00", "2025-03-01 01:00:00"]),
            "lat": [-15.0, -15.2],
            "lon": [150.0, 150.1],
            "name": ["ALICE", "ALICE"],
            "basin": ["SI", "SI"],
        }
    )
    idx = build_track_index(pts)
    hit = nearest_ibtracs_fix(
        idx,
        time=pd.Timestamp("2025-03-01 00:30:00"),
        lat=-15.1,
        lon=150.05,
        dt_max_hours=2.0,
        r_max_km=500.0,
    )
    assert hit is not None
    assert hit["sid"] == "SI012025"
    assert hit["dist_km"] < 100.0
    assert hit["dt_hours"] <= 1.0
