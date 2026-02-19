from __future__ import annotations

import pandas as pd

from gka.weather.ibtracs import load_ibtracs_points


def test_load_ibtracs_points_normalizes_columns(tmp_path) -> None:
    csv = tmp_path / "ib.csv"
    csv.write_text(
        "SID,ISO_TIME,LAT,LON,NAME\n"
        "AL012025,2025-03-01 00:00:00,-15.0,150.0,TEST\n"
        "AL012025,2025-03-01 06:00:00,-15.5,151.0,TEST\n",
        encoding="utf-8",
    )
    out = load_ibtracs_points(csv, lon_min=140.0, lon_max=170.0)
    assert not out.empty
    assert set(["time", "lat0", "lon0", "storm_id"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["time"])
    assert float(out["lon0"].min()) >= 140.0
    assert float(out["lon0"].max()) <= 170.0

