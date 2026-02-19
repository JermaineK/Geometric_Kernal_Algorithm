import pandas as pd

from gka.adapters.weather_era5 import WeatherERA5Adapter


def test_weather_mirror_lon_and_parity_channels():
    adapter = WeatherERA5Adapter()
    frame = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01 00:00:00")] * 4,
            "lat": [-20.0, -20.0, -19.75, -19.75],
            "lon": [140.0, 160.0, 140.0, 160.0],
            "u10": [1.0, 2.0, 3.0, 4.0],
            "v10": [5.0, 6.0, 7.0, 8.0],
            "S": [0.1, 0.2, 0.3, 0.4],
        }
    )

    mirrored = adapter.mirror_lon_about(
        frame,
        lon0=150.0,
        scalar_cols=["S"],
        max_lon_distance=0.01,
    )
    out = adapter.add_parity_channels(mirrored)

    row = out.loc[(out["lat"] == -20.0) & (out["lon"] == 140.0)].iloc[0]
    assert row["u10_mirror"] == -2.0
    assert row["v10_mirror"] == 6.0
    assert row["S_mirror"] == 0.2
    assert 0.0 <= float(row["eta_parity"]) <= 1.0
