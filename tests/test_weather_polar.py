from __future__ import annotations

import numpy as np
import pandas as pd

from gka.weather.polar import summarize_polar_features


def test_summarize_polar_features_returns_valid_metrics() -> None:
    lat0 = -20.0
    lon0 = 150.0
    lat = np.linspace(lat0 - 1.0, lat0 + 1.0, 25)
    lon = np.linspace(lon0 - 1.0, lon0 + 1.0, 25)
    rows: list[dict[str, float | str]] = []
    for la in lat:
        for lo in lon:
            x = (lo - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
            y = (la - lat0) * 111.32
            u = -0.03 * y
            v = 0.03 * x
            rows.append({"time": "2025-03-01T00:00:00", "lat": la, "lon": lo, "u10": u, "v10": v})
    frame = pd.DataFrame(rows)
    out = summarize_polar_features(
        frame,
        lat0=lat0,
        lon0=lon0,
        n_r=48,
        n_theta=96,
        r_max_km=180.0,
    )
    assert bool(out["polar_valid"]) is True
    assert float(out["spiral_score"]) >= 0.0
    assert int(out["dominant_m"]) in {1, 2, 3}
    assert float(out["vt_mean_abs"]) > 0.0

