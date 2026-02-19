from __future__ import annotations

import numpy as np
import pandas as pd

from gka.ops.polar import compute_polar_parity_metrics


def test_compute_polar_parity_metrics_returns_expected_keys() -> None:
    lat0 = -20.0
    lon0 = 150.0
    lat = np.linspace(lat0 - 0.8, lat0 + 0.8, 17)
    lon = np.linspace(lon0 - 0.8, lon0 + 0.8, 17)
    rows: list[dict[str, float | str]] = []
    for la in lat:
        for lo in lon:
            x = (lo - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
            y = (la - lat0) * 111.32
            u = -0.02 * y
            v = 0.02 * x
            rows.append(
                {
                    "time": "2025-03-01T00:00:00",
                    "lat": la,
                    "lon": lo,
                    "u10": u,
                    "v10": v,
                    "u10_mirror": -u,
                    "v10_mirror": v,
                }
            )
    frame = pd.DataFrame(rows)
    out = compute_polar_parity_metrics(frame, lat0=lat0, lon0=lon0, r_max_km=120.0)
    assert "polar_odd_energy_ratio" in out
    assert "eta_parity_polar" in out
    assert float(out["polar_odd_energy_ratio"]) >= 0.0
