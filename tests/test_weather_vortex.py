from __future__ import annotations

import numpy as np
import pandas as pd

from gka.weather.vortex_detect import VortexDetectConfig, detect_vortex_candidates
from gka.weather.vortex_track import VortexTrackConfig, summarize_tracks, track_vortex_candidates


def _synthetic_swirl_frame() -> pd.DataFrame:
    lat0 = -20.0
    lon0 = 150.0
    lat = np.linspace(lat0 - 1.0, lat0 + 1.0, 17)
    lon = np.linspace(lon0 - 1.0, lon0 + 1.0, 17)
    rows: list[dict[str, float | str]] = []
    for la in lat:
        for lo in lon:
            x = (lo - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
            y = (la - lat0) * 111.32
            amp = np.exp(-((x * x + y * y) / (220.0 * 220.0)))
            u = -0.03 * y * amp
            v = 0.03 * x * amp
            rows.append(
                {
                    "time": "2025-03-01T00:00:00",
                    "lat": la,
                    "lon": lo,
                    "u10": u,
                    "v10": v,
                }
            )
    return pd.DataFrame(rows)


def test_detect_vortex_candidates_finds_core() -> None:
    frame = _synthetic_swirl_frame()
    out = detect_vortex_candidates(
        frame,
        config=VortexDetectConfig(
            sigma_cells=1.0,
            zeta_percentile=95.0,
            min_separation_km=50.0,
            ow_threshold=0.5,
            max_candidates_per_time=4,
        ),
    )
    assert not out.empty
    best = out.sort_values("score", ascending=False).iloc[0]
    assert abs(float(best["lat0"]) + 20.0) < 0.4
    assert abs(float(best["lon0"]) - 150.0) < 0.4


def test_track_vortex_candidates_links_points() -> None:
    candidates = pd.DataFrame(
        [
            {"time": "2025-03-01T00:00:00", "lat0": -20.0, "lon0": 150.0, "zeta_peak": 2.0e-4, "ow_median": -1.0, "score": 4.0},
            {"time": "2025-03-01T01:00:00", "lat0": -20.05, "lon0": 150.1, "zeta_peak": 2.1e-4, "ow_median": -1.2, "score": 4.1},
            {"time": "2025-03-01T02:00:00", "lat0": -20.10, "lon0": 150.2, "zeta_peak": 1.9e-4, "ow_median": -1.1, "score": 3.9},
        ]
    )
    tracks = track_vortex_candidates(candidates, config=VortexTrackConfig(max_speed_kmh=500.0, max_gap_hours=2.0))
    assert not tracks.empty
    assert int(tracks["track_id"].nunique()) == 1
    summary = summarize_tracks(tracks, min_duration_hours=1.0, min_points=2)
    assert not summary.empty
    assert bool(summary.iloc[0]["passes_minimums"]) is True
