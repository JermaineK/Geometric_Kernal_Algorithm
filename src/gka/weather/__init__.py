"""Weather-focused discovery and geometry utilities used by minipilots."""

from gka.weather.ibtracs import (
    interpolate_ibtracs_hourly,
    load_ibtracs_points,
    match_tracks_to_ibtracs,
    prepare_ibtracs_catalog,
)
from gka.weather.contract import CoverageContract, TimeBasis, strict_coverage_ok, strict_ib_case_flags, strict_ib_coverage
from gka.weather.polar import summarize_polar_features
from gka.weather.vortex import discover_vortex_tracks, save_vortex_discovery_artifacts
from gka.weather.vortex_detect import detect_vortex_candidates
from gka.weather.vortex_track import select_track_centers, summarize_tracks, track_vortex_candidates

__all__ = [
    "detect_vortex_candidates",
    "track_vortex_candidates",
    "summarize_tracks",
    "select_track_centers",
    "summarize_polar_features",
    "discover_vortex_tracks",
    "save_vortex_discovery_artifacts",
    "load_ibtracs_points",
    "interpolate_ibtracs_hourly",
    "prepare_ibtracs_catalog",
    "match_tracks_to_ibtracs",
    "TimeBasis",
    "CoverageContract",
    "strict_ib_case_flags",
    "strict_ib_coverage",
    "strict_coverage_ok",
]
