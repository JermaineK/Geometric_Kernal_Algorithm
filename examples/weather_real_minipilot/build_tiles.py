from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from gka.external.ibtracs import (
    build_track_index,
    nearest_ibtracs_fix,
    normalize_lon_180,
)
from gka.ops.polar import compute_polar_metrics, compute_polar_parity_metrics
from gka.weather.ibtracs import (
    haversine_km_vec,
    interpolate_ibtracs_hourly,
    load_ibtracs_points,
    match_tracks_to_ibtracs,
    prepare_ibtracs_catalog,
)
from gka.weather.vortex_detect import VortexDetectConfig, detect_vortex_candidates
from gka.weather.vortex_track import VortexTrackConfig, select_track_centers, summarize_tracks, track_vortex_candidates
from gka.utils.time import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build storm-centered canonical tiles from prepared weather data")
    parser.add_argument("--prepared-root", required=True, help="Prepared weather root (lead/date partitions)")
    parser.add_argument("--out", required=True, help="Output canonical dataset directory")
    parser.add_argument(
        "--cohort",
        choices=["all", "events", "background"],
        default="all",
        help="Tile cohort mode: all=events+matched controls, events=event cases only, background=far-nonstorm controls",
    )
    parser.add_argument(
        "--lead-buckets",
        nargs="+",
        default=["24", "120", "240", "none"],
        help="Lead buckets to include (use 'none' for null lead rows)",
    )
    parser.add_argument("--max-events-per-lead", type=int, default=12, help="Maximum storm centers per lead bucket")
    parser.add_argument(
        "--centers",
        choices=["ibtracs", "vortex", "hybrid", "labels"],
        default="hybrid",
        help="Center discovery mode: ibtracs (anchored), vortex (self-discovery), hybrid (ibtracs else vortex), labels (legacy labels)",
    )
    parser.add_argument(
        "--event-source",
        choices=["vortex_or_labels", "vortex", "labels", "ibtracs"],
        default=None,
        help="Deprecated alias for --centers (kept for backward compatibility)",
    )
    parser.add_argument(
        "--ibtracs-csv",
        default=None,
        help="Optional IBTrACS CSV path used for IBTrACS center anchoring and validation",
    )
    parser.add_argument(
        "--ibtracs-hourly-mode",
        choices=["interp", "window"],
        default="interp",
        help="IBTrACS time-linkage mode: interp=hourly interpolated tracks, window=raw 6-hour fixes with tolerance windows.",
    )
    parser.add_argument(
        "--ibtracs-out-dir",
        default="data/external",
        help="Directory for filtered/hourly IBTrACS parquet catalogs",
    )
    parser.add_argument(
        "--ibtracs-time-min",
        default="2025-02-01 00:00:00",
        help="IBTrACS filter start time (UTC-like timestamp)",
    )
    parser.add_argument(
        "--ibtracs-time-max",
        default="2025-05-31 23:00:00",
        help="IBTrACS filter end time (UTC-like timestamp)",
    )
    parser.add_argument(
        "--distance-time-tolerance-hours",
        type=float,
        default=3.0,
        help="Temporal tolerance (hours) when matching rows to nearest IBTrACS point",
    )
    parser.add_argument(
        "--distance-event-km",
        type=float,
        default=300.0,
        help="Distance threshold (km) for event cohort",
    )
    parser.add_argument(
        "--distance-near-km",
        type=float,
        default=800.0,
        help="Distance threshold (km) upper bound for near-storm cohort",
    )
    parser.add_argument(
        "--distance-far-km",
        type=float,
        default=1000.0,
        help="Distance threshold (km) lower bound for far-nonstorm cohort",
    )
    parser.add_argument(
        "--distance-far-loose-km",
        type=float,
        default=2000.0,
        help="Loose distance threshold (km) used with exclusion window for strict far-nonstorm qualification",
    )
    parser.add_argument(
        "--ibtracs-exclusion-window-hours",
        type=float,
        default=48.0,
        help="Far-nonstorm exclusion window (hours). Far requires no storm within this window.",
    )
    parser.add_argument(
        "--ibtracs-dt-max-strict-hours",
        type=float,
        default=1.0,
        help="Strict IBTrACS labels require nearest-fix |dt| <= this threshold (hours).",
    )
    parser.add_argument(
        "--ibtracs-no-storm-implies-far",
        dest="ibtracs_no_storm_implies_far",
        action="store_true",
        help="Treat hours with no storm present in the exclusion window as strict-far candidates (default).",
    )
    parser.add_argument(
        "--ibtracs-no-storm-not-far",
        dest="ibtracs_no_storm_implies_far",
        action="store_false",
        help="Do not mark no-storm-window hours as strict-far.",
    )
    parser.set_defaults(ibtracs_no_storm_implies_far=True)
    parser.add_argument(
        "--soft-event-r0-km",
        type=float,
        default=300.0,
        help="Distance scale (km) used for soft event proximity weight.",
    )
    parser.add_argument(
        "--soft-far-r-km",
        type=float,
        default=1300.0,
        help="Distance scale (km) used for soft far proximity weight.",
    )
    parser.add_argument(
        "--soft-time-td0-hours",
        type=float,
        default=1.5,
        help="Time-confidence scale (hours) used for soft IBTrACS weights.",
    )
    parser.add_argument(
        "--alignment-eta-threshold",
        type=float,
        default=0.10,
        help="Eta threshold used for per-window alignment fraction summaries.",
    )
    parser.add_argument(
        "--alignment-block-hours",
        type=float,
        default=3.0,
        help="Time block size (hours) used for alignment fraction window summaries.",
    )
    parser.add_argument(
        "--ibtracs-use-valid-time-shift",
        action="store_true",
        help="Use t_valid = time + lead_h for IBTrACS matching. Default uses source time directly.",
    )
    parser.add_argument(
        "--vortex-sigma-cells",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma (grid cells) for vorticity peak detection",
    )
    parser.add_argument(
        "--vortex-zeta-percentile",
        type=float,
        default=99.0,
        help="Percentile threshold for |zeta| candidate extraction per time slice",
    )
    parser.add_argument(
        "--vortex-min-separation-km",
        type=float,
        default=180.0,
        help="Minimum separation (km) between vortex candidates in one time slice",
    )
    parser.add_argument(
        "--vortex-ow-threshold",
        type=float,
        default=0.0,
        help="Maximum local OW median for accepted vortex candidates (vortex cores typically OW<0)",
    )
    parser.add_argument(
        "--vortex-max-candidates-per-time",
        type=int,
        default=16,
        help="Maximum vortex candidates retained per timestamp and lead bucket",
    )
    parser.add_argument(
        "--vortex-max-fragments-per-lead",
        type=int,
        default=0,
        help="Optional cap on scanned parquet fragments per lead for fast smoke runs (0 = no cap)",
    )
    parser.add_argument(
        "--track-max-speed-kmh",
        type=float,
        default=300.0,
        help="Max allowed advection speed for candidate-to-track association",
    )
    parser.add_argument(
        "--track-max-gap-hours",
        type=float,
        default=3.0,
        help="Track can remain unmatched for up to this gap before closure",
    )
    parser.add_argument(
        "--track-min-duration-hours",
        type=float,
        default=6.0,
        help="Minimum duration for track quality filtering",
    )
    parser.add_argument(
        "--track-min-points",
        type=int,
        default=4,
        help="Minimum points for track quality filtering",
    )
    parser.add_argument(
        "--control-mode",
        choices=["matched_background", "offset"],
        default="matched_background",
        help="How to build controls for --cohort all",
    )
    parser.add_argument(
        "--control-source",
        choices=["label_background", "all_timeslices"],
        default="all_timeslices",
        help="Background pool source: label_background uses storm/near/pregen==0, all_timeslices uses full lead partition then IB strict-far filtering",
    )
    parser.add_argument(
        "--controls-per-event",
        type=int,
        default=6,
        help="Number of matched far controls to target per event case",
    )
    parser.add_argument(
        "--match-lat-bin-deg",
        type=float,
        default=2.0,
        help="Latitude bin (degrees) used for matched background control sampling",
    )
    parser.add_argument(
        "--match-lon-bin-deg",
        type=float,
        default=2.0,
        help="Longitude bin (degrees) used for matched background control sampling",
    )
    parser.add_argument(
        "--allow-nonexact-controls",
        action="store_true",
        help="Allow fallback non-exact controls when exact match is unavailable",
    )
    parser.add_argument(
        "--require-physical-match",
        action="store_true",
        help="Require matched controls to also match coarse physical bins (speed/zeta) when available",
    )
    parser.add_argument(
        "--physical-speed-bin-ms",
        type=float,
        default=2.0,
        help="Bin width (m/s) for coarse speed matching in control selection",
    )
    parser.add_argument(
        "--physical-zeta-bin",
        type=float,
        default=1e-5,
        help="Bin width for coarse relative-vorticity matching in control selection",
    )
    parser.add_argument(
        "--far-kinematic-speed-max-ms",
        type=float,
        default=18.0,
        help="Maximum coarse speed (m/s) for strict-far controls to qualify as kinematic-clean",
    )
    parser.add_argument(
        "--far-kinematic-zeta-abs-max",
        type=float,
        default=8e-5,
        help="Maximum absolute coarse zeta for strict-far controls to qualify as kinematic-clean",
    )
    parser.add_argument("--pre-hours", type=int, default=240, help="Hours before onset to include")
    parser.add_argument("--post-hours", type=int, default=48, help="Hours after onset to include")
    parser.add_argument("--tile-half-cells", type=int, default=10, help="Half-width in grid cells (10 -> 21x21)")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[5, 9, 13, 17, 21],
        help="Odd spatial scales (in grid cells) used as L",
    )
    parser.add_argument("--grid-step-deg", type=float, default=0.25, help="Grid step in degrees")
    parser.add_argument("--cell-km", type=float, default=25.0, help="Approximate cell width in km for L units")
    parser.add_argument(
        "--min-distinct-scales-per-case",
        type=int,
        default=4,
        help="Minimum distinct L scales required per case after filtering",
    )
    parser.add_argument(
        "--min-controls-per-lead-bucket",
        type=int,
        default=2,
        help="Fail if used control cases per lead bucket falls below this threshold",
    )
    parser.add_argument(
        "--min-far-nonstorm-per-lead-bucket",
        type=int,
        default=2,
        help="Fail if far-nonstorm control cases per lead bucket falls below this threshold",
    )
    parser.add_argument(
        "--adaptive-control-coverage",
        dest="adaptive_control_coverage",
        action="store_true",
        help="Allow coverage gate relaxation only for sparse lead buckets",
    )
    parser.add_argument(
        "--strict-control-coverage",
        dest="adaptive_control_coverage",
        action="store_false",
        help="Disable adaptive relaxation and enforce per-lead control minima strictly",
    )
    parser.set_defaults(adaptive_control_coverage=True)
    parser.add_argument("--lon-offset-control", type=float, default=10.0, help="Longitude shift for matched controls")
    parser.add_argument("--lon-min", type=float, default=125.0, help="Longitude minimum for clamping controls")
    parser.add_argument("--lon-max", type=float, default=175.0, help="Longitude maximum for clamping controls")
    parser.add_argument(
        "--anomaly-mode",
        choices=["none", "lat_hour", "lat_day"],
        default="none",
        help="Optional anomaly removal applied before parity observables are aggregated",
    )
    parser.add_argument(
        "--anomaly-lat-bin-deg",
        type=float,
        default=1.0,
        help="Latitude bin (degrees) used for anomaly baseline grouping",
    )
    parser.add_argument(
        "--anomaly-lon-bin-deg",
        type=float,
        default=2.0,
        help="Longitude bin (degrees) used for anomaly baseline grouping",
    )
    parser.add_argument(
        "--anomaly-commute-rmse-max",
        type=float,
        default=4.0,
        help="Max anomaly commutation RMSE before forcing anomaly_mode=none for a case",
    )
    parser.add_argument(
        "--anomaly-commute-corr-min",
        type=float,
        default=0.05,
        help="Min anomaly commutation correlation before forcing anomaly_mode=none for a case",
    )
    parser.add_argument(
        "--anomaly-min-bin-samples",
        type=int,
        default=24,
        help="Minimum samples per anomaly baseline bin for a bin to be considered sufficient",
    )
    parser.add_argument(
        "--anomaly-min-covered-frac",
        type=float,
        default=0.70,
        help="Minimum row fraction covered by sufficient anomaly bins",
    )
    parser.add_argument(
        "--background-per-lead",
        type=int,
        default=None,
        help="For --cohort background, controls sampled per lead (default: --max-events-per-lead)",
    )
    parser.add_argument(
        "--scan-batch-rows",
        type=int,
        default=200000,
        help="Record-batch row target used for streaming parquet scans",
    )
    parser.add_argument(
        "--background-pool-max-rows",
        type=int,
        default=300000,
        help="Maximum in-memory rows retained per lead for matched-background candidate pool",
    )
    parser.add_argument(
        "--background-max-batches-per-lead",
        type=int,
        default=0,
        help="Optional cap on scanned batches for background pool generation (0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--polar-enable",
        action="store_true",
        help="Compute storm-centered polar features and emit polar_features.parquet",
    )
    parser.add_argument("--polar-r-bins", type=int, default=64, help="Number of radial bins in polar transform")
    parser.add_argument("--polar-theta-bins", type=int, default=128, help="Number of angular bins in polar transform")
    parser.add_argument(
        "--polar-pitches",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.7, 0.7, 1.0, 1.5],
        help="Pitch values used by the log-spiral filter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_root = Path(args.prepared_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")

    dataset = ds.dataset(
        prepared_root,
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    rng = np.random.default_rng(int(args.seed))

    scales = sorted({int(s) for s in args.scales if int(s) >= 3 and int(s) % 2 == 1})
    if not scales:
        raise ValueError("No valid odd scales provided")

    lead_buckets = [str(v) for v in args.lead_buckets]
    cohort = str(args.cohort).lower()
    selected_time_basis = "valid" if bool(args.ibtracs_use_valid_time_shift) else "source"
    centers_requested = _resolve_centers_mode(centers=args.centers, event_source=args.event_source)
    event_source_requested = str(centers_requested)
    event_source_effective = str(event_source_requested)

    events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    control_stats: dict[str, Any] = {}
    vortex_candidates_df = pd.DataFrame()
    storm_tracks_df = pd.DataFrame()
    track_summary_df = pd.DataFrame()
    ibtracs_match: dict[str, Any] | None = None
    ibtracs_catalog: dict[str, Any] | None = None
    ib_points_filtered = pd.DataFrame()
    ib_points_hourly = pd.DataFrame()
    ib_points_match = pd.DataFrame()
    ib_points_centers = pd.DataFrame()
    ib_hourly_mode = str(args.ibtracs_hourly_mode).strip().lower()
    if ib_hourly_mode not in {"interp", "window"}:
        ib_hourly_mode = "interp"
    ib_index = None
    ib_index_meta: dict[str, Any] = {}
    all_cases: list[dict[str, Any]] = []
    if args.ibtracs_csv:
        try:
            ib_out_dir = Path(args.ibtracs_out_dir)
            ib_out_dir.mkdir(parents=True, exist_ok=True)
            t_min = pd.Timestamp(args.ibtracs_time_min)
            t_max = pd.Timestamp(args.ibtracs_time_max)
            ib_points_path = ib_out_dir / "ibtracs_au_2025_FMA.parquet"
            ib_hourly_path = ib_out_dir / "ibtracs_au_2025_FMA_hourly.parquet"
            ibtracs_catalog = prepare_ibtracs_catalog(
                args.ibtracs_csv,
                out_path=ib_points_path,
                hourly_out_path=ib_hourly_path,
                lon_min=float(args.lon_min),
                lon_max=float(args.lon_max),
                lat_min=float(-35.0),
                lat_max=float(-5.0),
                time_min=t_min,
                time_max=t_max,
            )
            ib_points_filtered = pd.read_parquet(ib_points_path)
            ib_points_hourly = pd.read_parquet(ib_hourly_path)
            if ib_hourly_mode == "interp" and not ib_points_hourly.empty:
                ib_points_match = ib_points_hourly.copy()
            else:
                ib_points_match = ib_points_filtered.copy()
            ib_points_centers = ib_points_match if not ib_points_match.empty else (ib_points_hourly if not ib_points_hourly.empty else ib_points_filtered)
            ib_index = _build_ibtracs_index(ib_points_match if not ib_points_match.empty else ib_points_filtered)
            ib_index_meta = _ibtracs_index_meta(ib_index)
            ib_index_meta["ibtracs_hourly_mode"] = str(ib_hourly_mode)
            if not ib_points_hourly.empty:
                ib_points_hourly.to_parquet(out_dir / "ibtracs_hourly_tracks.parquet", index=False)
        except Exception as exc:
            ibtracs_catalog = {"error": f"{type(exc).__name__}: {exc}"}

    if cohort in {"all", "events"}:
        if event_source_requested == "ibtracs":
            if not args.ibtracs_csv:
                raise ValueError("--centers ibtracs requires --ibtracs-csv")
            events = _collect_ibtracs_centers(
                ib_points=ib_points_centers if not ib_points_centers.empty else (ib_points_hourly if not ib_points_hourly.empty else ib_points_filtered),
                lead_buckets=lead_buckets,
                max_events_per_lead=int(args.max_events_per_lead),
                rng=rng,
            )
        elif event_source_requested == "hybrid":
            if not ib_points_hourly.empty or not ib_points_filtered.empty:
                events = _collect_ibtracs_centers(
                    ib_points=ib_points_centers if not ib_points_centers.empty else (ib_points_hourly if not ib_points_hourly.empty else ib_points_filtered),
                    lead_buckets=lead_buckets,
                    max_events_per_lead=int(args.max_events_per_lead),
                    rng=rng,
                )
                event_source_effective = "ibtracs"
            if not events:
                events, vortex_candidates_df, storm_tracks_df, track_summary_df = _collect_vortex_storm_centers(
                    dataset=dataset,
                    lead_buckets=lead_buckets,
                    max_events_per_lead=int(args.max_events_per_lead),
                    detect_cfg=VortexDetectConfig(
                        sigma_cells=float(args.vortex_sigma_cells),
                        zeta_percentile=float(args.vortex_zeta_percentile),
                        min_separation_km=float(args.vortex_min_separation_km),
                        ow_threshold=float(args.vortex_ow_threshold),
                        max_candidates_per_time=int(args.vortex_max_candidates_per_time),
                    ),
                    track_cfg=VortexTrackConfig(
                        max_speed_kmh=float(args.track_max_speed_kmh),
                        max_gap_hours=float(args.track_max_gap_hours),
                    ),
                    min_track_duration_hours=float(args.track_min_duration_hours),
                    min_track_points=int(args.track_min_points),
                    max_fragments_per_lead=int(args.vortex_max_fragments_per_lead),
                    scan_batch_rows=int(args.scan_batch_rows),
                )
                event_source_effective = "vortex" if events else "vortex_none"
            if not events:
                events = _collect_storm_centers(
                    dataset=dataset,
                    lead_buckets=lead_buckets,
                    max_events_per_lead=int(args.max_events_per_lead),
                    speed_bin_ms=float(args.physical_speed_bin_ms),
                    zeta_bin=float(args.physical_zeta_bin),
                    rng=rng,
                    scan_batch_rows=int(args.scan_batch_rows),
                    max_batches_per_lead=int(args.background_max_batches_per_lead),
                )
                if events:
                    event_source_effective = "labels_fallback"
        elif event_source_requested == "vortex":
            events, vortex_candidates_df, storm_tracks_df, track_summary_df = _collect_vortex_storm_centers(
                dataset=dataset,
                lead_buckets=lead_buckets,
                max_events_per_lead=int(args.max_events_per_lead),
                detect_cfg=VortexDetectConfig(
                    sigma_cells=float(args.vortex_sigma_cells),
                    zeta_percentile=float(args.vortex_zeta_percentile),
                    min_separation_km=float(args.vortex_min_separation_km),
                    ow_threshold=float(args.vortex_ow_threshold),
                    max_candidates_per_time=int(args.vortex_max_candidates_per_time),
                ),
                track_cfg=VortexTrackConfig(
                    max_speed_kmh=float(args.track_max_speed_kmh),
                    max_gap_hours=float(args.track_max_gap_hours),
                ),
                min_track_duration_hours=float(args.track_min_duration_hours),
                min_track_points=int(args.track_min_points),
                max_fragments_per_lead=int(args.vortex_max_fragments_per_lead),
                scan_batch_rows=int(args.scan_batch_rows),
            )
        else:
            events = _collect_storm_centers(
                dataset=dataset,
                lead_buckets=lead_buckets,
                max_events_per_lead=int(args.max_events_per_lead),
                speed_bin_ms=float(args.physical_speed_bin_ms),
                zeta_bin=float(args.physical_zeta_bin),
                rng=rng,
                scan_batch_rows=int(args.scan_batch_rows),
                max_batches_per_lead=int(args.background_max_batches_per_lead),
            )
        if not events and cohort == "events":
            raise RuntimeError("No storm centers were found for requested lead buckets")
        for event in events:
            event.setdefault("center_source", str(event_source_effective))
        if (not storm_tracks_df.empty) and (not ib_points_filtered.empty or not ib_points_hourly.empty):
            try:
                ib_points = ib_points_match if not ib_points_match.empty else (ib_points_filtered if not ib_points_filtered.empty else ib_points_hourly)
                ibtracs_match = match_tracks_to_ibtracs(storm_tracks_df, ib_points)
            except Exception as exc:
                ibtracs_match = {"error": f"{type(exc).__name__}: {exc}"}
    if cohort == "all":
        if str(args.control_mode) == "matched_background":
            controls, control_stats = _make_matched_background_controls(
                dataset=dataset,
                events=events,
                rng=rng,
                lat_bin_deg=float(args.match_lat_bin_deg),
                lon_bin_deg=float(args.match_lon_bin_deg),
                allow_nonexact=bool(args.allow_nonexact_controls),
                require_physical_match=bool(args.require_physical_match),
                speed_bin_ms=float(args.physical_speed_bin_ms),
                zeta_bin=float(args.physical_zeta_bin),
                lon_offset=float(args.lon_offset_control),
                lon_min=float(args.lon_min),
                lon_max=float(args.lon_max),
                scan_batch_rows=int(args.scan_batch_rows),
                pool_max_rows=int(args.background_pool_max_rows),
                max_batches_per_lead=int(args.background_max_batches_per_lead),
                controls_per_event=int(args.controls_per_event),
                control_source=str(args.control_source),
            )
        else:
            controls = _make_offset_controls(
                events=events,
                lon_offset=float(args.lon_offset_control),
                lon_min=float(args.lon_min),
                lon_max=float(args.lon_max),
            )
            control_stats = {
                "requested": int(len(events)),
                "matched": int(len(controls)),
                "dropped": int(max(0, len(events) - len(controls))),
                "mode": "offset",
            }
        for control in controls:
            control.setdefault("center_source", "background")
        if ib_index is not None:
            events = _annotate_cases_with_storm_distance(
                events,
                ib_index=ib_index,
                tolerance_hours=float(args.distance_time_tolerance_hours),
                event_km=float(args.distance_event_km),
                near_km=float(args.distance_near_km),
                far_km=float(args.distance_far_km),
                far_loose_km=float(args.distance_far_loose_km),
                exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
            )
            controls = _annotate_cases_with_storm_distance(
                controls,
                ib_index=ib_index,
                tolerance_hours=float(args.distance_time_tolerance_hours),
                event_km=float(args.distance_event_km),
                near_km=float(args.distance_near_km),
                far_km=float(args.distance_far_km),
                far_loose_km=float(args.distance_far_loose_km),
                exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
            )
            before = len(controls)
            controls = [c for c in controls if bool(c.get("ib_far_strict", c.get("ib_far", False)))]
            control_stats = dict(control_stats or {})
            control_stats["dropped_not_far_nonstorm"] = int(max(0, before - len(controls)))
            # Full-period IBTrACS anchoring can leave matched controls too close to storms.
            # Fallback to time-sampled background centers so far_nonstorm coverage remains evaluable.
            if (not controls) and bool(events):
                fallback_controls_raw = _collect_background_centers(
                    dataset=dataset,
                    lead_buckets=lead_buckets,
                    max_per_lead=max(1, int(args.max_events_per_lead)),
                    rng=rng,
                    scan_batch_rows=int(args.scan_batch_rows),
                    max_batches_per_lead=int(args.background_max_batches_per_lead),
                    control_source=str(args.control_source),
                )
                for control in fallback_controls_raw:
                    control.setdefault("center_source", "background")
                    control.setdefault("case_type", "control")
                    control.setdefault("control_tier", "C")
                    control.setdefault("match_quality", "tier_c_fallback_background")
                fallback_controls = _annotate_cases_with_storm_distance(
                    fallback_controls_raw,
                    ib_index=ib_index,
                    tolerance_hours=float(args.distance_time_tolerance_hours),
                    event_km=float(args.distance_event_km),
                    near_km=float(args.distance_near_km),
                    far_km=float(args.distance_far_km),
                    far_loose_km=float(args.distance_far_loose_km),
                    exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                    use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                    far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                    far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
                )
                fallback_controls = [c for c in fallback_controls if bool(c.get("ib_far_strict", c.get("ib_far", False)))]
                controls.extend(fallback_controls)
                control_stats["fallback_background_requested"] = int(len(fallback_controls_raw))
                control_stats["fallback_background_used"] = int(len(fallback_controls))
            # If strict-far filtering left some leads uncovered, augment with additional matched strict-far controls.
            target_floor = int(max(int(args.min_controls_per_lead_bucket), int(args.min_far_nonstorm_per_lead_bucket), 1))
            event_count_by_lead: dict[str, int] = {}
            for ev in events:
                key = str(ev.get("lead_bucket", ""))
                event_count_by_lead[key] = event_count_by_lead.get(key, 0) + 1
            target_by_lead: dict[str, int] = {}
            for lead in [str(v) for v in lead_buckets]:
                ecount = int(event_count_by_lead.get(str(lead), 0))
                target_by_lead[str(lead)] = int(max(target_floor, int(args.controls_per_event) * max(ecount, 1 if ecount > 0 else 0)))

            if target_by_lead:
                have_by_lead: dict[str, int] = {}
                for c in controls:
                    if not bool(c.get("ib_far_strict", c.get("ib_far", False))):
                        continue
                    lead_key = str(c.get("lead_bucket", ""))
                    have_by_lead[lead_key] = have_by_lead.get(lead_key, 0) + 1
                missing_leads = [
                    str(lead)
                    for lead in [str(v) for v in lead_buckets]
                    if int(have_by_lead.get(str(lead), 0)) < int(target_by_lead.get(str(lead), target_floor))
                ]
                if missing_leads:
                    extra_raw = _collect_background_centers(
                        dataset=dataset,
                        lead_buckets=missing_leads,
                        max_per_lead=max(
                            int(args.max_events_per_lead) * 24,
                            int(max(target_by_lead.values() or [target_floor])) * 20,
                            512,
                        ),
                        rng=rng,
                        scan_batch_rows=int(args.scan_batch_rows),
                        max_batches_per_lead=int(args.background_max_batches_per_lead),
                        control_source=str(args.control_source),
                    )
                    for control in extra_raw:
                        control.setdefault("center_source", "background")
                        control.setdefault("case_type", "control")
                        control.setdefault("control_tier", "C")
                        control.setdefault("match_quality", "tier_c_augmented_background")
                    extra_ann = _annotate_cases_with_storm_distance(
                        extra_raw,
                        ib_index=ib_index,
                        tolerance_hours=float(args.distance_time_tolerance_hours),
                        event_km=float(args.distance_event_km),
                        near_km=float(args.distance_near_km),
                        far_km=float(args.distance_far_km),
                        far_loose_km=float(args.distance_far_loose_km),
                        exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                        use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                        far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                        far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
                    )
                    extra_ann = [c for c in extra_ann if bool(c.get("ib_far_strict", c.get("ib_far", False)))]
                    add_by_lead: dict[str, list[dict[str, Any]]] = {}
                    for c in extra_ann:
                        add_by_lead.setdefault(str(c.get("lead_bucket", "")), []).append(c)
                    added = 0
                    for lead in missing_leads:
                        need = max(0, int(target_by_lead.get(str(lead), target_floor)) - int(have_by_lead.get(str(lead), 0)))
                        if need <= 0:
                            continue
                        candidates = add_by_lead.get(str(lead), [])
                        if not candidates:
                            continue
                        pool_rows: list[dict[str, Any]] = []
                        for idx_c, cand in enumerate(candidates):
                            t0 = pd.Timestamp(cand.get("time0"))
                            pool_rows.append(
                                {
                                    "case_ref_idx": int(idx_c),
                                    "time": t0,
                                    "lat": float(cand.get("lat0", np.nan)),
                                    "lon": float(cand.get("lon0", np.nan)),
                                    "month": int(t0.month) if pd.notna(t0) else -1,
                                    "hour": int(t0.hour) if pd.notna(t0) else -1,
                                    "lat_bin": float(_lat_bin(pd.Series([float(cand.get("lat0", np.nan))]), float(args.match_lat_bin_deg)).iloc[0])
                                    if np.isfinite(float(cand.get("lat0", np.nan)))
                                    else np.nan,
                                    "lon_bin": float(_lon_bin(pd.Series([float(cand.get("lon0", np.nan))]), float(args.match_lon_bin_deg)).iloc[0])
                                    if np.isfinite(float(cand.get("lon0", np.nan)))
                                    else np.nan,
                                    "speed_bin": cand.get("speed_bin"),
                                    "zeta_bin": cand.get("zeta_bin"),
                                }
                            )
                        pool_df = pd.DataFrame(pool_rows)
                        pool_df = pool_df.dropna(subset=["time", "lat", "lon", "lat_bin", "lon_bin"]).reset_index(drop=True)
                        if pool_df.empty:
                            continue
                        used_idx: set[int] = set()
                        lead_events = [e for e in events if str(e.get("lead_bucket", "")) == str(lead)]
                        cursor = 0
                        max_iter = max(int(need) * 30, 200)
                        iter_count = 0
                        while need > 0 and iter_count < max_iter:
                            iter_count += 1
                            if lead_events:
                                event_ref = lead_events[cursor % len(lead_events)]
                                cursor += 1
                                pick, quality = _pick_matched_background_row(
                                    pool=pool_df,
                                    event=event_ref,
                                    used_idx=used_idx,
                                    rng=rng,
                                    lat_bin_deg=float(args.match_lat_bin_deg),
                                    lon_bin_deg=float(args.match_lon_bin_deg),
                                    allow_nonexact=True,
                                    require_physical_match=False,
                                )
                            else:
                                pick_idx = int(rng.integers(0, pool_df.shape[0]))
                                pick = pool_df.iloc[pick_idx]
                                quality = "tier_c_augmented_background"
                            if pick is None:
                                break
                            ref_idx = int(pick.get("case_ref_idx", -1))
                            if ref_idx < 0 or ref_idx >= len(candidates):
                                continue
                            new_ctrl = dict(candidates[ref_idx])
                            q = str(quality or "tier_c_augmented_background")
                            new_ctrl["case_id"] = f"bgm_lead{lead}_aug_{int(have_by_lead.get(str(lead), 0)) + 1:04d}_{added + 1:05d}"
                            new_ctrl["case_type"] = "control"
                            new_ctrl["event_label"] = "background_matched_strict_far"
                            new_ctrl["center_source"] = "background"
                            new_ctrl["storm_id"] = f"background_matched_strict_far_{lead}_{added + 1:05d}"
                            new_ctrl["match_quality"] = q if q != "none" else "tier_c_augmented_background"
                            new_ctrl["control_tier"] = _control_tier(new_ctrl["match_quality"])
                            controls.append(new_ctrl)
                            have_by_lead[str(lead)] = int(have_by_lead.get(str(lead), 0)) + 1
                            need -= 1
                            added += 1
                    control_stats["augmented_background_requested"] = int(len(extra_raw))
                    control_stats["augmented_background_used"] = int(added)
                    control_stats["augmented_target_by_lead"] = {str(k): int(v) for k, v in sorted(target_by_lead.items())}
                    control_stats["augmented_have_by_lead"] = {str(k): int(v) for k, v in sorted(have_by_lead.items())}
        all_cases = events + controls
    elif cohort == "events":
        if ib_index is not None:
            events = _annotate_cases_with_storm_distance(
                events,
                ib_index=ib_index,
                tolerance_hours=float(args.distance_time_tolerance_hours),
                event_km=float(args.distance_event_km),
                near_km=float(args.distance_near_km),
                far_km=float(args.distance_far_km),
                far_loose_km=float(args.distance_far_loose_km),
                exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
            )
        all_cases = events
    elif cohort == "background":
        controls = _collect_background_centers(
            dataset=dataset,
            lead_buckets=lead_buckets,
            max_per_lead=int(args.background_per_lead or args.max_events_per_lead),
            rng=rng,
            scan_batch_rows=int(args.scan_batch_rows),
            max_batches_per_lead=int(args.background_max_batches_per_lead),
            control_source=str(args.control_source),
        )
        for control in controls:
            control.setdefault("center_source", "background")
        if ib_index is not None:
            controls = _annotate_cases_with_storm_distance(
                controls,
                ib_index=ib_index,
                tolerance_hours=float(args.distance_time_tolerance_hours),
                event_km=float(args.distance_event_km),
                near_km=float(args.distance_near_km),
                far_km=float(args.distance_far_km),
                far_loose_km=float(args.distance_far_loose_km),
                exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
                use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
                far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
                far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
            )
            controls = [c for c in controls if bool(c.get("ib_far_strict", c.get("ib_far", False)))]
        control_stats = {"mode": "background_only"}
        all_cases = controls
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    if not all_cases:
        raise RuntimeError(f"No cases were produced for cohort={cohort}")

    rows: list[dict[str, Any]] = []
    polar_rows: list[dict[str, Any]] = []
    case_manifest: list[dict[str, Any]] = []
    case_manifest_all: list[dict[str, Any]] = []
    anomaly_audit_rows: list[dict[str, Any]] = []
    for case in all_cases:
        case_rows, case_polar_rows, case_audit = _extract_case_rows(
            dataset=dataset,
            case=case,
            pre_hours=int(args.pre_hours),
            post_hours=int(args.post_hours),
            tile_half_cells=int(args.tile_half_cells),
            grid_step_deg=float(args.grid_step_deg),
            scales=scales,
            cell_km=float(args.cell_km),
            anomaly_mode=str(args.anomaly_mode),
            anomaly_lat_bin_deg=float(args.anomaly_lat_bin_deg),
            anomaly_lon_bin_deg=float(args.anomaly_lon_bin_deg),
            anomaly_commute_rmse_max=float(args.anomaly_commute_rmse_max),
            anomaly_commute_corr_min=float(args.anomaly_commute_corr_min),
            anomaly_min_bin_samples=int(args.anomaly_min_bin_samples),
            anomaly_min_covered_frac=float(args.anomaly_min_covered_frac),
            min_distinct_scales=int(args.min_distinct_scales_per_case),
            polar_enable=bool(args.polar_enable),
            polar_r_bins=int(args.polar_r_bins),
            polar_theta_bins=int(args.polar_theta_bins),
            polar_pitches=[float(v) for v in args.polar_pitches],
        )
        case_meta = dict(case)
        case_meta.update(case_audit)
        case_meta["rows_generated"] = int(len(case_rows))
        case_meta["excluded"] = bool(len(case_rows) == 0)
        case_manifest_all.append(case_meta)
        anomaly_audit_rows.append(case_audit)
        if not case_rows:
            continue
        rows.extend(case_rows)
        polar_rows.extend(case_polar_rows)
        case_manifest.append(case_meta)

    if not rows:
        raise RuntimeError("No tile rows were generated")

    samples = pd.DataFrame(rows)
    samples["t"] = pd.to_datetime(samples["t"], errors="coerce")
    samples = samples.sort_values(["case_id", "t", "L", "hand"]).reset_index(drop=True)
    if ib_index is not None:
        samples = _annotate_samples_with_storm_distance(
            samples=samples,
            ib_index=ib_index,
            tolerance_hours=float(args.distance_time_tolerance_hours),
            event_km=float(args.distance_event_km),
            near_km=float(args.distance_near_km),
            far_km=float(args.distance_far_km),
            far_loose_km=float(args.distance_far_loose_km),
            exclusion_window_hours=float(args.ibtracs_exclusion_window_hours),
            use_valid_time_shift=bool(args.ibtracs_use_valid_time_shift),
            far_kinematic_speed_max_ms=float(args.far_kinematic_speed_max_ms),
            far_kinematic_zeta_abs_max=float(args.far_kinematic_zeta_abs_max),
                strict_dt_max_hours=float(args.ibtracs_dt_max_strict_hours),
                soft_event_r0_km=float(args.soft_event_r0_km),
                soft_far_r_km=float(args.soft_far_r_km),
                soft_time_td0_hours=float(args.soft_time_td0_hours),
                no_storm_implies_far=bool(args.ibtracs_no_storm_implies_far),
        )
    else:
        samples["nearest_storm_distance_km"] = np.nan
        samples["nearest_storm_id"] = None
        samples["nearest_storm_time_delta_h"] = np.nan
        samples["nearest_storm_distance_km_source"] = np.nan
        samples["nearest_storm_distance_km_valid"] = np.nan
        samples["nearest_storm_time_delta_h_source"] = np.nan
        samples["nearest_storm_time_delta_h_valid"] = np.nan
        samples["nearest_storm_dt_fix_h_source"] = np.nan
        samples["nearest_storm_dt_fix_h_valid"] = np.nan
        samples["nearest_storm_speed_kmh_source"] = np.nan
        samples["nearest_storm_speed_kmh_valid"] = np.nan
        samples["nearest_storm_center_uncert_km_source"] = np.nan
        samples["nearest_storm_center_uncert_km_valid"] = np.nan
        samples["nearest_storm_min_dist_window_km"] = np.nan
        samples["nearest_storm_min_dist_window_km_source"] = np.nan
        samples["nearest_storm_min_dist_window_km_valid"] = np.nan
        samples["ib_min_dist_any_storm_km"] = np.nan
        samples["ib_min_dist_any_storm_km_source"] = np.nan
        samples["ib_min_dist_any_storm_km_valid"] = np.nan
        samples["storm_distance_cohort"] = None
        samples["storm_phase"] = None
        samples["storm_dist_km"] = np.nan
        samples["nearest_storm_lat"] = np.nan
        samples["nearest_storm_lon"] = np.nan
        samples["nearest_storm_fix_time"] = pd.NaT
        samples["nearest_storm_id_source"] = None
        samples["nearest_storm_id_valid"] = None
        samples["ib_sid"] = None
        samples["ib_name"] = None
        samples["ib_basin"] = None
        samples["ib_dt_hours"] = np.nan
        samples["ib_dt_hours_source"] = np.nan
        samples["ib_dt_hours_valid"] = np.nan
        samples["ib_dt_to_fix_hours_source"] = np.nan
        samples["ib_dt_to_fix_hours_valid"] = np.nan
        samples["ib_dt_to_fix_hours"] = np.nan
        samples["ib_speed_kmh_source"] = np.nan
        samples["ib_speed_kmh_valid"] = np.nan
        samples["ib_center_uncert_km_source"] = np.nan
        samples["ib_center_uncert_km_valid"] = np.nan
        samples["storm_center_uncert_km"] = np.nan
        samples["dt_to_nearest_ib_hours"] = np.nan
        samples["ib_has_fix_window"] = False
        samples["ib_dist_km"] = np.nan
        samples["ib_dist_km_source"] = np.nan
        samples["ib_dist_km_valid"] = np.nan
        samples["w_event"] = 0.0
        samples["w_far"] = 0.0
        samples["eventness"] = 0.0
        samples["farness"] = 0.0
        samples["ib_event"] = False
        samples["ib_event_source"] = False
        samples["ib_event_valid"] = False
        samples["ib_far"] = False
        samples["ib_far_source"] = False
        samples["ib_far_valid"] = False
        samples["ib_event_strict"] = False
        samples["ib_event_strict_source"] = False
        samples["ib_event_strict_valid"] = False
        samples["ib_far_strict"] = False
        samples["ib_far_strict_source"] = False
        samples["ib_far_strict_valid"] = False
        samples["ib_far_quality_tag"] = "no_ibtracs_index"
        samples["ib_kinematic_clean"] = None
        samples["ib_conflict_flag"] = None
        samples["ib_strict_dt_max_hours"] = float(args.ibtracs_dt_max_strict_hours)
        samples["time_basis"] = selected_time_basis
        samples["ib_time_basis_used"] = selected_time_basis
        samples["t_source"] = pd.to_datetime(samples["t"], errors="coerce")
        samples["t_valid"] = pd.to_datetime(samples["t"], errors="coerce")
        samples["storm_match_time"] = samples["t_valid"] if selected_time_basis == "valid" else samples["t_source"]

    control_coverage = _compute_control_coverage(samples=samples)
    lead_case_counts = _lead_case_counts(samples=samples)
    coverage_relaxation: dict[str, Any] = {
        "adaptive_enabled": bool(args.adaptive_control_coverage),
        "relaxed_control_leads": [],
        "relaxed_far_nonstorm_leads": [],
    }
    min_controls = int(args.min_controls_per_lead_bucket)
    min_far = int(args.min_far_nonstorm_per_lead_bucket)
    if min_controls > 0:
        lead_targets = [str(v) for v in args.lead_buckets]
        missing: dict[str, int] = {}
        for lead in lead_targets:
            cnt = int((control_coverage.get("control_cases_by_lead") or {}).get(str(lead), 0))
            if cnt >= min_controls:
                continue
            sparse = int((lead_case_counts or {}).get(str(lead), 0)) < min_controls
            if bool(args.adaptive_control_coverage) and sparse:
                coverage_relaxation["relaxed_control_leads"].append(str(lead))
                continue
            missing[str(lead)] = cnt
        if missing:
            raise ValueError(
                "insufficient_control_coverage:"
                + ",".join(f"{lead}:{cnt}<{min_controls}" for lead, cnt in sorted(missing.items()))
            )
    if min_far > 0:
        lead_targets = [str(v) for v in args.lead_buckets]
        missing: dict[str, int] = {}
        for lead in lead_targets:
            cnt = int((control_coverage.get("far_nonstorm_controls_by_lead") or {}).get(str(lead), 0))
            if cnt >= min_far:
                continue
            sparse = int((lead_case_counts or {}).get(str(lead), 0)) < min_far
            if bool(args.adaptive_control_coverage) and sparse:
                coverage_relaxation["relaxed_far_nonstorm_leads"].append(str(lead))
                continue
            missing[str(lead)] = cnt
        if missing:
            raise ValueError(
                "insufficient_far_nonstorm_coverage:"
                + ",".join(f"{lead}:{cnt}<{min_far}" for lead, cnt in sorted(missing.items()))
            )

    ib_diag_df, ib_diag_summary = _build_ibtracs_alignment_diagnostics(
        samples=samples,
        case_manifest_all=case_manifest_all,
    )
    ib_contract_cases = _build_ibtracs_contract_cases(ib_diag_df)
    contract_sample_cols = [
        c
        for c in (
            "case_id",
            "lead_bucket",
            "case_type",
            "center_source",
            "control_tier",
            "match_quality",
            "t_source",
            "t_valid",
            "storm_match_time",
            "time_basis",
            "ib_time_basis_used",
            "lat0",
            "lon0",
            "nearest_storm_id",
            "nearest_storm_id_source",
            "nearest_storm_id_valid",
            "nearest_storm_distance_km",
            "nearest_storm_distance_km_source",
            "nearest_storm_distance_km_valid",
            "nearest_storm_time_delta_h",
            "nearest_storm_time_delta_h_source",
            "nearest_storm_time_delta_h_valid",
            "nearest_storm_dt_fix_h_source",
            "nearest_storm_dt_fix_h_valid",
            "nearest_storm_speed_kmh_source",
            "nearest_storm_speed_kmh_valid",
            "nearest_storm_center_uncert_km_source",
            "nearest_storm_center_uncert_km_valid",
            "nearest_storm_min_dist_window_km",
            "nearest_storm_min_dist_window_km_source",
            "nearest_storm_min_dist_window_km_valid",
            "ib_min_dist_any_storm_km",
            "ib_min_dist_any_storm_km_source",
            "ib_min_dist_any_storm_km_valid",
            "ib_event",
            "ib_far",
            "ib_event_strict",
            "ib_far_strict",
            "ib_event_source",
            "ib_far_source",
            "ib_event_valid",
            "ib_far_valid",
            "ib_far_quality_tag",
            "ib_kinematic_clean",
            "ib_conflict_flag",
            "dt_to_nearest_ib_hours",
            "ib_dt_to_fix_hours",
            "ib_dt_to_fix_hours_source",
            "ib_dt_to_fix_hours_valid",
            "ib_speed_kmh_source",
            "ib_speed_kmh_valid",
            "ib_center_uncert_km_source",
            "ib_center_uncert_km_valid",
            "storm_center_uncert_km",
            "ib_has_fix_window",
            "w_event",
            "w_far",
            "eventness",
            "farness",
        )
        if c in samples.columns
    ]
    ib_contract_samples = samples[contract_sample_cols].copy() if contract_sample_cols else pd.DataFrame()
    time_basis_audit = _build_time_basis_audit(ib_contract_cases)
    center_definition_audit = _build_center_definition_audit(ib_contract_cases)
    cohort_audit = _build_cohort_audit(ib_contract_cases)
    case_windows = _build_case_windows(
        samples,
        eta_threshold=float(args.alignment_eta_threshold),
        block_hours=float(args.alignment_block_hours),
    )
    case_contract = ib_contract_cases.copy()
    if not case_windows.empty and "case_id" in case_windows.columns:
        case_contract = case_contract.merge(
            case_windows[
                [
                    c
                    for c in (
                        "case_id",
                        "eventness",
                        "farness",
                        "A_align",
                        "S_align",
                        "A_align_event",
                        "A_align_far",
                        "S_align_event",
                        "S_align_far",
                        "n_time_steps",
                        "n_blocks",
                    )
                    if c in case_windows.columns
                ]
            ],
            on="case_id",
            how="left",
        )

    samples.to_parquet(out_dir / "samples.parquet", index=False)
    eta_long = _build_eta_long(samples)
    eta_long.to_parquet(out_dir / "eta_long.parquet", index=False)
    case_windows.to_parquet(out_dir / "case_windows.parquet", index=False)
    case_contract.to_parquet(out_dir / "case_contract.parquet", index=False)
    strict_coverage_weekly = _build_strict_coverage_by_week(case_contract)
    strict_coverage_weekly.to_csv(out_dir / "strict_coverage_by_week.csv", index=False)
    polar_features = pd.DataFrame(polar_rows)
    if not polar_features.empty:
        polar_features["t"] = pd.to_datetime(polar_features["t"], errors="coerce")
        polar_features = polar_features.sort_values(["case_id", "t", "L"]).reset_index(drop=True)
    polar_features.to_parquet(out_dir / "polar_features.parquet", index=False)
    if not vortex_candidates_df.empty:
        vortex_candidates_df.to_parquet(out_dir / "vortex_candidates.parquet", index=False)
    if not storm_tracks_df.empty:
        storm_tracks_df.to_parquet(out_dir / "storm_tracks.parquet", index=False)
        derived_dir = Path("data/derived")
        derived_dir.mkdir(parents=True, exist_ok=True)
        storm_tracks_df.to_parquet(derived_dir / "vortex_tracks_FMA_2025.parquet", index=False)
    if not track_summary_df.empty:
        track_summary_df.to_parquet(out_dir / "storm_track_summary.parquet", index=False)
    if ibtracs_match is not None:
        (out_dir / "ibtracs_match.json").write_text(json.dumps(ibtracs_match, indent=2, sort_keys=True), encoding="utf-8")
    ib_diag_df.to_parquet(out_dir / "ibtracs_alignment_diagnostics.parquet", index=False)
    (out_dir / "ibtracs_alignment_summary.json").write_text(
        json.dumps(ib_diag_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ib_contract_cases.to_parquet(out_dir / "ibtracs_contract.parquet", index=False)
    ib_contract_samples.to_parquet(out_dir / "ibtracs_contract_samples.parquet", index=False)
    (out_dir / "time_basis_audit.json").write_text(json.dumps(time_basis_audit, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "center_definition_audit.json").write_text(
        json.dumps(center_definition_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "cohort_audit.json").write_text(json.dumps(cohort_audit, indent=2, sort_keys=True), encoding="utf-8")

    dataset_yaml = {
        "schema_version": 1,
        "domain": "weather",
        "id": "weather_real_vnext_tiles",
        "description": "Storm-centered real-weather tile aggregates with mirror and polar diagnostics",
        "units": {"time": "UTC timestamp", "L": "km", "omega": "rad/s"},
        "mirror": {
            "type": "spatial_reflection+polar_angle_reflection",
            "details": {"axis": "lon", "lon0": 150.0, "polar_theta_mirror": "theta->-theta"},
        },
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "preprocessing": {
            "anomaly_mode": str(args.anomaly_mode),
            "anomaly_lat_bin_deg": float(args.anomaly_lat_bin_deg),
            "anomaly_lon_bin_deg": float(args.anomaly_lon_bin_deg),
            "anomaly_commute_rmse_max": float(args.anomaly_commute_rmse_max),
            "anomaly_commute_corr_min": float(args.anomaly_commute_corr_min),
            "anomaly_min_bin_samples": int(args.anomaly_min_bin_samples),
            "anomaly_min_covered_frac": float(args.anomaly_min_covered_frac),
            "control_mode": str(args.control_mode),
            "control_source": str(args.control_source),
            "controls_per_event": int(args.controls_per_event),
            "match_lat_bin_deg": float(args.match_lat_bin_deg),
            "match_lon_bin_deg": float(args.match_lon_bin_deg),
            "allow_nonexact_controls": bool(args.allow_nonexact_controls),
            "require_physical_match": bool(args.require_physical_match),
            "physical_speed_bin_ms": float(args.physical_speed_bin_ms),
            "physical_zeta_bin": float(args.physical_zeta_bin),
            "min_distinct_scales_per_case": int(args.min_distinct_scales_per_case),
            "min_controls_per_lead_bucket": int(args.min_controls_per_lead_bucket),
            "min_far_nonstorm_per_lead_bucket": int(args.min_far_nonstorm_per_lead_bucket),
            "adaptive_control_coverage": bool(args.adaptive_control_coverage),
            "event_source_requested": str(event_source_requested),
            "event_source_effective": str(event_source_effective),
            "centers_mode": str(event_source_requested),
            "vortex_sigma_cells": float(args.vortex_sigma_cells),
            "vortex_zeta_percentile": float(args.vortex_zeta_percentile),
            "vortex_min_separation_km": float(args.vortex_min_separation_km),
            "vortex_ow_threshold": float(args.vortex_ow_threshold),
            "vortex_max_candidates_per_time": int(args.vortex_max_candidates_per_time),
            "vortex_max_fragments_per_lead": int(args.vortex_max_fragments_per_lead),
            "scan_batch_rows": int(args.scan_batch_rows),
            "background_pool_max_rows": int(args.background_pool_max_rows),
            "background_max_batches_per_lead": int(args.background_max_batches_per_lead),
            "polar_enable": bool(args.polar_enable),
            "polar_r_bins": int(args.polar_r_bins),
            "polar_theta_bins": int(args.polar_theta_bins),
            "polar_pitches": [float(v) for v in args.polar_pitches],
            "distance_time_tolerance_hours": float(args.distance_time_tolerance_hours),
            "distance_event_km": float(args.distance_event_km),
            "distance_near_km": float(args.distance_near_km),
            "distance_far_km": float(args.distance_far_km),
            "distance_far_loose_km": float(args.distance_far_loose_km),
            "ibtracs_exclusion_window_hours": float(args.ibtracs_exclusion_window_hours),
            "ibtracs_dt_max_strict_hours": float(args.ibtracs_dt_max_strict_hours),
            "soft_event_r0_km": float(args.soft_event_r0_km),
            "soft_far_r_km": float(args.soft_far_r_km),
            "soft_time_td0_hours": float(args.soft_time_td0_hours),
            "ibtracs_use_valid_time_shift": bool(args.ibtracs_use_valid_time_shift),
            "ibtracs_hourly_mode": str(ib_hourly_mode),
            "ibtracs_no_storm_implies_far": bool(args.ibtracs_no_storm_implies_far),
            "time_basis": selected_time_basis,
            "far_kinematic_speed_max_ms": float(args.far_kinematic_speed_max_ms),
            "far_kinematic_zeta_abs_max": float(args.far_kinematic_zeta_abs_max),
            "alignment_eta_threshold": float(args.alignment_eta_threshold),
            "alignment_block_hours": float(args.alignment_block_hours),
        },
        "analysis": {
            "knee": {"method": "segmented", "rho": 1.5},
            "scaling": {"method": "wls", "exclude_forbidden": True},
            "stability": {"b": 2.0},
            "impedance": {"enabled": True, "tolerance": 0.15},
        },
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    pd.DataFrame(case_manifest).to_parquet(out_dir / "case_manifest.parquet", index=False)
    pd.DataFrame(case_manifest_all).to_parquet(out_dir / "case_manifest_all.parquet", index=False)
    scale_audit = _build_scale_audit(case_manifest_all)
    (out_dir / "scale_audit.json").write_text(json.dumps(scale_audit, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "created_at_utc": utc_now_iso(),
        "prepared_root": str(prepared_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "cohort": cohort,
        "event_source_requested": str(event_source_requested),
        "event_source_effective": str(event_source_effective),
        "centers_mode": str(event_source_requested),
        "vortex_sigma_cells": float(args.vortex_sigma_cells),
        "vortex_zeta_percentile": float(args.vortex_zeta_percentile),
        "vortex_min_separation_km": float(args.vortex_min_separation_km),
        "vortex_ow_threshold": float(args.vortex_ow_threshold),
        "vortex_max_candidates_per_time": int(args.vortex_max_candidates_per_time),
        "vortex_max_fragments_per_lead": int(args.vortex_max_fragments_per_lead),
        "scan_batch_rows": int(args.scan_batch_rows),
        "background_pool_max_rows": int(args.background_pool_max_rows),
        "background_max_batches_per_lead": int(args.background_max_batches_per_lead),
        "control_mode": str(args.control_mode),
        "control_source": str(args.control_source),
        "controls_per_event": int(args.controls_per_event),
        "allow_nonexact_controls": bool(args.allow_nonexact_controls),
        "ibtracs_csv": str(args.ibtracs_csv) if args.ibtracs_csv else None,
        "anomaly_mode": str(args.anomaly_mode),
        "anomaly_lat_bin_deg": float(args.anomaly_lat_bin_deg),
        "anomaly_lon_bin_deg": float(args.anomaly_lon_bin_deg),
        "anomaly_commute_rmse_max": float(args.anomaly_commute_rmse_max),
        "anomaly_commute_corr_min": float(args.anomaly_commute_corr_min),
        "anomaly_min_bin_samples": int(args.anomaly_min_bin_samples),
        "anomaly_min_covered_frac": float(args.anomaly_min_covered_frac),
        "match_lat_bin_deg": float(args.match_lat_bin_deg),
        "match_lon_bin_deg": float(args.match_lon_bin_deg),
        "require_physical_match": bool(args.require_physical_match),
        "distance_time_tolerance_hours": float(args.distance_time_tolerance_hours),
        "distance_event_km": float(args.distance_event_km),
        "distance_near_km": float(args.distance_near_km),
        "distance_far_km": float(args.distance_far_km),
        "distance_far_loose_km": float(args.distance_far_loose_km),
        "ibtracs_exclusion_window_hours": float(args.ibtracs_exclusion_window_hours),
        "ibtracs_dt_max_strict_hours": float(args.ibtracs_dt_max_strict_hours),
        "soft_event_r0_km": float(args.soft_event_r0_km),
        "soft_far_r_km": float(args.soft_far_r_km),
        "soft_time_td0_hours": float(args.soft_time_td0_hours),
        "ibtracs_use_valid_time_shift": bool(args.ibtracs_use_valid_time_shift),
        "ibtracs_hourly_mode": str(ib_hourly_mode),
        "ibtracs_no_storm_implies_far": bool(args.ibtracs_no_storm_implies_far),
        "time_basis": selected_time_basis,
        "far_kinematic_speed_max_ms": float(args.far_kinematic_speed_max_ms),
        "far_kinematic_zeta_abs_max": float(args.far_kinematic_zeta_abs_max),
        "alignment_eta_threshold": float(args.alignment_eta_threshold),
        "alignment_block_hours": float(args.alignment_block_hours),
        "physical_speed_bin_ms": float(args.physical_speed_bin_ms),
        "physical_zeta_bin": float(args.physical_zeta_bin),
        "min_distinct_scales_per_case": int(args.min_distinct_scales_per_case),
        "min_controls_per_lead_bucket": int(args.min_controls_per_lead_bucket),
        "min_far_nonstorm_per_lead_bucket": int(args.min_far_nonstorm_per_lead_bucket),
        "adaptive_control_coverage": bool(args.adaptive_control_coverage),
        "n_event_cases_requested": int(len(events)),
        "n_control_cases_requested": int(len(controls)),
        "n_event_cases_used": int(sum(1 for c in case_manifest if str(c.get("case_type")) == "storm")),
        "n_control_cases_used": int(sum(1 for c in case_manifest if str(c.get("case_type")) == "control")),
        "n_cases_excluded": int(sum(1 for c in case_manifest_all if bool(c.get("excluded", False)))),
        "n_cases_used": int(pd.Series(samples["case_id"]).nunique()),
        "n_rows": int(samples.shape[0]),
        "n_pairs": int(samples.shape[0] // 2),
        "lead_buckets": [str(v) for v in sorted(samples["lead_bucket"].astype(str).unique())],
        "scales_cells": scales,
        "polar_enable": bool(args.polar_enable),
        "polar_feature_rows": int(polar_features.shape[0]),
        "control_match_quality_counts": _count_values(case_manifest, "match_quality"),
        "control_tier_counts": _count_values(case_manifest, "control_tier"),
        "center_source_counts": _count_values(case_manifest, "center_source"),
        "control_stats": control_stats,
        "control_coverage": control_coverage,
        "control_coverage_relaxation": coverage_relaxation,
        "lead_case_counts": lead_case_counts,
        "distance_cohort_counts": samples.get("storm_distance_cohort", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "n_vortex_candidates": int(vortex_candidates_df.shape[0]),
        "n_storm_track_points": int(storm_tracks_df.shape[0]),
        "n_tracks": int(storm_tracks_df["track_id"].nunique()) if not storm_tracks_df.empty and "track_id" in storm_tracks_df.columns else 0,
        "anomaly_commutation": _summarize_anomaly_audit(anomaly_audit_rows),
        "scale_audit_path": str((out_dir / "scale_audit.json").resolve()),
        "eta_long_path": str((out_dir / "eta_long.parquet").resolve()),
        "polar_features_path": str((out_dir / "polar_features.parquet").resolve()),
        "vortex_candidates_path": str((out_dir / "vortex_candidates.parquet").resolve()) if not vortex_candidates_df.empty else None,
        "storm_tracks_path": str((out_dir / "storm_tracks.parquet").resolve()) if not storm_tracks_df.empty else None,
        "storm_tracks_derived_path": str((Path("data/derived") / "vortex_tracks_FMA_2025.parquet").resolve()) if not storm_tracks_df.empty else None,
        "storm_track_summary_path": str((out_dir / "storm_track_summary.parquet").resolve()) if not track_summary_df.empty else None,
        "ibtracs_match": ibtracs_match,
        "ibtracs_catalog": ibtracs_catalog,
        "ibtracs_index": ib_index_meta,
        "ibtracs_hourly_tracks_path": str((out_dir / "ibtracs_hourly_tracks.parquet").resolve()) if (out_dir / "ibtracs_hourly_tracks.parquet").exists() else None,
        "ibtracs_alignment_diagnostics_path": str((out_dir / "ibtracs_alignment_diagnostics.parquet").resolve()),
        "ibtracs_alignment_summary_path": str((out_dir / "ibtracs_alignment_summary.json").resolve()),
        "ibtracs_alignment_summary": ib_diag_summary,
        "ibtracs_contract_path": str((out_dir / "ibtracs_contract.parquet").resolve()),
        "ibtracs_contract_samples_path": str((out_dir / "ibtracs_contract_samples.parquet").resolve()),
        "case_windows_path": str((out_dir / "case_windows.parquet").resolve()),
        "case_contract_path": str((out_dir / "case_contract.parquet").resolve()),
        "strict_coverage_by_week_path": str((out_dir / "strict_coverage_by_week.csv").resolve()),
        "case_windows_summary": {
            "n_cases": int(case_windows["case_id"].nunique()) if not case_windows.empty and "case_id" in case_windows.columns else 0,
            "eventness_mean": float(pd.to_numeric(case_windows.get("eventness"), errors="coerce").mean()) if not case_windows.empty else 0.0,
            "farness_mean": float(pd.to_numeric(case_windows.get("farness"), errors="coerce").mean()) if not case_windows.empty else 0.0,
            "align_event_mean": float(pd.to_numeric(case_windows.get("A_align_event"), errors="coerce").mean()) if not case_windows.empty else 0.0,
            "align_far_mean": float(pd.to_numeric(case_windows.get("A_align_far"), errors="coerce").mean()) if not case_windows.empty else 0.0,
        },
        "time_basis_audit_path": str((out_dir / "time_basis_audit.json").resolve()),
        "time_basis_audit": time_basis_audit,
        "center_definition_audit_path": str((out_dir / "center_definition_audit.json").resolve()),
        "center_definition_audit": center_definition_audit,
        "cohort_audit_path": str((out_dir / "cohort_audit.json").resolve()),
        "cohort_audit": cohort_audit,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote canonical tile dataset to {out_dir}")
    print(f"Rows: {summary['n_rows']}, cases: {summary['n_cases_used']}")
    return 0


def _collect_storm_centers(
    dataset: ds.Dataset,
    lead_buckets: list[str],
    max_events_per_lead: int,
    speed_bin_ms: float,
    zeta_bin: float,
    rng: np.random.Generator,
    scan_batch_rows: int = 200000,
    max_batches_per_lead: int = 0,
) -> list[dict[str, Any]]:
    optional_id_cols = [c for c in ("storm_id", "ms_id", "tc_id", "sid") if c in set(dataset.schema.names)]
    extra_phys_cols = [c for c in ("u10", "v10", "zeta") if c in set(dataset.schema.names)]
    cols = [
        "time",
        "lat",
        "lon",
        "lead_partition",
        "storm_point",
        "storm",
        "near_storm",
        "gka_score",
        *extra_phys_cols,
    ] + optional_id_cols
    out: list[dict[str, Any]] = []
    for lead in lead_buckets:
        label_used = None
        centers = pd.DataFrame()
        for label_col in ("storm_point", "storm", "near_storm"):
            filt = _lead_filter(lead) & (ds.field(label_col) == 1)
            scanner = dataset.scanner(
                columns=cols,
                filter=filt,
                batch_size=max(int(scan_batch_rows), 1),
            )
            best_by_time: dict[pd.Timestamp, dict[str, Any]] = {}
            batch_idx = 0
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                if frame.empty:
                    continue
                batch_idx += 1
                if int(max_batches_per_lead) > 0 and batch_idx > int(max_batches_per_lead):
                    break
                frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
                frame = frame.dropna(subset=["time", "lat", "lon"])
                if frame.empty:
                    continue
                frame["gka_score"] = pd.to_numeric(frame["gka_score"], errors="coerce").fillna(-np.inf)
                frame = frame.sort_values(["time", "gka_score"], ascending=[True, False])
                frame = frame.drop_duplicates(subset=["time"], keep="first")
                for _, row in frame.iterrows():
                    t_key = pd.Timestamp(row["time"])
                    current = best_by_time.get(t_key)
                    score = float(row.get("gka_score", -np.inf))
                    if current is None or float(current.get("gka_score", -np.inf)) < score:
                        best_by_time[t_key] = row.to_dict()
            if best_by_time:
                centers = pd.DataFrame(list(best_by_time.values()))
                label_used = label_col
                break
        if centers.empty:
            continue
        if centers.shape[0] > int(max_events_per_lead):
            pick = np.sort(rng.choice(centers.index.to_numpy(), size=int(max_events_per_lead), replace=False))
            centers = centers.loc[pick].reset_index(drop=True)

        for i, row in centers.iterrows():
            storm_id = None
            for c in optional_id_cols:
                if c in row and pd.notna(row[c]):
                    storm_id = str(row[c])
                    break
            if storm_id is None:
                storm_id = (
                    f"storm_{lead}_{pd.Timestamp(row['time']).strftime('%Y%m%d%H')}_"
                    f"{float(row['lat']):.2f}_{float(row['lon']):.2f}"
                )
            out.append(
                {
                    "case_id": f"storm_lead{lead}_{i:04d}",
                    "case_type": "storm",
                    "event_label": str(label_used),
                    "center_source": "labels",
                    "storm_id": storm_id,
                    "lead_bucket": str(lead),
                    "time0": pd.Timestamp(row["time"]),
                    "lat0": float(row["lat"]),
                    "lon0": float(row["lon"]),
                    "speed_bin": (
                        float(
                            np.floor(
                                np.sqrt(float(row.get("u10", np.nan)) ** 2 + float(row.get("v10", np.nan)) ** 2)
                                / max(float(speed_bin_ms), 1e-6)
                            )
                            * max(float(speed_bin_ms), 1e-6)
                        )
                        if np.isfinite(float(row.get("u10", np.nan))) and np.isfinite(float(row.get("v10", np.nan)))
                        else None
                    ),
                    "zeta_bin": (
                        float(np.floor(float(row.get("zeta", np.nan)) / max(float(zeta_bin), 1e-12)) * max(float(zeta_bin), 1e-12))
                        if np.isfinite(float(row.get("zeta", np.nan)))
                        else None
                    ),
                }
            )
    return out


def _collect_vortex_storm_centers(
    *,
    dataset: ds.Dataset,
    lead_buckets: list[str],
    max_events_per_lead: int,
    detect_cfg: VortexDetectConfig,
    track_cfg: VortexTrackConfig,
    min_track_duration_hours: float,
    min_track_points: int,
    max_fragments_per_lead: int,
    scan_batch_rows: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = set(dataset.schema.names)
    base_cols = ["time", "lat", "lon", "u10", "v10", "lead_partition"]
    mslp_col = "mslp" if "mslp" in available else None
    if mslp_col:
        base_cols.append(mslp_col)
    missing = [c for c in ("time", "lat", "lon", "u10", "v10", "lead_partition") if c not in available]
    if missing:
        raise ValueError(
            f"Prepared dataset is missing required fields for vortex discovery: {missing}. "
            "Run prepare_weather_v1.py to populate u10/v10 gridded partitions."
        )

    events: list[dict[str, Any]] = []
    all_candidates: list[pd.DataFrame] = []
    all_tracks: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for lead in lead_buckets:
        filt = _lead_filter(lead)
        lead_candidates: list[pd.DataFrame] = []
        scanner = dataset.scanner(
            columns=base_cols,
            filter=filt,
            batch_size=max(int(scan_batch_rows), 1),
        )
        batch_idx = 0
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            batch_idx += 1
            if int(max_fragments_per_lead) > 0 and batch_idx > int(max_fragments_per_lead):
                break
            cand = detect_vortex_candidates(
                frame,
                lead_bucket=str(lead),
                mslp_col=mslp_col,
                config=detect_cfg,
            )
            if not cand.empty:
                lead_candidates.append(cand)
        if not lead_candidates:
            continue
        cand_df = pd.concat(lead_candidates, ignore_index=True)
        cand_df["lead_bucket"] = str(lead)
        tracks = track_vortex_candidates(cand_df, config=track_cfg)
        if tracks.empty:
            cand_ranked = cand_df.sort_values(["score", "time"], ascending=[False, True]).head(int(max_events_per_lead))
            for idx, row in cand_ranked.reset_index(drop=True).iterrows():
                events.append(
                    {
                        "case_id": f"storm_lead{lead}_{idx:04d}",
                        "case_type": "storm",
                        "event_label": "vortex_candidate",
                        "center_source": "vortex",
                        "storm_id": f"candidate_{lead}_{idx:04d}",
                        "lead_bucket": str(lead),
                        "time0": pd.Timestamp(row["time"]),
                        "lat0": float(row["lat0"]),
                        "lon0": float(row["lon0"]),
                        "speed_bin": None,
                        "zeta_bin": (
                            float(np.floor(float(row.get("zeta_peak", np.nan)) / 1e-5) * 1e-5)
                            if np.isfinite(float(row.get("zeta_peak", np.nan)))
                            else None
                        ),
                    }
                )
            all_candidates.append(cand_df)
            continue

        tracks["lead_bucket"] = str(lead)
        track_summary = summarize_tracks(
            tracks,
            min_duration_hours=float(min_track_duration_hours),
            min_points=int(min_track_points),
        )
        centers = select_track_centers(
            tracks,
            track_summary,
            max_events=int(max_events_per_lead),
            lead_bucket=str(lead),
        )
        events.extend(centers)
        all_candidates.append(cand_df)
        all_tracks.append(tracks)
        all_summaries.append(track_summary)

    cand_out = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    track_out = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    summary_out = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    return events, cand_out, track_out, summary_out


def _collect_ibtracs_centers(
    *,
    ib_points: pd.DataFrame,
    lead_buckets: list[str],
    max_events_per_lead: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    ib = ib_points.copy()
    if ib.empty:
        return []
    ib["time"] = pd.to_datetime(ib["time"], errors="coerce")
    ib = ib.dropna(subset=["time", "lat0", "lon0", "storm_id"]).copy()
    if ib.empty:
        return []
    out: list[dict[str, Any]] = []
    for lead in lead_buckets:
        take = ib.copy()
        take = take.sort_values(["storm_id", "time"]).drop_duplicates(subset=["storm_id", "time"], keep="first").reset_index(drop=True)
        if take.shape[0] > int(max_events_per_lead):
            sel = np.sort(rng.choice(take.index.to_numpy(), size=int(max_events_per_lead), replace=False))
            take = take.loc[sel].reset_index(drop=True)
        for i, row in take.iterrows():
            out.append(
                {
                    "case_id": f"storm_lead{lead}_{i:04d}",
                    "case_type": "storm",
                    "event_label": "ibtracs",
                    "storm_id": str(row["storm_id"]),
                    "lead_bucket": str(lead),
                    "time0": pd.Timestamp(row["time"]),
                    "lat0": float(row["lat0"]),
                    "lon0": float(row["lon0"]),
                    "speed_bin": None,
                    "zeta_bin": None,
                    "track_id": None,
                    "center_source": "ibtracs",
                }
            )
    return out


def _make_offset_controls(
    events: list[dict[str, Any]],
    lon_offset: float,
    lon_min: float,
    lon_max: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        lon = float(event["lon0"]) + float(lon_offset)
        if lon > lon_max:
            lon = float(event["lon0"]) - float(lon_offset)
        lon = float(np.clip(lon, lon_min, lon_max))
        out.append(
            {
                "case_id": event["case_id"].replace("storm_", "ctrl_"),
                "case_type": "control",
                "event_label": str(event.get("event_label", "storm_point")),
                "center_source": "background",
                "storm_id": str(event.get("storm_id", event["case_id"])),
                "lead_bucket": event["lead_bucket"],
                "time0": event["time0"],
                "lat0": event["lat0"],
                "lon0": lon,
                "dist_from_event_deg": float(abs(lon - float(event["lon0"]))),
                "dist_from_event_bin_deg": float(np.floor(abs(lon - float(event["lon0"])) / 2.0) * 2.0),
                "match_quality": "tier_c_offset_lon",
                "control_tier": "C",
            }
        )
    return out


def _make_matched_background_controls(
    dataset: ds.Dataset,
    events: list[dict[str, Any]],
    rng: np.random.Generator,
    lat_bin_deg: float,
    lon_bin_deg: float,
    allow_nonexact: bool,
    require_physical_match: bool,
    speed_bin_ms: float,
    zeta_bin: float,
    lon_offset: float,
    lon_min: float,
    lon_max: float,
    scan_batch_rows: int,
    pool_max_rows: int,
    max_batches_per_lead: int,
    controls_per_event: int,
    control_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not events:
        return [], {"requested": 0, "matched": 0, "dropped": 0, "mode": "matched_background"}

    by_lead: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_lead.setdefault(str(event["lead_bucket"]), []).append(event)

    controls: list[dict[str, Any]] = []
    requested = 0
    dropped = 0
    dropped_by_lead: dict[str, int] = {}
    pool_none_cache: pd.DataFrame | None = None
    for lead, lead_events in by_lead.items():
        pool = _background_pool_for_lead(
            dataset=dataset,
            lead=lead,
            lat_bin_deg=lat_bin_deg,
            lon_bin_deg=lon_bin_deg,
            speed_bin_ms=float(speed_bin_ms),
            zeta_bin=float(zeta_bin),
            rng=rng,
            scan_batch_rows=int(scan_batch_rows),
            max_rows=int(pool_max_rows),
            max_batches_per_lead=int(max_batches_per_lead),
            control_source=str(control_source),
        )
        using_cross_lead_none = False
        if pool.empty:
            if allow_nonexact:
                if pool_none_cache is None:
                    pool_none_cache = _background_pool_for_lead(
                        dataset=dataset,
                        lead="none",
                        lat_bin_deg=lat_bin_deg,
                        lon_bin_deg=lon_bin_deg,
                        speed_bin_ms=float(speed_bin_ms),
                        zeta_bin=float(zeta_bin),
                        rng=rng,
                        scan_batch_rows=int(scan_batch_rows),
                        max_rows=int(pool_max_rows),
                        max_batches_per_lead=int(max_batches_per_lead),
                        control_source=str(control_source),
                    )
                if pool_none_cache is not None and not pool_none_cache.empty:
                    pool = pool_none_cache.copy()
                    using_cross_lead_none = True
                else:
                    controls.extend(
                        _make_offset_controls(
                            events=lead_events,
                            lon_offset=lon_offset,
                            lon_min=lon_min,
                            lon_max=lon_max,
                        )
                    )
                    continue
            else:
                dropped += int(len(lead_events))
                dropped_by_lead[str(lead)] = dropped_by_lead.get(str(lead), 0) + int(len(lead_events))
                continue
        used_idx: set[int] = set()
        n_per_event = max(int(controls_per_event), 1)
        for event in lead_events:
            event_base_id = str(event["case_id"]).replace("storm_", "bgm_")
            for k in range(n_per_event):
                requested += 1
                pick, quality = _pick_matched_background_row(
                    pool=pool,
                    event=event,
                    used_idx=used_idx,
                    rng=rng,
                    lat_bin_deg=lat_bin_deg,
                    lon_bin_deg=lon_bin_deg,
                    allow_nonexact=allow_nonexact,
                    require_physical_match=bool(require_physical_match),
                )
                if pick is None:
                    dropped += 1
                    dropped_by_lead[str(lead)] = dropped_by_lead.get(str(lead), 0) + 1
                    continue
                dist_deg = float(
                    np.sqrt(
                        np.square(float(pick["lat"]) - float(event["lat0"]))
                        + np.square(float(pick["lon"]) - float(event["lon0"]))
                    )
                )
                dist_bin = float(np.floor(dist_deg / 2.0) * 2.0)
                controls.append(
                    {
                        "case_id": f"{event_base_id}_c{k+1:02d}",
                        "case_type": "control",
                        "event_label": "background_matched",
                        "center_source": "background",
                        "storm_id": f"background_matched_{lead}_{len(controls):04d}",
                        "lead_bucket": str(lead),
                        "time0": pd.Timestamp(pick["time"]),
                        "lat0": float(pick["lat"]),
                        "lon0": float(pick["lon"]),
                        "dist_from_event_deg": dist_deg,
                        "dist_from_event_bin_deg": dist_bin,
                        "speed_bin": _to_float(pick.get("speed_bin")),
                        "zeta_bin": _to_float(pick.get("zeta_bin")),
                        "match_quality": (
                            f"tier_b_cross_lead_none_{str(quality).replace('tier_a_', '').replace('tier_b_', '')}"
                            if using_cross_lead_none and str(quality).startswith(("tier_a_", "tier_b_"))
                            else quality
                        ),
                        "control_tier": (
                            "B"
                            if using_cross_lead_none and str(quality).startswith(("tier_a_", "tier_b_"))
                            else _control_tier(quality)
                        ),
                    }
                )
    stats = {
        "requested": int(requested),
        "matched": int(len(controls)),
        "dropped": int(dropped),
        "dropped_by_lead": dict(sorted(dropped_by_lead.items())),
        "mode": "matched_background_exact_only" if not allow_nonexact else "matched_background_with_fallback",
        "quality_counts": _count_values(controls, "match_quality"),
        "tier_counts": _count_values(controls, "control_tier"),
    }
    return controls, stats


def _background_pool_for_lead(
    dataset: ds.Dataset,
    lead: str,
    lat_bin_deg: float,
    lon_bin_deg: float,
    speed_bin_ms: float,
    zeta_bin: float,
    rng: np.random.Generator,
    scan_batch_rows: int,
    max_rows: int,
    max_batches_per_lead: int,
    control_source: str,
) -> pd.DataFrame:
    cols = [
        "time",
        "lat",
        "lon",
        "lead_partition",
        "storm",
        "near_storm",
        "pregen",
        "u10",
        "v10",
        "zeta",
    ]
    lead_filt = _lead_filter(lead)
    if str(control_source).lower() == "all_timeslices":
        filt = lead_filt
    else:
        filt = (
            lead_filt
            & (ds.field("storm") == 0)
            & (ds.field("near_storm") == 0)
            & (ds.field("pregen") == 0)
        )
    out_cols = ["time", "lat", "lon", "month", "hour", "lat_bin", "lon_bin", "speed_bin", "zeta_bin"]
    scanner = dataset.scanner(
        columns=cols,
        filter=filt,
        batch_size=max(int(scan_batch_rows), 1),
    )
    pool: pd.DataFrame | None = None
    batch_idx = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        batch_idx += 1
        if int(max_batches_per_lead) > 0 and batch_idx > int(max_batches_per_lead):
            break
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame.dropna(subset=["time", "lat", "lon"])
        if frame.empty:
            continue
        frame["month"] = frame["time"].dt.month.astype(int)
        frame["hour"] = frame["time"].dt.hour.astype(int)
        frame["lat_bin"] = _lat_bin(pd.to_numeric(frame["lat"], errors="coerce"), lat_bin_deg)
        frame["lon_bin"] = _lon_bin(pd.to_numeric(frame["lon"], errors="coerce"), lon_bin_deg)
        u = pd.to_numeric(frame.get("u10"), errors="coerce")
        v = pd.to_numeric(frame.get("v10"), errors="coerce")
        speed = np.sqrt(np.square(u) + np.square(v))
        frame["speed_bin"] = np.floor(speed / max(float(speed_bin_ms), 1e-6)) * max(float(speed_bin_ms), 1e-6)
        zeta = pd.to_numeric(frame.get("zeta"), errors="coerce")
        frame["zeta_bin"] = np.floor(zeta / max(float(zeta_bin), 1e-12)) * max(float(zeta_bin), 1e-12)
        frame = frame.dropna(subset=["lat_bin", "lon_bin"])
        if frame.empty:
            continue
        chunk = frame[out_cols].reset_index(drop=True)
        if int(max_rows) <= 0:
            if pool is None:
                pool = chunk
            else:
                pool = pd.concat([pool, chunk], ignore_index=True)
            continue
        if pool is None:
            pool = chunk
        else:
            pool = pd.concat([pool, chunk], ignore_index=True)
        if pool.shape[0] > int(max_rows):
            keep_idx = np.sort(rng.choice(pool.index.to_numpy(), size=int(max_rows), replace=False))
            pool = pool.loc[keep_idx].reset_index(drop=True)
    if pool is None or pool.empty:
        return pd.DataFrame(columns=out_cols)
    return pool[out_cols].reset_index(drop=True)


def _pick_matched_background_row(
    pool: pd.DataFrame,
    event: dict[str, Any],
    used_idx: set[int],
    rng: np.random.Generator,
    lat_bin_deg: float,
    lon_bin_deg: float,
    allow_nonexact: bool,
    require_physical_match: bool,
) -> tuple[pd.Series | None, str]:
    t0 = pd.Timestamp(event["time0"])
    month = int(t0.month)
    hour = int(t0.hour)
    lat_bin = float(_lat_bin(pd.Series([float(event["lat0"])]), lat_bin_deg).iloc[0])
    lon_bin = float(_lon_bin(pd.Series([float(event["lon0"])]), lon_bin_deg).iloc[0])
    speed_bin = event.get("speed_bin")
    zeta_bin = event.get("zeta_bin")
    remaining_mask = ~pool.index.isin(used_idx)
    if not remaining_mask.any():
        used_idx.clear()
        remaining_mask = np.ones(pool.shape[0], dtype=bool)
    base = pool.loc[remaining_mask]

    physical_mask = pd.Series(True, index=base.index, dtype=bool)
    physical_available = ("speed_bin" in base.columns) and ("zeta_bin" in base.columns)
    if physical_available and speed_bin is not None:
        physical_mask = physical_mask & (pd.to_numeric(base["speed_bin"], errors="coerce") == float(speed_bin))
    if physical_available and zeta_bin is not None:
        physical_mask = physical_mask & (pd.to_numeric(base["zeta_bin"], errors="coerce") == float(zeta_bin))
    base_phys = base.loc[physical_mask] if physical_available else base
    use_phys = bool(require_physical_match and physical_available and not base_phys.empty)
    exact_base = base_phys if use_phys else base

    candidates: list[tuple[pd.DataFrame, str]] = [
        (
            exact_base.loc[
                (exact_base["month"] == month)
                & (exact_base["hour"] == hour)
                & (exact_base["lat_bin"] == lat_bin)
                & (exact_base["lon_bin"] == lon_bin)
            ],
            "tier_a_exact_phys" if use_phys else "tier_a_exact",
        ),
    ]
    if allow_nonexact:
        lon_adj = exact_base.loc[
            (exact_base["month"] == month)
            & (exact_base["hour"] == hour)
            & (exact_base["lat_bin"] == lat_bin)
            & (np.abs(pd.to_numeric(exact_base["lon_bin"], errors="coerce") - float(lon_bin)) <= float(lon_bin_deg))
        ]
        tier_b_prefix = "tier_b_phys_" if use_phys else "tier_b_"
        tier_c_prefix = "tier_c_phys_" if use_phys else "tier_c_"
        candidates.extend(
            [
                (lon_adj, f"{tier_b_prefix}lon_adjacent"),
                (
                    exact_base.loc[
                        (exact_base["month"] == month) & (exact_base["hour"] == hour) & (exact_base["lat_bin"] == lat_bin)
                    ],
                    f"{tier_b_prefix}month_hour_lat",
                ),
                (base.loc[(base["month"] == month) & (base["lat_bin"] == lat_bin)], f"{tier_c_prefix}month_lat"),
                (base.loc[(base["hour"] == hour) & (base["lat_bin"] == lat_bin)], f"{tier_c_prefix}hour_lat"),
                (base.loc[(base["month"] == month) & (base["hour"] == hour)], f"{tier_c_prefix}month_hour"),
                (base.loc[base["month"] == month], f"{tier_c_prefix}month_only"),
                (base.loc[base["lat_bin"] == lat_bin], f"{tier_c_prefix}lat_only"),
                (base, f"{tier_c_prefix}lead_only"),
            ]
        )
    for cand, quality in candidates:
        if cand.empty:
            continue
        pick_idx = int(rng.choice(cand.index.to_numpy()))
        used_idx.add(pick_idx)
        return pool.loc[pick_idx], quality
    return None, "none"


def _collect_background_centers(
    dataset: ds.Dataset,
    lead_buckets: list[str],
    max_per_lead: int,
    rng: np.random.Generator,
    scan_batch_rows: int,
    max_batches_per_lead: int,
    control_source: str,
) -> list[dict[str, Any]]:
    cols = ["time", "lat", "lon", "lead_partition", "u10", "v10", "zeta"]
    out: list[dict[str, Any]] = []
    for lead in lead_buckets:
        lead_filt = _lead_filter(lead)
        if str(control_source).lower() == "all_timeslices":
            filt = lead_filt
        else:
            filt = (
                lead_filt
                & (ds.field("storm") == 0)
                & (ds.field("near_storm") == 0)
                & (ds.field("pregen") == 0)
            )
        scanner = dataset.scanner(
            columns=cols,
            filter=filt,
            batch_size=max(int(scan_batch_rows), 1),
        )
        picks_by_time: dict[pd.Timestamp, dict[str, Any]] = {}
        batch_idx = 0
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            batch_idx += 1
            if int(max_batches_per_lead) > 0 and batch_idx > int(max_batches_per_lead):
                break
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
            frame = frame.dropna(subset=["time", "lat", "lon"])
            if frame.empty:
                continue
            for t_val, grp in frame.groupby("time", sort=False):
                if grp.empty:
                    continue
                row = grp.iloc[int(rng.integers(0, grp.shape[0]))]
                u = _to_float(row.get("u10"))
                v = _to_float(row.get("v10"))
                zeta = _to_float(row.get("zeta"))
                speed = None
                if (u is not None) and (v is not None):
                    speed = float(np.sqrt(float(u) ** 2 + float(v) ** 2))
                # Reservoir-style replacement across batches by random keep.
                if t_val not in picks_by_time or bool(rng.random() < 0.5):
                    picks_by_time[pd.Timestamp(t_val)] = {
                        "time": pd.Timestamp(row["time"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "speed": speed,
                        "zeta": zeta,
                    }
        if not picks_by_time:
            continue
        centers = pd.DataFrame(list(picks_by_time.values())).sort_values("time").reset_index(drop=True)
        if centers.shape[0] > int(max_per_lead):
            sel = np.sort(rng.choice(centers.index.to_numpy(), size=int(max_per_lead), replace=False))
            centers = centers.loc[sel].reset_index(drop=True)

        for i, row in centers.iterrows():
            out.append(
                {
                    "case_id": f"bg_lead{lead}_{i:04d}",
                    "case_type": "control",
                    "event_label": "background",
                    "center_source": "background",
                    "storm_id": f"background_{lead}_{i:04d}",
                    "lead_bucket": str(lead),
                    "time0": pd.Timestamp(row["time"]),
                    "lat0": float(row["lat"]),
                    "lon0": float(row["lon"]),
                    "speed_bin": _to_float(row.get("speed")),
                    "zeta_bin": _to_float(row.get("zeta")),
                    "dist_from_event_deg": None,
                    "dist_from_event_bin_deg": None,
                    "match_quality": "tier_c_unmatched_background",
                    "control_tier": "C",
                }
            )
    return out


def _extract_case_rows(
    dataset: ds.Dataset,
    case: dict[str, Any],
    pre_hours: int,
    post_hours: int,
    tile_half_cells: int,
    grid_step_deg: float,
    scales: list[int],
    cell_km: float,
    anomaly_mode: str,
    anomaly_lat_bin_deg: float,
    anomaly_lon_bin_deg: float,
    anomaly_commute_rmse_max: float,
    anomaly_commute_corr_min: float,
    anomaly_min_bin_samples: int,
    anomaly_min_covered_frac: float,
    min_distinct_scales: int,
    polar_enable: bool,
    polar_r_bins: int,
    polar_theta_bins: int,
    polar_pitches: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    t0 = pd.Timestamp(case["time0"])
    lead = str(case["lead_bucket"])
    lat0 = float(case["lat0"])
    lon0 = float(case["lon0"])

    t_start = (t0 - pd.Timedelta(hours=int(pre_hours))).to_pydatetime()
    t_end = (t0 + pd.Timedelta(hours=int(post_hours))).to_pydatetime()
    half_deg = float(tile_half_cells) * float(grid_step_deg)

    filt = (
        _lead_filter(lead)
        & (ds.field("time") >= t_start)
        & (ds.field("time") <= t_end)
        & (ds.field("lat") >= (lat0 - half_deg))
        & (ds.field("lat") <= (lat0 + half_deg))
        & (ds.field("lon") >= (lon0 - half_deg))
        & (ds.field("lon") <= (lon0 + half_deg))
    )
    cols = [
        "time",
        "lat",
        "lon",
        "u10",
        "v10",
        "u10_mirror",
        "v10_mirror",
        "speed_l",
        "speed_r",
        "eta_parity",
        "storm_point",
        "storm",
        "near_storm",
        "pregen",
        "lead_partition",
    ]
    scalar_pref = ["S", "zeta", "div", "shear_proxy", "gka_score", "gka_parity_eta", "gka_chirality"]
    for col in scalar_pref:
        mcol = f"{col}_mirror"
        if col in set(dataset.schema.names) and mcol in set(dataset.schema.names):
            cols.extend([col, mcol])
    cols = sorted(set(cols))
    available = set(dataset.schema.names)
    required = [
        "time",
        "lat",
        "lon",
        "u10",
        "v10",
        "u10_mirror",
        "v10_mirror",
        "speed_l",
        "speed_r",
        "storm_point",
        "storm",
        "near_storm",
        "pregen",
        "lead_partition",
    ]
    missing_cols = [c for c in required if c not in available]
    if missing_cols:
        raise ValueError(
            f"Prepared dataset is missing required columns for tile extraction: {missing_cols}. "
            "Re-run prepare_weather_v1.py with current schema."
        )
    table = dataset.to_table(columns=cols, filter=filt)
    if table.num_rows == 0:
        return [], [], _default_case_audit(anomaly_mode)
    frame = table.to_pandas()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time", "lat", "lon"])
    if frame.empty:
        return [], [], _default_case_audit(anomaly_mode)

    frame["u10"] = pd.to_numeric(frame["u10"], errors="coerce")
    frame["v10"] = pd.to_numeric(frame["v10"], errors="coerce")
    frame["u10_mirror"] = pd.to_numeric(frame["u10_mirror"], errors="coerce")
    frame["v10_mirror"] = pd.to_numeric(frame["v10_mirror"], errors="coerce")
    frame["speed_l_raw"] = np.sqrt(np.square(frame["u10"]) + np.square(frame["v10"]))
    frame["speed_r_raw"] = np.sqrt(np.square(frame["u10_mirror"]) + np.square(frame["v10_mirror"]))

    # Local-frame components around each case center reduce global-coordinate parity artifacts.
    etx, ety = _local_tangential_basis(
        lat=pd.to_numeric(frame["lat"], errors="coerce").to_numpy(dtype=float),
        lon=pd.to_numeric(frame["lon"], errors="coerce").to_numpy(dtype=float),
        lat0=float(lat0),
        lon0=float(lon0),
    )
    u = pd.to_numeric(frame["u10"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(frame["v10"], errors="coerce").to_numpy(dtype=float)
    um = pd.to_numeric(frame["u10_mirror"], errors="coerce").to_numpy(dtype=float)
    vm = pd.to_numeric(frame["v10_mirror"], errors="coerce").to_numpy(dtype=float)
    tan_l = u * etx + v * ety
    tan_r = um * etx + vm * ety
    frame["local_tan_l_raw"] = np.abs(tan_l)
    frame["local_tan_r_raw"] = np.abs(tan_r)

    if "zeta" in frame.columns and "zeta_mirror" in frame.columns:
        frame["vort_l_raw"] = np.abs(pd.to_numeric(frame["zeta"], errors="coerce"))
        frame["vort_r_raw"] = np.abs(pd.to_numeric(frame["zeta_mirror"], errors="coerce"))
    else:
        frame["vort_l_raw"] = frame["speed_l_raw"]
        frame["vort_r_raw"] = frame["speed_r_raw"]

    scalar_pairs = [(c, f"{c}_mirror") for c in scalar_pref if c in frame.columns and f"{c}_mirror" in frame.columns]
    if scalar_pairs:
        left_mat = np.column_stack([np.abs(pd.to_numeric(frame[l], errors="coerce").to_numpy(dtype=float)) for l, _ in scalar_pairs])
        right_mat = np.column_stack([np.abs(pd.to_numeric(frame[r], errors="coerce").to_numpy(dtype=float)) for _, r in scalar_pairs])
        frame["scalar_l_raw"] = np.nanmean(left_mat, axis=1)
        frame["scalar_r_raw"] = np.nanmean(right_mat, axis=1)
    else:
        frame["scalar_l_raw"] = np.nan
        frame["scalar_r_raw"] = np.nan

    frame_hour, audit_hour = _apply_anomaly_removal(
        frame.copy(),
        mode="lat_hour",
        lat_bin_deg=float(anomaly_lat_bin_deg),
        lon_bin_deg=float(anomaly_lon_bin_deg),
        min_bin_samples=int(anomaly_min_bin_samples),
    )
    frame_day, audit_day = _apply_anomaly_removal(
        frame.copy(),
        mode="lat_day",
        lat_bin_deg=float(anomaly_lat_bin_deg),
        lon_bin_deg=float(anomaly_lon_bin_deg),
        min_bin_samples=int(anomaly_min_bin_samples),
    )

    frame["speed_l_lat_hour"] = pd.to_numeric(frame_hour["speed_l"], errors="coerce")
    frame["speed_r_lat_hour"] = pd.to_numeric(frame_hour["speed_r"], errors="coerce")
    frame["speed_l_lat_day"] = pd.to_numeric(frame_day["speed_l"], errors="coerce")
    frame["speed_r_lat_day"] = pd.to_numeric(frame_day["speed_r"], errors="coerce")

    requested_mode = str(anomaly_mode)
    effective_mode = "none"
    commute_pass = True
    selected_audit = _default_anomaly_metrics()
    selection_reason = "requested_none"

    hour_sufficient = bool(float(audit_hour.get("bin_coverage_frac") or 0.0) >= float(anomaly_min_covered_frac))
    day_sufficient = bool(float(audit_day.get("bin_coverage_frac") or 0.0) >= float(anomaly_min_covered_frac))

    if requested_mode == "lat_hour":
        if hour_sufficient:
            effective_mode = "lat_hour"
            selected_audit = audit_hour
            selection_reason = "lat_hour_sufficient"
        elif day_sufficient:
            effective_mode = "lat_day"
            selected_audit = audit_day
            selection_reason = "fallback_lat_day_insufficient_lat_hour"
        else:
            effective_mode = "none"
            selected_audit = _default_anomaly_metrics()
            selection_reason = "fallback_none_insufficient_bins"
    elif requested_mode == "lat_day":
        if day_sufficient:
            effective_mode = "lat_day"
            selected_audit = audit_day
            selection_reason = "lat_day_sufficient"
        else:
            effective_mode = "none"
            selected_audit = _default_anomaly_metrics()
            selection_reason = "fallback_none_insufficient_bins"
    else:
        effective_mode = "none"
        selected_audit = _default_anomaly_metrics()
        selection_reason = "requested_none"

    if effective_mode == "lat_hour":
        commute_pass = _anomaly_commute_pass(
            audit_hour,
            rmse_max=float(anomaly_commute_rmse_max),
            corr_min=float(anomaly_commute_corr_min),
        )
    elif effective_mode == "lat_day":
        commute_pass = _anomaly_commute_pass(
            audit_day,
            rmse_max=float(anomaly_commute_rmse_max),
            corr_min=float(anomaly_commute_corr_min),
        )
    else:
        commute_pass = True

    frame["speed_l_effective"] = frame[f"speed_l_{effective_mode}"] if effective_mode != "none" else frame["speed_l_raw"]
    frame["speed_r_effective"] = frame[f"speed_r_{effective_mode}"] if effective_mode != "none" else frame["speed_r_raw"]

    frame = frame.dropna(subset=["speed_l_raw", "speed_r_raw", "speed_l_effective", "speed_r_effective"])
    if frame.empty:
        return [], [], _default_case_audit(
            requested_mode,
            effective_mode=effective_mode,
            commute_pass=commute_pass,
            metrics=selected_audit,
        )

    rows: list[dict[str, Any]] = []
    polar_rows: list[dict[str, Any]] = []
    scale_point_counts: dict[str, int] = {}
    scales_used_cells: list[int] = []
    for scale in scales:
        half = (int(scale) // 2) * float(grid_step_deg)
        mask = (np.abs(frame["lat"] - lat0) <= half) & (np.abs(frame["lon"] - lon0) <= half)
        subset = frame.loc[mask]
        if subset.empty:
            continue
        agg_spec: dict[str, tuple[str, str]] = {
            "O_L": ("speed_l_effective", "mean"),
            "O_R": ("speed_r_effective", "mean"),
            "O_L_raw": ("speed_l_raw", "mean"),
            "O_R_raw": ("speed_r_raw", "mean"),
            "O_L_local": ("local_tan_l_raw", "mean"),
            "O_R_local": ("local_tan_r_raw", "mean"),
            "O_L_vort": ("vort_l_raw", "mean"),
            "O_R_vort": ("vort_r_raw", "mean"),
            "O_L_lat_hour": ("speed_l_lat_hour", "mean"),
            "O_R_lat_hour": ("speed_r_lat_hour", "mean"),
            "O_L_lat_day": ("speed_l_lat_day", "mean"),
            "O_R_lat_day": ("speed_r_lat_day", "mean"),
            "eta_parity": ("eta_parity", "mean"),
            "storm": ("storm", "max"),
            "near_storm": ("near_storm", "max"),
            "pregen": ("pregen", "max"),
            "storm_point": ("storm_point", "max"),
        }
        if "scalar_l_raw" in subset.columns and "scalar_r_raw" in subset.columns:
            agg_spec["O_L_scalar"] = ("scalar_l_raw", "mean")
            agg_spec["O_R_scalar"] = ("scalar_r_raw", "mean")
        agg = subset.groupby("time", as_index=False).agg(**agg_spec).sort_values("time")
        if agg.empty:
            continue
        subset_by_time = {pd.Timestamp(k): g.copy() for k, g in subset.groupby("time", sort=False)}
        valid_for_scale = 0
        L_km = float(scale) * float(cell_km)
        for _, row in agg.iterrows():
            t_val = pd.Timestamp(row["time"])
            t_rel_h = float((t_val - t0).total_seconds() / 3600.0)
            o_l = float(row["O_L"])
            o_r = float(row["O_R"])
            if not (np.isfinite(o_l) and np.isfinite(o_r)):
                continue
            if o_l <= 0 or o_r <= 0:
                continue
            o_l_raw = float(row["O_L_raw"])
            o_r_raw = float(row["O_R_raw"])
            o_l_local = float(row["O_L_local"])
            o_r_local = float(row["O_R_local"])
            o_l_vort = float(row["O_L_vort"])
            o_r_vort = float(row["O_R_vort"])
            o_l_h = float(row["O_L_lat_hour"])
            o_r_h = float(row["O_R_lat_hour"])
            o_l_d = float(row["O_L_lat_day"])
            o_r_d = float(row["O_R_lat_day"])
            o_l_scalar = float(row.get("O_L_scalar", np.nan))
            o_r_scalar = float(row.get("O_R_scalar", np.nan))
            if not np.isfinite(o_l_scalar) or o_l_scalar <= 0:
                o_l_scalar = o_l_raw
            if not np.isfinite(o_r_scalar) or o_r_scalar <= 0:
                o_r_scalar = o_r_raw

            polar_orig = {
                "left_response": np.nan,
                "right_response": np.nan,
                "chirality": np.nan,
                "spiral_score": np.nan,
                "dominant_m": np.nan,
                "dominant_m_power": np.nan,
                "phase_consistency": np.nan,
            }
            polar_mirror = {
                "left_response": np.nan,
                "right_response": np.nan,
                "chirality": np.nan,
                "spiral_score": np.nan,
                "dominant_m": np.nan,
                "dominant_m_power": np.nan,
                "phase_consistency": np.nan,
            }
            polar_parity = {
                "polar_odd_energy_ratio": np.nan,
                "eta_parity_polar": np.nan,
            }
            if bool(polar_enable):
                slice_df = subset_by_time.get(t_val)
                if slice_df is not None and not slice_df.empty:
                    try:
                        polar_orig = compute_polar_metrics(
                            slice_df,
                            lat0=float(lat0),
                            lon0=float(lon0),
                            u_col="u10",
                            v_col="v10",
                            n_r=int(polar_r_bins),
                            n_theta=int(polar_theta_bins),
                            r_max_km=float(L_km),
                            pitch_values=polar_pitches,
                        )
                        polar_mirror = compute_polar_metrics(
                            slice_df,
                            lat0=float(lat0),
                            lon0=float(lon0),
                            u_col="u10_mirror",
                            v_col="v10_mirror",
                            n_r=int(polar_r_bins),
                            n_theta=int(polar_theta_bins),
                            r_max_km=float(L_km),
                            pitch_values=polar_pitches,
                        )
                        polar_parity = compute_polar_parity_metrics(
                            slice_df,
                            lat0=float(lat0),
                            lon0=float(lon0),
                            n_r=int(polar_r_bins),
                            n_theta=int(polar_theta_bins),
                            r_max_km=float(L_km),
                            pitch_values=polar_pitches,
                        )
                    except Exception:
                        polar_orig = {
                            "left_response": np.nan,
                            "right_response": np.nan,
                            "chirality": np.nan,
                            "spiral_score": np.nan,
                            "dominant_m": np.nan,
                            "dominant_m_power": np.nan,
                            "phase_consistency": np.nan,
                        }
                        polar_mirror = dict(polar_orig)
                        polar_parity = {
                            "polar_odd_energy_ratio": np.nan,
                            "eta_parity_polar": np.nan,
                        }
                polar_rows.append(
                    {
                        "case_id": case["case_id"],
                        "case_type": case["case_type"],
                        "t": t_val,
                        "L": L_km,
                        "lead_bucket": lead,
                        "lat0": float(lat0),
                        "lon0": float(lon0),
                        "polar_left_response_orig": float(polar_orig.get("left_response", np.nan)),
                        "polar_right_response_orig": float(polar_orig.get("right_response", np.nan)),
                        "polar_left_response_mirror": float(polar_mirror.get("left_response", np.nan)),
                        "polar_right_response_mirror": float(polar_mirror.get("right_response", np.nan)),
                        "polar_chirality_orig": float(polar_orig.get("chirality", np.nan)),
                        "polar_chirality_mirror": float(polar_mirror.get("chirality", np.nan)),
                        "polar_spiral_score_orig": float(polar_orig.get("spiral_score", np.nan)),
                        "polar_spiral_score_mirror": float(polar_mirror.get("spiral_score", np.nan)),
                        "polar_phase_consistency_orig": float(polar_orig.get("phase_consistency", np.nan)),
                        "polar_phase_consistency_mirror": float(polar_mirror.get("phase_consistency", np.nan)),
                        "polar_vt_mean_abs_orig": float(polar_orig.get("vt_mean_abs", np.nan)),
                        "polar_vr_mean_abs_orig": float(polar_orig.get("vr_mean_abs", np.nan)),
                        "polar_vt_mean_abs_mirror": float(polar_mirror.get("vt_mean_abs", np.nan)),
                        "polar_vr_mean_abs_mirror": float(polar_mirror.get("vr_mean_abs", np.nan)),
                        "polar_odd_energy_ratio": float(polar_parity.get("polar_odd_energy_ratio", np.nan)),
                        "eta_parity_polar": float(polar_parity.get("eta_parity_polar", np.nan)),
                    }
                )
            is_control_case = str(case.get("case_type", "")).lower() == "control"
            storm_v = 0 if is_control_case else int(row["storm"])
            near_v = 0 if is_control_case else int(row["near_storm"])
            pregen_v = 0 if is_control_case else int(row["pregen"])
            storm_point_v = 0 if is_control_case else int(row["storm_point"])
            common = {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "center_source": case.get("center_source"),
                "match_quality": case.get("match_quality"),
                "control_tier": case.get("control_tier"),
                "lead_bucket": lead,
                "t": t_val,
                "L": L_km,
                "omega": 1.0,
                "t_rel_h": t_rel_h,
                "onset_time": t0,
                "storm_id": str(case.get("storm_id", case["case_id"])),
                "lat0": float(lat0),
                "lon0": float(lon0),
                "dist_from_event_deg": (
                    float(case.get("dist_from_event_deg"))
                    if case.get("dist_from_event_deg") is not None
                    else None
                ),
                "dist_from_event_bin_deg": (
                    float(case.get("dist_from_event_bin_deg"))
                    if case.get("dist_from_event_bin_deg") is not None
                    else None
                ),
                "case_nearest_storm_distance_km": (
                    float(case.get("nearest_storm_distance_km"))
                    if case.get("nearest_storm_distance_km") is not None and np.isfinite(float(case.get("nearest_storm_distance_km")))
                    else np.nan
                ),
                "case_nearest_storm_id": case.get("nearest_storm_id"),
                "case_storm_distance_cohort": case.get("storm_distance_cohort"),
                "case_speed_bin": _to_float(case.get("speed_bin")),
                "case_zeta_bin": _to_float(case.get("zeta_bin")),
                "storm_phase": case.get("storm_phase"),
                "storm_dist_km": (
                    float(case.get("storm_dist_km"))
                    if case.get("storm_dist_km") is not None and np.isfinite(float(case.get("storm_dist_km")))
                    else np.nan
                ),
                "ib_event_strict_case": bool(case.get("ib_event_strict", case.get("ib_event", False))),
                "ib_far_strict_case": bool(case.get("ib_far_strict", case.get("ib_far", False))),
                "ib_far_quality_tag": case.get("ib_far_quality_tag"),
                "eta_parity_mean": float(row["eta_parity"]),
                "storm": int(storm_v),
                "near_storm": int(near_v),
                "pregen": int(pregen_v),
                "storm_point": int(storm_point_v),
                "anomaly_mode_requested": requested_mode,
                "anomaly_mode_effective": effective_mode,
                "anomaly_commute_pass": bool(commute_pass),
            }
            rows.append(
                {
                    **common,
                    "hand": "L",
                    "O": o_l,
                    "O_vector": o_l,
                    "O_scalar": o_l_scalar,
                    "O_raw": o_l_raw,
                    "O_local_frame": o_l_local,
                    "O_vorticity": o_l_vort,
                    "O_meanflow": o_l_h,
                    "O_lat_hour": o_l_h,
                    "O_lat_day": o_l_d,
                    "O_polar_spiral": _finite_positive(polar_orig.get("spiral_score"), eps=1e-8, default=np.nan),
                    "O_polar_chiral": _finite_positive(abs(float(polar_orig.get("chirality", np.nan))), eps=1e-8, default=np.nan),
                    "O_polar_left": _finite_positive(polar_orig.get("left_response"), eps=1e-8, default=np.nan),
                    "O_polar_right": _finite_positive(polar_orig.get("right_response"), eps=1e-8, default=np.nan),
                    "O_polar_odd_ratio": _finite_positive(polar_parity.get("polar_odd_energy_ratio"), eps=1e-8, default=np.nan),
                    "O_polar_eta": _finite_positive(abs(float(polar_parity.get("eta_parity_polar", np.nan))), eps=1e-8, default=np.nan),
                    "O_polar_tangential": _finite_positive(polar_orig.get("vt_mean_abs"), eps=1e-8, default=np.nan),
                    "O_polar_radial": _finite_positive(polar_orig.get("vr_mean_abs"), eps=1e-8, default=np.nan),
                }
            )
            rows.append(
                {
                    **common,
                    "hand": "R",
                    "O": o_r,
                    "O_vector": o_r,
                    "O_scalar": o_r_scalar,
                    "O_raw": o_r_raw,
                    "O_local_frame": o_r_local,
                    "O_vorticity": o_r_vort,
                    "O_meanflow": o_r_h,
                    "O_lat_hour": o_r_h,
                    "O_lat_day": o_r_d,
                    "O_polar_spiral": _finite_positive(polar_mirror.get("spiral_score"), eps=1e-8, default=np.nan),
                    "O_polar_chiral": _finite_positive(abs(float(polar_mirror.get("chirality", np.nan))), eps=1e-8, default=np.nan),
                    "O_polar_left": _finite_positive(polar_mirror.get("left_response"), eps=1e-8, default=np.nan),
                    "O_polar_right": _finite_positive(polar_mirror.get("right_response"), eps=1e-8, default=np.nan),
                    "O_polar_odd_ratio": _finite_positive(polar_parity.get("polar_odd_energy_ratio"), eps=1e-8, default=np.nan),
                    "O_polar_eta": _finite_positive(abs(float(polar_parity.get("eta_parity_polar", np.nan))), eps=1e-8, default=np.nan),
                    "O_polar_tangential": _finite_positive(polar_mirror.get("vt_mean_abs"), eps=1e-8, default=np.nan),
                    "O_polar_radial": _finite_positive(polar_mirror.get("vr_mean_abs"), eps=1e-8, default=np.nan),
                }
            )
            valid_for_scale += 1
        if valid_for_scale > 0:
            scale_point_counts[str(int(scale))] = int(valid_for_scale)
            scales_used_cells.append(int(scale))
    case_audit = _default_case_audit(
        requested_mode,
        effective_mode=effective_mode,
        commute_pass=commute_pass,
        metrics=selected_audit,
        selection_reason=selection_reason,
    )
    unique_scales = sorted(set(scales_used_cells))
    case_audit.update(
        {
            "n_scales_total": int(len(scales)),
            "n_scales_valid": int(len(unique_scales)),
            "scales_used_cells": unique_scales,
            "scale_point_counts": scale_point_counts,
            "scale_gate_pass": bool(len(unique_scales) >= int(min_distinct_scales)),
            "scale_gate_min_distinct": int(min_distinct_scales),
        }
    )
    if len(unique_scales) < int(min_distinct_scales):
        case_audit["excluded_reason"] = "insufficient_scales"
        return [], [], case_audit
    return rows, polar_rows, case_audit


def _apply_anomaly_removal(
    frame: pd.DataFrame,
    *,
    mode: str,
    lat_bin_deg: float,
    lon_bin_deg: float,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    out = frame.copy()
    out["lat_bin"] = _lat_bin(pd.to_numeric(out["lat"], errors="coerce"), lat_bin_deg)
    out["lon_bin"] = _lon_bin(pd.to_numeric(out["lon"], errors="coerce"), lon_bin_deg)
    ts = pd.to_datetime(out["time"], errors="coerce")
    if mode == "lat_hour":
        out["time_key"] = ts.dt.hour.astype("Int64")
        out["month_key"] = ts.dt.month.astype("Int64")
        keys = ["lat_bin", "lon_bin", "time_key", "month_key"]
    elif mode == "lat_day":
        out["time_key"] = ts.dt.dayofyear.astype("Int64")
        out["month_key"] = ts.dt.month.astype("Int64")
        keys = ["lat_bin", "lon_bin", "time_key"]
    else:
        out["time_key"] = 0
        out["month_key"] = 0
        keys = ["lat_bin", "lon_bin", "time_key"]

    base_cols = ["u10", "v10", "u10_mirror", "v10_mirror"]
    present = [c for c in base_cols if c in out.columns]
    if len(present) < 4:
        return out, _default_anomaly_metrics()

    for col in present:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    counts = out.groupby(keys, dropna=False)["time"].transform("count")
    means = out.groupby(keys, dropna=False)[present].transform("mean")
    out["u10_anom"] = out["u10"] - means["u10"]
    out["v10_anom"] = out["v10"] - means["v10"]
    out["u10_mirror_anom"] = out["u10_mirror"] - means["u10_mirror"]
    out["v10_mirror_anom"] = out["v10_mirror"] - means["v10_mirror"]

    u = pd.to_numeric(out["u10_anom"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(out["v10_anom"], errors="coerce").to_numpy(dtype=float)
    um = pd.to_numeric(out["u10_mirror_anom"], errors="coerce").to_numpy(dtype=float)
    vm = pd.to_numeric(out["v10_mirror_anom"], errors="coerce").to_numpy(dtype=float)

    out["speed_l"] = np.sqrt(np.square(u) + np.square(v))
    out["speed_r"] = np.sqrt(np.square(um) + np.square(vm))

    # Recompute parity contrast from anomaly channels to suppress mean-flow leakage.
    u_even = 0.5 * (u + um)
    u_odd = 0.5 * (u - um)
    v_even = 0.5 * (v + vm)
    v_odd = 0.5 * (v - vm)
    odd_mag = np.sqrt(np.square(u_odd) + np.square(v_odd))
    even_mag = np.sqrt(np.square(u_even) + np.square(v_even))
    out["eta_parity"] = odd_mag / (odd_mag + even_mag + 1e-8)
    metrics = _anomaly_commutation_metrics(out)
    valid_bins = np.isfinite(pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float))
    if np.any(valid_bins):
        cvals = pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float)
        cvals = cvals[np.isfinite(cvals)]
        metrics["bin_count_median"] = float(np.median(cvals)) if cvals.size else None
        metrics["bin_coverage_frac"] = float(np.mean(cvals >= max(int(min_bin_samples), 1))) if cvals.size else 0.0
    else:
        metrics["bin_count_median"] = None
        metrics["bin_coverage_frac"] = 0.0
    return out, metrics


def _lead_filter(lead: str) -> ds.Expression:
    """Partition-aware lead filter."""

    return ds.field("lead") == str(lead)


def _lat_bin(values: pd.Series, lat_bin_deg: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    step = max(float(lat_bin_deg), 1e-6)
    return np.round(v / step) * step


def _lon_bin(values: pd.Series, lon_bin_deg: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    step = max(float(lon_bin_deg), 1e-6)
    return np.round(v / step) * step


def _local_tangential_basis(*, lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat.astype(float))
    lon_rad = np.deg2rad(lon.astype(float))
    lat0_rad = float(np.deg2rad(lat0))
    lon0_rad = float(np.deg2rad(lon0))
    dlon = lon_rad - lon0_rad

    east = np.cos(lat_rad) * np.sin(dlon)
    north = np.cos(lat0_rad) * np.sin(lat_rad) - np.sin(lat0_rad) * np.cos(lat_rad) * np.cos(dlon)
    norm = np.sqrt(np.square(east) + np.square(north))
    norm = np.where(norm > 1e-12, norm, 1.0)
    er_e = east / norm
    er_n = north / norm

    et_e = -er_n
    et_n = er_e
    return et_e, et_n


def _default_anomaly_metrics() -> dict[str, float | None]:
    return {
        "u_commute_rmse": None,
        "v_commute_rmse": None,
        "u_commute_corr": None,
        "v_commute_corr": None,
        "bin_count_median": None,
        "bin_coverage_frac": None,
    }


def _anomaly_commutation_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    u = pd.to_numeric(frame.get("u10_anom"), errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(frame.get("v10_anom"), errors="coerce").to_numpy(dtype=float)
    um = pd.to_numeric(frame.get("u10_mirror_anom"), errors="coerce").to_numpy(dtype=float)
    vm = pd.to_numeric(frame.get("v10_mirror_anom"), errors="coerce").to_numpy(dtype=float)

    u_mask = np.isfinite(u) & np.isfinite(um)
    v_mask = np.isfinite(v) & np.isfinite(vm)

    if np.any(u_mask):
        u_resid = u[u_mask] + um[u_mask]
        u_rmse = float(np.sqrt(np.mean(np.square(u_resid))))
        u_corr = _safe_corr(u[u_mask], -um[u_mask])
    else:
        u_rmse = None
        u_corr = None

    if np.any(v_mask):
        v_resid = v[v_mask] - vm[v_mask]
        v_rmse = float(np.sqrt(np.mean(np.square(v_resid))))
        v_corr = _safe_corr(v[v_mask], vm[v_mask])
    else:
        v_rmse = None
        v_corr = None

    return {
        "u_commute_rmse": u_rmse,
        "v_commute_rmse": v_rmse,
        "u_commute_corr": u_corr,
        "v_commute_corr": v_corr,
    }


def _anomaly_commute_pass(metrics: dict[str, float | None], rmse_max: float, corr_min: float) -> bool:
    u_rmse = metrics.get("u_commute_rmse")
    v_rmse = metrics.get("v_commute_rmse")
    u_corr = metrics.get("u_commute_corr")
    v_corr = metrics.get("v_commute_corr")
    if u_rmse is None or v_rmse is None or u_corr is None or v_corr is None:
        return False
    return bool(
        (float(u_rmse) <= float(rmse_max))
        and (float(v_rmse) <= float(rmse_max))
        and (float(u_corr) >= float(corr_min))
        and (float(v_corr) >= float(corr_min))
    )


def _default_case_audit(
    requested_mode: str,
    *,
    effective_mode: str | None = None,
    commute_pass: bool | None = None,
    metrics: dict[str, float | None] | None = None,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    default_pass = str(requested_mode) == "none"
    rec = {
        "anomaly_mode_requested": str(requested_mode),
        "anomaly_mode_effective": str(effective_mode or "none"),
        "anomaly_commute_pass": bool(commute_pass) if commute_pass is not None else bool(default_pass),
        "anomaly_selection_reason": str(selection_reason or "unset"),
        "n_scales_total": 0,
        "n_scales_valid": 0,
        "scales_used_cells": [],
        "scale_point_counts": {},
        "scale_gate_pass": False,
        "scale_gate_min_distinct": 0,
    }
    rec.update(metrics or _default_anomaly_metrics())
    return rec


def _summarize_anomaly_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_cases": 0}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_cases": 0}
    pass_rate = float(pd.to_numeric(frame.get("anomaly_commute_pass"), errors="coerce").fillna(0).mean())
    out = {
        "n_cases": int(frame.shape[0]),
        "pass_rate": pass_rate,
        "effective_mode_counts": frame.get("anomaly_mode_effective", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "selection_reason_counts": frame.get("anomaly_selection_reason", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }
    for col in ("u_commute_rmse", "v_commute_rmse", "u_commute_corr", "v_commute_corr", "bin_count_median", "bin_coverage_frac"):
        vals = pd.to_numeric(frame.get(col), errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{col}_median"] = float(np.median(vals)) if vals.size else None
    return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    sa = float(np.std(a, ddof=0))
    sb = float(np.std(b, ddof=0))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _control_tier(match_quality: str | None) -> str:
    q = str(match_quality or "").lower()
    if q.startswith("tier_a_"):
        return "A"
    if q.startswith("tier_b_"):
        return "B"
    if q.startswith("tier_c_"):
        return "C"
    return "unknown"


def _compute_control_coverage(samples: pd.DataFrame) -> dict[str, Any]:
    if samples.empty:
        return {
            "control_cases_by_lead": {},
            "far_nonstorm_controls_by_lead": {},
            "total_control_cases": 0,
            "total_far_nonstorm_controls": 0,
        }
    grp = (
        samples.groupby(["case_id", "lead_bucket", "case_type"], as_index=False)
        .agg(
            storm_max=("storm", "max"),
            near_storm_max=("near_storm", "max"),
            pregen_max=("pregen", "max"),
        )
    )
    controls = grp.loc[grp["case_type"].astype(str) == "control"].copy()
    if controls.empty:
        return {
            "control_cases_by_lead": {},
            "far_nonstorm_controls_by_lead": {},
            "total_control_cases": 0,
            "total_far_nonstorm_controls": 0,
        }
    controls["lead_bucket"] = controls["lead_bucket"].astype(str)
    far = controls.loc[
        (pd.to_numeric(controls["storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(controls["near_storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(controls["pregen_max"], errors="coerce").fillna(0) == 0)
    ].copy()
    control_by_lead = controls.groupby("lead_bucket")["case_id"].nunique().to_dict()
    far_by_lead = far.groupby("lead_bucket")["case_id"].nunique().to_dict()
    return {
        "control_cases_by_lead": {str(k): int(v) for k, v in sorted(control_by_lead.items())},
        "far_nonstorm_controls_by_lead": {str(k): int(v) for k, v in sorted(far_by_lead.items())},
        "total_control_cases": int(controls["case_id"].nunique()),
        "total_far_nonstorm_controls": int(far["case_id"].nunique()),
    }


def _lead_case_counts(samples: pd.DataFrame) -> dict[str, int]:
    if samples.empty or "lead_bucket" not in samples.columns or "case_id" not in samples.columns:
        return {}
    grp = (
        samples[["lead_bucket", "case_id"]]
        .dropna(subset=["lead_bucket", "case_id"])
        .drop_duplicates()
        .groupby("lead_bucket")["case_id"]
        .nunique()
        .to_dict()
    )
    return {str(k): int(v) for k, v in sorted(grp.items())}


def _build_eta_long(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "control_tier",
                "lead_bucket",
                "t",
                "L",
                "omega",
                "eta",
                "eta_sign",
                "O_L",
                "O_R",
            ]
        )
    time_col = "t" if "t" in samples.columns else ("t_valid" if "t_valid" in samples.columns else ("time" if "time" in samples.columns else None))
    if time_col is None:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "control_tier",
                "lead_bucket",
                "t",
                "L",
                "omega",
                "eta",
                "eta_sign",
                "O_L",
                "O_R",
            ]
        )
    pivot = samples.pivot_table(
        index=[
            "case_id",
            "case_type",
            "control_tier",
            "lead_bucket",
            time_col,
            "L",
            "omega",
        ],
        columns="hand",
        values="O",
        aggfunc="first",
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={"L": "O_L", "R": "O_R"}).reset_index()
    if "O_L" not in pivot.columns:
        pivot["O_L"] = np.nan
    if "O_R" not in pivot.columns:
        pivot["O_R"] = np.nan
    o_l = pd.to_numeric(pivot["O_L"], errors="coerce").to_numpy(dtype=float)
    o_r = pd.to_numeric(pivot["O_R"], errors="coerce").to_numpy(dtype=float)
    denom = 0.5 * (o_l + o_r)
    valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
    eta = np.full(o_l.shape, np.nan, dtype=float)
    eta_sign = np.full(o_l.shape, np.nan, dtype=float)
    eta[valid] = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
    eta_sign[valid] = np.sign(o_l[valid] - o_r[valid])
    pivot["eta"] = eta
    pivot["eta_sign"] = eta_sign
    if time_col != "t":
        pivot = pivot.rename(columns={time_col: "t"})
    return pivot.dropna(subset=["eta"]).reset_index(drop=True)


def _build_case_windows(
    samples: pd.DataFrame,
    *,
    eta_threshold: float,
    block_hours: float,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "lead_bucket",
                "center_source",
                "control_tier",
                "match_quality",
                "storm_id",
                "time_basis",
                "eventness",
                "farness",
                "A_align",
                "S_align",
                "A_align_event",
                "A_align_far",
                "S_align_event",
                "S_align_far",
                "n_time_steps",
                "n_blocks",
                "ib_event_strict",
                "ib_far_strict",
                "ib_far_quality_tag",
            ]
        )
    if "hand" not in samples.columns or "O" not in samples.columns:
        return pd.DataFrame()

    time_col = "t" if "t" in samples.columns else ("t_valid" if "t_valid" in samples.columns else None)
    if time_col is None:
        return pd.DataFrame()

    base = samples.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col, "case_id", "L"]).copy()
    if base.empty:
        return pd.DataFrame()

    base["L_scale"] = pd.to_numeric(base["L"], errors="coerce")
    pair = base.pivot_table(
        index=["case_id", time_col, "L_scale"],
        columns="hand",
        values="O",
        aggfunc="first",
    ).reset_index()
    pair = pair.rename(columns={"L_scale": "L_km", "L": "O_L", "R": "O_R"})
    if "O_L" not in pair.columns or "O_R" not in pair.columns:
        return pd.DataFrame()

    o_l = pd.to_numeric(pair["O_L"], errors="coerce").to_numpy(dtype=float)
    o_r = pd.to_numeric(pair["O_R"], errors="coerce").to_numpy(dtype=float)
    denom = 0.5 * (o_l + o_r)
    valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
    eta = np.full(o_l.shape, np.nan, dtype=float)
    eta[valid] = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
    pair["eta_tl"] = eta
    pair = pair.dropna(subset=["eta_tl"]).copy()
    if pair.empty:
        return pd.DataFrame()

    t_key = time_col
    time_eta = (
        pair.groupby(["case_id", t_key], as_index=False)
        .agg(
            eta_t=("eta_tl", "median"),
            n_scales=("eta_tl", "count"),
        )
        .sort_values(["case_id", t_key])
    )

    meta_cols: dict[str, tuple[str, str]] = {
        "lead_bucket": ("lead_bucket", "first"),
        "case_type": ("case_type", "first"),
        "center_source": ("center_source", "first"),
        "control_tier": ("control_tier", "first"),
        "match_quality": ("match_quality", "first"),
        "storm_id": ("storm_id", "first"),
        "time_basis": ("time_basis", "first"),
        "ib_event_strict": ("ib_event_strict", "max"),
        "ib_far_strict": ("ib_far_strict", "max"),
        "ib_far_quality_tag": ("ib_far_quality_tag", "first"),
        "w_event_t": ("w_event", "median"),
        "w_far_t": ("w_far", "median"),
    }
    avail_meta = {k: v for k, v in meta_cols.items() if v[0] in base.columns}
    meta = base.groupby(["case_id", t_key], as_index=False).agg(**avail_meta) if avail_meta else pd.DataFrame()
    if not meta.empty:
        time_eta = time_eta.merge(meta, on=["case_id", t_key], how="left")
    else:
        time_eta["w_event_t"] = 0.0
        time_eta["w_far_t"] = 0.0

    thr = max(float(eta_threshold), 0.0)
    time_eta["parity_pass_t"] = pd.to_numeric(time_eta["eta_t"], errors="coerce").fillna(0.0) >= thr
    bh = max(float(block_hours), 0.0)
    if bh > 0:
        block_freq = f"{max(int(round(bh)), 1)}h"
        time_eta["time_block"] = pd.to_datetime(time_eta[t_key], errors="coerce").dt.floor(block_freq)
    else:
        time_eta["time_block"] = pd.to_datetime(time_eta[t_key], errors="coerce")
    block_df = (
        time_eta.groupby(["case_id", "time_block"], as_index=False)
        .agg(
            parity_pass_block=("parity_pass_t", "mean"),
            eta_block=("eta_t", "mean"),
            w_event_block=("w_event_t", "mean"),
            w_far_block=("w_far_t", "mean"),
        )
    )
    block_df["parity_pass_block"] = pd.to_numeric(block_df["parity_pass_block"], errors="coerce").fillna(0.0) >= 0.5

    case_rows: list[dict[str, Any]] = []
    for case_id, grp in time_eta.groupby("case_id", dropna=False):
        event_w = pd.to_numeric(grp.get("w_event_t"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        far_w = pd.to_numeric(grp.get("w_far_t"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pass_t = pd.to_numeric(grp.get("parity_pass_t"), errors="coerce").fillna(0).to_numpy(dtype=float)
        eta_t = pd.to_numeric(grp.get("eta_t"), errors="coerce").to_numpy(dtype=float)
        block_grp = block_df.loc[block_df["case_id"].astype(str) == str(case_id)].copy()
        rec = {
            "case_id": str(case_id),
            "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns and grp["case_type"].notna().any() else None,
            "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns and grp["lead_bucket"].notna().any() else None,
            "center_source": str(grp["center_source"].iloc[0]) if "center_source" in grp.columns and grp["center_source"].notna().any() else None,
            "control_tier": str(grp["control_tier"].iloc[0]) if "control_tier" in grp.columns and grp["control_tier"].notna().any() else None,
            "match_quality": str(grp["match_quality"].iloc[0]) if "match_quality" in grp.columns and grp["match_quality"].notna().any() else None,
            "storm_id": str(grp["storm_id"].iloc[0]) if "storm_id" in grp.columns and grp["storm_id"].notna().any() else None,
            "time_basis": str(grp["time_basis"].iloc[0]) if "time_basis" in grp.columns and grp["time_basis"].notna().any() else None,
            "eventness": float(np.nanmean(event_w)) if event_w.size else 0.0,
            "farness": float(np.nanmean(far_w)) if far_w.size else 0.0,
            "A_align": float(np.nanmean(pass_t)) if pass_t.size else 0.0,
            "S_align": float(np.nanmean(eta_t)) if eta_t.size else 0.0,
            "A_align_event": _weighted_mean(pass_t, event_w),
            "A_align_far": _weighted_mean(pass_t, far_w),
            "S_align_event": _weighted_mean(eta_t, event_w),
            "S_align_far": _weighted_mean(eta_t, far_w),
            "n_time_steps": int(grp.shape[0]),
            "n_blocks": int(block_grp.shape[0]),
            "ib_event_strict": bool(pd.to_numeric(grp.get("ib_event_strict"), errors="coerce").fillna(0).max() > 0)
            if "ib_event_strict" in grp.columns
            else False,
            "ib_far_strict": bool(pd.to_numeric(grp.get("ib_far_strict"), errors="coerce").fillna(0).max() > 0)
            if "ib_far_strict" in grp.columns
            else False,
            "ib_far_quality_tag": str(grp["ib_far_quality_tag"].iloc[0])
            if "ib_far_quality_tag" in grp.columns and grp["ib_far_quality_tag"].notna().any()
            else None,
        }
        case_rows.append(rec)
    return pd.DataFrame(case_rows).sort_values(["case_id"]).reset_index(drop=True)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(v[valid] * w[valid]) / max(np.sum(w[valid]), 1e-12))


def _build_scale_audit(case_manifest_all: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_manifest_all:
        return {"n_cases": 0, "by_case": [], "summary": {}}
    frame = pd.DataFrame(case_manifest_all).copy()
    if frame.empty:
        return {"n_cases": 0, "by_case": [], "summary": {}}
    for col, default in (
        ("control_tier", None),
        ("excluded_reason", None),
        ("rows_generated", 0),
        ("lead_bucket", None),
        ("n_scales_total", 0),
        ("n_scales_valid", 0),
        ("scales_used_cells", []),
        ("scale_point_counts", {}),
        ("scale_gate_pass", False),
    ):
        if col not in frame.columns:
            frame[col] = default
    frame["n_scales_valid"] = pd.to_numeric(frame.get("n_scales_valid"), errors="coerce").fillna(0).astype(int)
    frame["scale_gate_pass"] = pd.to_numeric(frame.get("scale_gate_pass"), errors="coerce").fillna(0).astype(bool)
    by_case = frame[
        [
            "case_id",
            "case_type",
            "control_tier",
            "lead_bucket",
            "n_scales_total",
            "n_scales_valid",
            "scales_used_cells",
            "scale_point_counts",
            "scale_gate_pass",
            "excluded_reason",
            "rows_generated",
        ]
    ].copy()
    lead_summary = (
        frame.groupby("lead_bucket", dropna=False)
        .agg(
            n_cases=("case_id", "nunique"),
            n_scale_gate_pass=("scale_gate_pass", "sum"),
            n_scales_valid_median=("n_scales_valid", "median"),
        )
        .reset_index()
    )
    return {
        "n_cases": int(frame["case_id"].nunique()),
        "scale_gate_pass_rate": float(frame["scale_gate_pass"].mean()) if frame.shape[0] else 0.0,
        "n_scales_valid_median": float(frame["n_scales_valid"].median()) if frame.shape[0] else 0.0,
        "by_lead": lead_summary.to_dict(orient="records"),
        "by_case": by_case.to_dict(orient="records"),
    }


def _finite_positive(value: Any, eps: float = 0.0, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out) or out <= 0.0:
        return default
    return float(out + float(eps))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _resolve_centers_mode(*, centers: str | None, event_source: str | None) -> str:
    if event_source:
        legacy = str(event_source).strip().lower()
        mapping = {
            "ibtracs": "ibtracs",
            "vortex": "vortex",
            "labels": "labels",
            "vortex_or_labels": "hybrid",
        }
        return mapping.get(legacy, str(centers or "hybrid").lower())
    return str(centers or "hybrid").strip().lower()


def _build_ibtracs_index(ib_points: pd.DataFrame):
    if ib_points is None or ib_points.empty:
        return None
    ib = ib_points.copy()
    if "time_utc" not in ib.columns:
        if "time" in ib.columns:
            ib["time_utc"] = pd.to_datetime(ib["time"], errors="coerce")
        else:
            return None
    else:
        ib["time_utc"] = pd.to_datetime(ib["time_utc"], errors="coerce")
    if "sid" not in ib.columns:
        if "storm_id" in ib.columns:
            ib["sid"] = ib["storm_id"].astype(str)
        else:
            return None
    if "lat" not in ib.columns:
        if "lat0" in ib.columns:
            ib["lat"] = pd.to_numeric(ib["lat0"], errors="coerce")
        else:
            return None
    if "lon" not in ib.columns:
        if "lon0" in ib.columns:
            ib["lon"] = normalize_lon_180(pd.to_numeric(ib["lon0"], errors="coerce").to_numpy(dtype=float))
        else:
            return None
    for col in ("name", "basin", "wind", "pres", "speed_kmh", "dt_to_nearest_ib_hours", "interp_flag", "center_uncert_km"):
        if col not in ib.columns:
            ib[col] = None
    base = ib[
        [
            "sid",
            "time_utc",
            "lat",
            "lon",
            "name",
            "basin",
            "wind",
            "pres",
            "speed_kmh",
            "dt_to_nearest_ib_hours",
            "interp_flag",
            "center_uncert_km",
        ]
    ].copy()
    base = base.dropna(subset=["sid", "time_utc", "lat", "lon"]).reset_index(drop=True)
    if base.empty:
        return None
    return build_track_index(base)


def _ibtracs_index_meta(index: Any) -> dict[str, Any]:
    if index is None:
        return {"available": False}
    pts = index.points if hasattr(index, "points") else pd.DataFrame()
    if pts is None or pts.empty:
        return {"available": False}
    t = pd.to_datetime(pts.get("time_utc"), errors="coerce")
    return {
        "available": True,
        "n_points": int(pts.shape[0]),
        "n_sid": int(pts["sid"].astype(str).nunique()) if "sid" in pts.columns else 0,
        "n_hour_buckets": int(len(getattr(index, "by_time", {}) or {})),
        "time_min": str(t.min()) if t.notna().any() else None,
        "time_max": str(t.max()) if t.notna().any() else None,
    }


def _storm_center_uncert_km(
    *,
    center_uncert_km: float | None = None,
    speed_kmh: float | None = None,
    dt_fix_h: float | None = None,
    multiplier: float = 0.5,
    cap_km: float = 500.0,
) -> float:
    if center_uncert_km is not None:
        try:
            v = float(center_uncert_km)
            if np.isfinite(v):
                return float(min(max(v, 0.0), float(cap_km)))
        except Exception:
            pass
    try:
        spd = float(speed_kmh) if speed_kmh is not None else np.nan
        dt = float(dt_fix_h) if dt_fix_h is not None else np.nan
    except Exception:
        spd = np.nan
        dt = np.nan
    if np.isfinite(spd) and np.isfinite(dt):
        est = float(max(0.0, float(multiplier) * spd * dt))
        return float(min(est, float(cap_km)))
    return 0.0


def _annotate_cases_with_storm_distance(
    cases: list[dict[str, Any]],
    *,
    ib_index: Any,
    tolerance_hours: float,
    event_km: float,
    near_km: float,
    far_km: float,
    far_loose_km: float,
    exclusion_window_hours: float,
    use_valid_time_shift: bool,
    far_kinematic_speed_max_ms: float,
    far_kinematic_zeta_abs_max: float,
    strict_dt_max_hours: float = 1.0,
    soft_event_r0_km: float = 300.0,
    soft_far_r_km: float = 1300.0,
    soft_time_td0_hours: float = 1.5,
    no_storm_implies_far: bool = True,
    center_uncert_multiplier: float = 0.5,
    center_uncert_cap_km: float = 500.0,
) -> list[dict[str, Any]]:
    if not cases:
        return cases
    if ib_index is None:
        return [dict(c) for c in cases]
    q_base = pd.DataFrame(
        {
            "case_id": [str(c.get("case_id")) for c in cases],
            "source_time": [pd.Timestamp(c.get("time0")) for c in cases],
            "lead_bucket": [str(c.get("lead_bucket")) for c in cases],
            "lat0": [pd.to_numeric(c.get("lat0"), errors="coerce") for c in cases],
            "lon0": [pd.to_numeric(c.get("lon0"), errors="coerce") for c in cases],
        }
    )
    q_base["valid_time"] = [
        _valid_time_from_components(
            time_value=row["source_time"],
            lead_bucket=row["lead_bucket"],
            use_lead_shift=True,
        )
        for _, row in q_base.iterrows()
    ]
    q_source = (
        q_base[["case_id", "lead_bucket", "lat0", "lon0", "source_time"]]
        .rename(columns={"source_time": "time"})
        .copy()
    )
    q_valid = (
        q_base[["case_id", "lead_bucket", "lat0", "lon0", "valid_time"]]
        .rename(columns={"valid_time": "time"})
        .copy()
    )
    ann_source = _nearest_storm_lookup(
        queries=q_source,
        ib_index=ib_index,
        tolerance_hours=float(tolerance_hours),
        exclusion_window_hours=float(exclusion_window_hours),
        no_storm_implies_far=bool(no_storm_implies_far),
    )
    ann_valid = _nearest_storm_lookup(
        queries=q_valid,
        ib_index=ib_index,
        tolerance_hours=float(tolerance_hours),
        exclusion_window_hours=float(exclusion_window_hours),
        no_storm_implies_far=bool(no_storm_implies_far),
    )
    key_cols = ["case_id", "lead_bucket", "lat0", "lon0"]
    ann_source = ann_source.rename(
        columns={
            c: f"{c}_source"
            for c in ann_source.columns
            if c not in key_cols and c != "time"
        }
    )
    ann_source = ann_source.rename(columns={"time": "source_time"})
    ann_valid = ann_valid.rename(
        columns={
            c: f"{c}_valid"
            for c in ann_valid.columns
            if c not in key_cols and c != "time"
        }
    )
    ann_valid = ann_valid.rename(columns={"time": "valid_time"})
    ann = q_base.merge(ann_source, on=key_cols + ["source_time"], how="left")
    ann = ann.merge(ann_valid, on=key_cols + ["valid_time"], how="left")

    selected_basis = "valid" if bool(use_valid_time_shift) else "source"

    def _is_clean(rec: dict[str, Any]) -> bool | None:
        speed = _to_float(rec.get("speed_bin"))
        zeta = _to_float(rec.get("zeta_bin"))
        checks: list[bool] = []
        if speed is not None:
            checks.append(float(speed) <= float(far_kinematic_speed_max_ms))
        if zeta is not None:
            checks.append(abs(float(zeta)) <= float(far_kinematic_zeta_abs_max))
        if not checks:
            return None
        return bool(all(checks))

    out: list[dict[str, Any]] = []
    by_case = {(str(r["case_id"]), str(r.get("lead_bucket", ""))): r for r in ann.to_dict(orient="records")}
    for c in cases:
        cc = dict(c)
        key = (str(cc.get("case_id")), str(cc.get("lead_bucket", "")))
        rec = by_case.get(key)
        d_source = (
            float(rec.get("nearest_storm_distance_km_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_distance_km_source", np.nan)))
            else np.nan
        )
        dt_source = (
            float(rec.get("nearest_storm_time_delta_h_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_time_delta_h_source", np.nan)))
            else np.nan
        )
        dt_fix_source = (
            float(rec.get("nearest_storm_dt_fix_h_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_dt_fix_h_source", np.nan)))
            else np.nan
        )
        speed_source = (
            float(rec.get("nearest_storm_speed_kmh_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_speed_kmh_source", np.nan)))
            else np.nan
        )
        uncert_source_raw = (
            float(rec.get("nearest_storm_center_uncert_km_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_center_uncert_km_source", np.nan)))
            else np.nan
        )
        min_source = (
            float(rec.get("nearest_storm_min_dist_window_km_source"))
            if rec and np.isfinite(float(rec.get("nearest_storm_min_dist_window_km_source", np.nan)))
            else np.nan
        )
        d_valid = (
            float(rec.get("nearest_storm_distance_km_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_distance_km_valid", np.nan)))
            else np.nan
        )
        dt_valid = (
            float(rec.get("nearest_storm_time_delta_h_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_time_delta_h_valid", np.nan)))
            else np.nan
        )
        dt_fix_valid = (
            float(rec.get("nearest_storm_dt_fix_h_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_dt_fix_h_valid", np.nan)))
            else np.nan
        )
        speed_valid = (
            float(rec.get("nearest_storm_speed_kmh_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_speed_kmh_valid", np.nan)))
            else np.nan
        )
        uncert_valid_raw = (
            float(rec.get("nearest_storm_center_uncert_km_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_center_uncert_km_valid", np.nan)))
            else np.nan
        )
        min_valid = (
            float(rec.get("nearest_storm_min_dist_window_km_valid"))
            if rec and np.isfinite(float(rec.get("nearest_storm_min_dist_window_km_valid", np.nan)))
            else np.nan
        )
        if selected_basis == "valid":
            d = d_valid
            dt_h = dt_valid
            min_dist_window = min_valid
            center_uncert_km = _storm_center_uncert_km(
                center_uncert_km=uncert_valid_raw,
                speed_kmh=speed_valid,
                dt_fix_h=dt_fix_valid,
                multiplier=float(center_uncert_multiplier),
                cap_km=float(center_uncert_cap_km),
            )
            sid = rec.get("nearest_storm_id_valid") if rec else None
            sname = rec.get("nearest_storm_name_valid") if rec else None
            sbasin = rec.get("nearest_storm_basin_valid") if rec else None
            slat = rec.get("nearest_storm_lat_valid") if rec else None
            slon = rec.get("nearest_storm_lon_valid") if rec else None
            stime = rec.get("nearest_storm_fix_time_valid") if rec else None
        else:
            d = d_source
            dt_h = dt_source
            min_dist_window = min_source
            center_uncert_km = _storm_center_uncert_km(
                center_uncert_km=uncert_source_raw,
                speed_kmh=speed_source,
                dt_fix_h=dt_fix_source,
                multiplier=float(center_uncert_multiplier),
                cap_km=float(center_uncert_cap_km),
            )
            sid = rec.get("nearest_storm_id_source") if rec else None
            sname = rec.get("nearest_storm_name_source") if rec else None
            sbasin = rec.get("nearest_storm_basin_source") if rec else None
            slat = rec.get("nearest_storm_lat_source") if rec else None
            slon = rec.get("nearest_storm_lon_source") if rec else None
            stime = rec.get("nearest_storm_fix_time_source") if rec else None

        ib_event_source = bool(np.isfinite(d_source) and np.isfinite(dt_source) and d_source <= float(event_km) and dt_source <= float(tolerance_hours))
        ib_far_source = bool(np.isfinite(min_source) and min_source >= float(far_km))
        ib_event_valid = bool(np.isfinite(d_valid) and np.isfinite(dt_valid) and d_valid <= float(event_km) and dt_valid <= float(tolerance_hours))
        ib_far_valid = bool(np.isfinite(min_valid) and min_valid >= float(far_km))

        ib_event = ib_event_valid if selected_basis == "valid" else ib_event_source
        ib_far = ib_far_valid if selected_basis == "valid" else ib_far_source
        strict_dt_ok = bool(np.isfinite(dt_h) and (float(dt_h) <= float(strict_dt_max_hours)))
        event_thresh_km = max(0.0, float(event_km) - float(center_uncert_km))
        far_strict_km = max(float(far_loose_km), float(far_km) + float(center_uncert_km))
        ib_event_strict = bool(ib_event and strict_dt_ok and np.isfinite(d) and (float(d) <= event_thresh_km))
        ib_far_strict = bool(
            np.isfinite(min_dist_window)
            and (float(min_dist_window) >= float(far_strict_km))
            and (not bool(ib_event_strict))
        )
        w_event = _soft_event_weight(
            dist_km=d,
            dt_hours=dt_h,
            r0_km=float(soft_event_r0_km),
            td0_hours=float(soft_time_td0_hours),
        )
        w_far = _soft_far_weight(
            dist_km=d,
            dt_hours=dt_h,
            r_far_km=float(soft_far_r_km),
            td0_hours=float(soft_time_td0_hours),
        )
        kinematic_clean = _is_clean(cc)
        far_quality_tag = _ib_far_quality_tag(
            dist_km=d,
            min_dist_window_km=min_dist_window,
            far_km=float(far_km),
            far_loose_km=float(max(far_loose_km, far_km)),
            ib_far=ib_far,
            ib_far_strict=ib_far_strict,
            kinematic_clean=kinematic_clean,
        )
        cohort = _distance_cohort_from_flags(
            dist_km=d,
            event_km=event_km,
            near_km=near_km,
            ib_event=ib_event_strict,
            ib_far=ib_far_strict,
        )
        cc["nearest_storm_distance_km"] = d
        cc["nearest_storm_time_delta_h"] = dt_h
        cc["nearest_storm_min_dist_window_km"] = min_dist_window
        cc["ib_min_dist_any_storm_km"] = min_dist_window
        cc["storm_center_uncert_km"] = float(center_uncert_km)
        cc["nearest_storm_id"] = sid
        cc["nearest_storm_name"] = sname
        cc["nearest_storm_basin"] = sbasin
        cc["nearest_storm_lat"] = _to_float(slat)
        cc["nearest_storm_lon"] = _to_float(slon)
        cc["nearest_storm_fix_time"] = pd.to_datetime(stime, errors="coerce") if stime is not None else pd.NaT
        cc["nearest_storm_distance_km_source"] = d_source
        cc["nearest_storm_time_delta_h_source"] = dt_source
        cc["nearest_storm_dt_fix_h_source"] = dt_fix_source
        cc["nearest_storm_speed_kmh_source"] = speed_source
        cc["nearest_storm_center_uncert_km_source"] = _storm_center_uncert_km(
            center_uncert_km=uncert_source_raw,
            speed_kmh=speed_source,
            dt_fix_h=dt_fix_source,
            multiplier=float(center_uncert_multiplier),
            cap_km=float(center_uncert_cap_km),
        )
        cc["nearest_storm_min_dist_window_km_source"] = min_source
        cc["ib_min_dist_any_storm_km_source"] = min_source
        cc["nearest_storm_id_source"] = rec.get("nearest_storm_id_source") if rec else None
        cc["nearest_storm_distance_km_valid"] = d_valid
        cc["nearest_storm_time_delta_h_valid"] = dt_valid
        cc["nearest_storm_dt_fix_h_valid"] = dt_fix_valid
        cc["nearest_storm_speed_kmh_valid"] = speed_valid
        cc["nearest_storm_center_uncert_km_valid"] = _storm_center_uncert_km(
            center_uncert_km=uncert_valid_raw,
            speed_kmh=speed_valid,
            dt_fix_h=dt_fix_valid,
            multiplier=float(center_uncert_multiplier),
            cap_km=float(center_uncert_cap_km),
        )
        cc["nearest_storm_min_dist_window_km_valid"] = min_valid
        cc["ib_min_dist_any_storm_km_valid"] = min_valid
        cc["nearest_storm_id_valid"] = rec.get("nearest_storm_id_valid") if rec else None
        cc["storm_distance_cohort"] = cohort
        cc["storm_phase"] = cohort
        cc["storm_dist_km"] = d
        cc["t_source"] = pd.to_datetime(rec.get("source_time"), errors="coerce") if rec else pd.to_datetime(cc.get("time0"), errors="coerce")
        cc["valid_time"] = pd.to_datetime(rec.get("valid_time"), errors="coerce") if rec else _valid_time_from_components(
            cc.get("time0"),
            cc.get("lead_bucket"),
            use_lead_shift=True,
        )
        cc["time_basis"] = selected_basis
        cc["ib_time_basis_used"] = selected_basis
        cc["storm_match_time"] = cc["valid_time"] if selected_basis == "valid" else cc["t_source"]
        cc["ib_sid"] = sid
        cc["ib_name"] = sname
        cc["ib_basin"] = sbasin
        cc["ib_dt_hours"] = dt_h
        cc["ib_dt_to_fix_hours"] = dt_fix_valid if selected_basis == "valid" else dt_fix_source
        cc["ib_has_fix_window"] = bool(
            np.isfinite(cc["ib_dt_to_fix_hours"]) and (float(cc["ib_dt_to_fix_hours"]) <= float(tolerance_hours))
        )
        cc["dt_to_nearest_ib_hours"] = dt_h
        cc["ib_dist_km"] = d
        cc["w_event"] = w_event
        cc["w_far"] = w_far
        cc["eventness"] = w_event
        cc["farness"] = w_far
        cc["ib_event"] = bool(ib_event)
        cc["ib_far"] = bool(ib_far)
        cc["ib_event_source"] = bool(ib_event_source)
        cc["ib_far_source"] = bool(ib_far_source)
        cc["ib_event_valid"] = bool(ib_event_valid)
        cc["ib_far_valid"] = bool(ib_far_valid)
        cc["ib_event_strict"] = bool(ib_event_strict)
        cc["ib_far_strict"] = bool(ib_far_strict)
        cc["ib_strict_dt_max_hours"] = float(strict_dt_max_hours)
        cc["ib_far_quality_tag"] = far_quality_tag
        cc["ib_kinematic_clean"] = kinematic_clean
        cc["ib_conflict_flag"] = _ib_conflict_flag(
            dist_km=d,
            dt_hours=dt_h,
            far_km=float(far_km),
            dt_max_hours=float(tolerance_hours),
            ib_event=ib_event,
            ib_far=ib_far,
        )
        if sid is not None:
            cc["storm_id"] = str(sid)
        out.append(cc)
    return out


def _annotate_samples_with_storm_distance(
    *,
    samples: pd.DataFrame,
    ib_index: Any,
    tolerance_hours: float,
    event_km: float,
    near_km: float,
    far_km: float,
    far_loose_km: float,
    exclusion_window_hours: float,
    use_valid_time_shift: bool,
    far_kinematic_speed_max_ms: float,
    far_kinematic_zeta_abs_max: float,
    strict_dt_max_hours: float = 1.0,
    soft_event_r0_km: float = 300.0,
    soft_far_r_km: float = 1300.0,
    soft_time_td0_hours: float = 1.5,
    no_storm_implies_far: bool = True,
    center_uncert_multiplier: float = 0.5,
    center_uncert_cap_km: float = 500.0,
) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    out = samples.copy()
    out["t"] = pd.to_datetime(out["t"], errors="coerce")
    out["t_source"] = pd.to_datetime(out["t"], errors="coerce")
    out["t_valid"] = [
        _valid_time_from_components(
            time_value=t,
            lead_bucket=lead,
            use_lead_shift=True,
        )
        for t, lead in zip(out["t"], out.get("lead_bucket", pd.Series([None] * len(out))), strict=False)
    ]
    selected_basis = "valid" if bool(use_valid_time_shift) else "source"
    out["time_basis"] = selected_basis
    out["ib_time_basis_used"] = selected_basis
    if ib_index is None:
        return out
    q_source = (
        out[["case_id", "t_source", "lead_bucket", "lat0", "lon0"]]
        .drop_duplicates(subset=["case_id", "t_source", "lead_bucket", "lat0", "lon0"])
        .rename(columns={"t_source": "time"})
        .reset_index(drop=True)
    )
    q_valid = (
        out[["case_id", "t_valid", "lead_bucket", "lat0", "lon0"]]
        .drop_duplicates(subset=["case_id", "t_valid", "lead_bucket", "lat0", "lon0"])
        .rename(columns={"t_valid": "time"})
        .reset_index(drop=True)
    )
    ann_source = _nearest_storm_lookup(
        queries=q_source,
        ib_index=ib_index,
        tolerance_hours=float(tolerance_hours),
        exclusion_window_hours=float(exclusion_window_hours),
        no_storm_implies_far=bool(no_storm_implies_far),
    )
    ann_valid = _nearest_storm_lookup(
        queries=q_valid,
        ib_index=ib_index,
        tolerance_hours=float(tolerance_hours),
        exclusion_window_hours=float(exclusion_window_hours),
        no_storm_implies_far=bool(no_storm_implies_far),
    )
    ann_source = ann_source.rename(
        columns={
            c: f"{c}_source"
            for c in ann_source.columns
            if c not in {"case_id", "lead_bucket", "lat0", "lon0", "time"}
        }
    ).rename(columns={"time": "t_source"})
    ann_valid = ann_valid.rename(
        columns={
            c: f"{c}_valid"
            for c in ann_valid.columns
            if c not in {"case_id", "lead_bucket", "lat0", "lon0", "time"}
        }
    ).rename(columns={"time": "t_valid"})
    out = out.merge(ann_source, on=["case_id", "t_source", "lead_bucket", "lat0", "lon0"], how="left")
    out = out.merge(ann_valid, on=["case_id", "t_valid", "lead_bucket", "lat0", "lon0"], how="left")

    out["ib_dist_km_source"] = pd.to_numeric(out.get("nearest_storm_distance_km_source"), errors="coerce")
    out["ib_dt_hours_source"] = pd.to_numeric(out.get("nearest_storm_time_delta_h_source"), errors="coerce")
    out["ib_dt_to_fix_hours_source"] = pd.to_numeric(out.get("nearest_storm_dt_fix_h_source"), errors="coerce")
    out["ib_speed_kmh_source"] = pd.to_numeric(out.get("nearest_storm_speed_kmh_source"), errors="coerce")
    out["ib_center_uncert_km_source"] = pd.to_numeric(out.get("nearest_storm_center_uncert_km_source"), errors="coerce")
    out["ib_dist_km_valid"] = pd.to_numeric(out.get("nearest_storm_distance_km_valid"), errors="coerce")
    out["ib_dt_hours_valid"] = pd.to_numeric(out.get("nearest_storm_time_delta_h_valid"), errors="coerce")
    out["ib_dt_to_fix_hours_valid"] = pd.to_numeric(out.get("nearest_storm_dt_fix_h_valid"), errors="coerce")
    out["ib_speed_kmh_valid"] = pd.to_numeric(out.get("nearest_storm_speed_kmh_valid"), errors="coerce")
    out["ib_center_uncert_km_valid"] = pd.to_numeric(out.get("nearest_storm_center_uncert_km_valid"), errors="coerce")

    out["ib_event_source"] = (
        np.isfinite(out["ib_dist_km_source"])
        & np.isfinite(out["ib_dt_hours_source"])
        & (out["ib_dist_km_source"] <= float(event_km))
        & (out["ib_dt_hours_source"] <= float(tolerance_hours))
    )
    out["ib_event_valid"] = (
        np.isfinite(out["ib_dist_km_valid"])
        & np.isfinite(out["ib_dt_hours_valid"])
        & (out["ib_dist_km_valid"] <= float(event_km))
        & (out["ib_dt_hours_valid"] <= float(tolerance_hours))
    )
    min_dist_window_source = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_source"), errors="coerce")
    min_dist_window_valid = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_valid"), errors="coerce")
    out["ib_far_source"] = np.isfinite(min_dist_window_source) & (min_dist_window_source >= float(far_km))
    out["ib_far_valid"] = np.isfinite(min_dist_window_valid) & (min_dist_window_valid >= float(far_km))

    far_loose = float(max(far_loose_km, far_km))
    strict_dt_source = np.isfinite(out["ib_dt_hours_source"]) & (out["ib_dt_hours_source"] <= float(strict_dt_max_hours))
    strict_dt_valid = np.isfinite(out["ib_dt_hours_valid"]) & (out["ib_dt_hours_valid"] <= float(strict_dt_max_hours))
    unc_source = pd.to_numeric(out["ib_center_uncert_km_source"], errors="coerce")
    unc_valid = pd.to_numeric(out["ib_center_uncert_km_valid"], errors="coerce")
    est_source = float(center_uncert_multiplier) * pd.to_numeric(out["ib_speed_kmh_source"], errors="coerce") * pd.to_numeric(out["ib_dt_to_fix_hours_source"], errors="coerce")
    est_valid = float(center_uncert_multiplier) * pd.to_numeric(out["ib_speed_kmh_valid"], errors="coerce") * pd.to_numeric(out["ib_dt_to_fix_hours_valid"], errors="coerce")
    unc_source = unc_source.where(np.isfinite(unc_source), est_source)
    unc_valid = unc_valid.where(np.isfinite(unc_valid), est_valid)
    unc_source = pd.to_numeric(unc_source, errors="coerce").clip(lower=0.0, upper=float(center_uncert_cap_km)).fillna(0.0)
    unc_valid = pd.to_numeric(unc_valid, errors="coerce").clip(lower=0.0, upper=float(center_uncert_cap_km)).fillna(0.0)
    out["ib_center_uncert_km_source"] = unc_source
    out["ib_center_uncert_km_valid"] = unc_valid
    event_thresh_source = np.maximum(float(event_km) - unc_source.to_numpy(dtype=float), 0.0)
    event_thresh_valid = np.maximum(float(event_km) - unc_valid.to_numpy(dtype=float), 0.0)
    far_thresh_source = np.maximum(float(far_loose), float(far_km) + unc_source.to_numpy(dtype=float))
    far_thresh_valid = np.maximum(float(far_loose), float(far_km) + unc_valid.to_numpy(dtype=float))
    out["ib_event_strict_source"] = out["ib_event_source"].astype(bool) & strict_dt_source & (pd.to_numeric(out["ib_dist_km_source"], errors="coerce").to_numpy(dtype=float) <= event_thresh_source)
    out["ib_event_strict_valid"] = out["ib_event_valid"].astype(bool) & strict_dt_valid & (pd.to_numeric(out["ib_dist_km_valid"], errors="coerce").to_numpy(dtype=float) <= event_thresh_valid)
    out["ib_far_strict_source"] = (
        np.isfinite(min_dist_window_source)
        & (min_dist_window_source.to_numpy(dtype=float) >= far_thresh_source)
        & (~out["ib_event_strict_source"].astype(bool))
    )
    out["ib_far_strict_valid"] = (
        np.isfinite(min_dist_window_valid)
        & (min_dist_window_valid.to_numpy(dtype=float) >= far_thresh_valid)
        & (~out["ib_event_strict_valid"].astype(bool))
    )

    if selected_basis == "valid":
        out["ib_dist_km"] = out["ib_dist_km_valid"]
        out["ib_dt_hours"] = out["ib_dt_hours_valid"]
        out["ib_event"] = out["ib_event_valid"].astype(bool)
        out["ib_far"] = out["ib_far_valid"].astype(bool)
        out["ib_event_strict"] = out["ib_event_strict_valid"].astype(bool)
        out["ib_far_strict"] = out["ib_far_strict_valid"].astype(bool)
        out["nearest_storm_distance_km"] = pd.to_numeric(out.get("nearest_storm_distance_km_valid"), errors="coerce")
        out["nearest_storm_time_delta_h"] = pd.to_numeric(out.get("nearest_storm_time_delta_h_valid"), errors="coerce")
        out["nearest_storm_min_dist_window_km"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_valid"), errors="coerce")
        out["ib_min_dist_any_storm_km"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_valid"), errors="coerce")
        out["nearest_storm_id"] = out.get("nearest_storm_id_valid")
        out["nearest_storm_name"] = out.get("nearest_storm_name_valid")
        out["nearest_storm_basin"] = out.get("nearest_storm_basin_valid")
        out["nearest_storm_lat"] = pd.to_numeric(out.get("nearest_storm_lat_valid"), errors="coerce")
        out["nearest_storm_lon"] = pd.to_numeric(out.get("nearest_storm_lon_valid"), errors="coerce")
        out["nearest_storm_fix_time"] = pd.to_datetime(out.get("nearest_storm_fix_time_valid"), errors="coerce")
    else:
        out["ib_dist_km"] = out["ib_dist_km_source"]
        out["ib_dt_hours"] = out["ib_dt_hours_source"]
        out["ib_event"] = out["ib_event_source"].astype(bool)
        out["ib_far"] = out["ib_far_source"].astype(bool)
        out["ib_event_strict"] = out["ib_event_strict_source"].astype(bool)
        out["ib_far_strict"] = out["ib_far_strict_source"].astype(bool)
        out["nearest_storm_distance_km"] = pd.to_numeric(out.get("nearest_storm_distance_km_source"), errors="coerce")
        out["nearest_storm_time_delta_h"] = pd.to_numeric(out.get("nearest_storm_time_delta_h_source"), errors="coerce")
        out["nearest_storm_min_dist_window_km"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_source"), errors="coerce")
        out["ib_min_dist_any_storm_km"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_source"), errors="coerce")
        out["nearest_storm_id"] = out.get("nearest_storm_id_source")
        out["nearest_storm_name"] = out.get("nearest_storm_name_source")
        out["nearest_storm_basin"] = out.get("nearest_storm_basin_source")
        out["nearest_storm_lat"] = pd.to_numeric(out.get("nearest_storm_lat_source"), errors="coerce")
        out["nearest_storm_lon"] = pd.to_numeric(out.get("nearest_storm_lon_source"), errors="coerce")
        out["nearest_storm_fix_time"] = pd.to_datetime(out.get("nearest_storm_fix_time_source"), errors="coerce")

    out["storm_match_time"] = out["t_valid"] if selected_basis == "valid" else out["t_source"]
    out["ib_min_dist_any_storm_km_source"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_source"), errors="coerce")
    out["ib_min_dist_any_storm_km_valid"] = pd.to_numeric(out.get("nearest_storm_min_dist_window_km_valid"), errors="coerce")
    out["ib_sid"] = out.get("nearest_storm_id")
    out["ib_name"] = out.get("nearest_storm_name")
    out["ib_basin"] = out.get("nearest_storm_basin")
    out["dt_to_nearest_ib_hours"] = pd.to_numeric(out.get("ib_dt_hours"), errors="coerce")
    out["ib_dt_to_fix_hours"] = (
        pd.to_numeric(out.get("ib_dt_to_fix_hours_valid"), errors="coerce")
        if selected_basis == "valid"
        else pd.to_numeric(out.get("ib_dt_to_fix_hours_source"), errors="coerce")
    )
    out["storm_center_uncert_km"] = (
        pd.to_numeric(out.get("ib_center_uncert_km_valid"), errors="coerce")
        if selected_basis == "valid"
        else pd.to_numeric(out.get("ib_center_uncert_km_source"), errors="coerce")
    )
    out["ib_has_fix_window"] = np.isfinite(pd.to_numeric(out.get("ib_dt_to_fix_hours"), errors="coerce")) & (
        pd.to_numeric(out.get("ib_dt_to_fix_hours"), errors="coerce") <= float(tolerance_hours)
    )
    out["ib_strict_dt_max_hours"] = float(strict_dt_max_hours)
    out["w_event"] = _soft_event_weight_series(
        dist_km=pd.to_numeric(out.get("ib_dist_km"), errors="coerce"),
        dt_hours=pd.to_numeric(out.get("ib_dt_hours"), errors="coerce"),
        r0_km=float(soft_event_r0_km),
        td0_hours=float(soft_time_td0_hours),
    )
    out["w_far"] = _soft_far_weight_series(
        dist_km=pd.to_numeric(out.get("ib_dist_km"), errors="coerce"),
        dt_hours=pd.to_numeric(out.get("ib_dt_hours"), errors="coerce"),
        r_far_km=float(soft_far_r_km),
        td0_hours=float(soft_time_td0_hours),
    )
    out["eventness"] = pd.to_numeric(out["w_event"], errors="coerce")
    out["farness"] = pd.to_numeric(out["w_far"], errors="coerce")

    out["ib_event_strict"] = out["ib_event_strict"].astype(bool)
    far_loose = float(max(far_loose_km, far_km))
    out["ib_far_strict"] = out["ib_far_strict"].astype(bool)
    speed_ref = pd.to_numeric(out.get("case_speed_bin"), errors="coerce")
    zeta_ref = pd.to_numeric(out.get("case_zeta_bin"), errors="coerce")
    kinematic_clean: list[bool | None] = []
    for spd, zet in zip(speed_ref, zeta_ref, strict=False):
        checks: list[bool] = []
        if np.isfinite(float(spd)):
            checks.append(float(spd) <= float(far_kinematic_speed_max_ms))
        if np.isfinite(float(zet)):
            checks.append(abs(float(zet)) <= float(far_kinematic_zeta_abs_max))
        if not checks:
            kinematic_clean.append(None)
        else:
            kinematic_clean.append(bool(all(checks)))
    out["ib_kinematic_clean"] = kinematic_clean
    out["ib_far_quality_tag"] = [
        _ib_far_quality_tag(
            dist_km=d,
            min_dist_window_km=md,
            far_km=float(far_km),
            far_loose_km=far_loose,
            ib_far=bool(far),
            ib_far_strict=bool(far_s),
            kinematic_clean=kc,
        )
        for d, md, far, far_s, kc in zip(
            out["ib_dist_km"],
            pd.to_numeric(out.get("nearest_storm_min_dist_window_km"), errors="coerce"),
            out["ib_far"],
            out["ib_far_strict"],
            out["ib_kinematic_clean"],
            strict=False,
        )
    ]
    out["ib_conflict_flag"] = [
        _ib_conflict_flag(
            dist_km=d,
            dt_hours=dt,
            far_km=float(far_km),
            dt_max_hours=float(tolerance_hours),
            ib_event=bool(ev),
            ib_far=bool(far),
        )
        for d, dt, ev, far in zip(out["ib_dist_km"], out["ib_dt_hours"], out["ib_event"], out["ib_far"], strict=False)
    ]
    out["storm_distance_cohort"] = [
        _distance_cohort_from_flags(
            float(v) if np.isfinite(float(v)) else np.nan,
            event_km=float(event_km),
            near_km=float(near_km),
            ib_event=bool(ev),
            ib_far=bool(far_s),
        )
        for v, ev, far_s in zip(
            pd.to_numeric(out["nearest_storm_distance_km"], errors="coerce").fillna(np.nan).to_numpy(dtype=float),
            out["ib_event_strict"].to_numpy(dtype=bool),
            out["ib_far_strict"].to_numpy(dtype=bool),
            strict=False,
        )
    ]
    out["storm_phase"] = out["storm_distance_cohort"]
    out["storm_dist_km"] = pd.to_numeric(out["nearest_storm_distance_km"], errors="coerce")
    if "nearest_storm_id" in out.columns:
        nearest = out["nearest_storm_id"].astype(str)
        nearest = nearest.where(out["nearest_storm_id"].notna(), None)
        if "storm_id" in out.columns:
            out["storm_id"] = np.where(out["nearest_storm_id"].notna(), nearest, out["storm_id"])
        else:
            out["storm_id"] = nearest

    # Override row labels from deterministic distance-to-track cohorts when IBTrACS is available.
    cohort = out["storm_distance_cohort"].astype(str)
    out["storm"] = (cohort == "event").astype(int)
    out["near_storm"] = (cohort == "near_storm").astype(int)
    out["pregen"] = (cohort == "transition").astype(int)
    out["storm_point"] = out["storm"]
    return out


def _nearest_storm_lookup(
    *,
    queries: pd.DataFrame,
    ib_index: Any,
    tolerance_hours: float,
    exclusion_window_hours: float,
    no_storm_implies_far: bool = True,
) -> pd.DataFrame:
    q = queries.copy()
    q["time"] = pd.to_datetime(q["time"], errors="coerce")
    q = q.dropna(subset=["time", "lat0", "lon0"]).copy()
    if "lead_bucket" not in q.columns:
        q["lead_bucket"] = ""
    if q.empty or ib_index is None:
        q["nearest_storm_distance_km"] = np.nan
        q["nearest_storm_time_delta_h"] = np.nan
        q["nearest_storm_dt_query_h"] = np.nan
        q["nearest_storm_dt_fix_h"] = np.nan
        q["nearest_storm_speed_kmh"] = np.nan
        q["nearest_storm_center_uncert_km"] = np.nan
        q["nearest_storm_interp_flag"] = None
        q["nearest_storm_min_dist_window_km"] = np.nan
        q["nearest_storm_id"] = None
        q["nearest_storm_name"] = None
        q["nearest_storm_basin"] = None
        q["nearest_storm_lat"] = np.nan
        q["nearest_storm_lon"] = np.nan
        q["nearest_storm_fix_time"] = pd.NaT
        q["ib_has_fix_window"] = False
        return q

    nearest_dist: list[float] = []
    nearest_dt: list[float] = []
    nearest_sid: list[str | None] = []
    nearest_name: list[str | None] = []
    nearest_basin: list[str | None] = []
    nearest_lat: list[float] = []
    nearest_lon: list[float] = []
    nearest_fix_time: list[pd.Timestamp | None] = []
    nearest_dt_query_h: list[float] = []
    nearest_dt_fix_h: list[float] = []
    nearest_speed_kmh: list[float] = []
    nearest_interp_flag: list[str | None] = []
    nearest_center_uncert_km: list[float] = []
    min_dist_window: list[float] = []
    for _, row in q.iterrows():
        hit = nearest_ibtracs_fix(
            ib_index,
            time=pd.Timestamp(row["time"]),
            lat=float(row["lat0"]),
            lon=float(row["lon0"]),
            dt_max_hours=float(tolerance_hours),
            r_max_km=None,
        )
        if hit is None:
            nearest_dist.append(np.nan)
            nearest_dt.append(np.nan)
            nearest_sid.append(None)
            nearest_name.append(None)
            nearest_basin.append(None)
            nearest_lat.append(np.nan)
            nearest_lon.append(np.nan)
            nearest_fix_time.append(pd.NaT)
            nearest_dt_query_h.append(np.nan)
            nearest_dt_fix_h.append(np.nan)
            nearest_speed_kmh.append(np.nan)
            nearest_interp_flag.append(None)
            nearest_center_uncert_km.append(np.nan)
        else:
            nearest_dist.append(float(hit.get("dist_km", np.nan)))
            nearest_dt.append(float(hit.get("dt_hours", np.nan)))
            nearest_sid.append(str(hit.get("sid")) if hit.get("sid") is not None else None)
            nearest_name.append(str(hit.get("name")) if hit.get("name") is not None else None)
            nearest_basin.append(str(hit.get("basin")) if hit.get("basin") is not None else None)
            nearest_lat.append(float(hit.get("lat", np.nan)))
            nearest_lon.append(float(hit.get("lon", np.nan)))
            nearest_fix_time.append(pd.to_datetime(hit.get("t_fix"), errors="coerce"))
            nearest_dt_query_h.append(_to_float(hit.get("dt_query_hours")) if _to_float(hit.get("dt_query_hours")) is not None else np.nan)
            nearest_dt_fix_h.append(_to_float(hit.get("dt_fix_hours")) if _to_float(hit.get("dt_fix_hours")) is not None else np.nan)
            nearest_speed_kmh.append(_to_float(hit.get("speed_kmh")) if _to_float(hit.get("speed_kmh")) is not None else np.nan)
            nearest_interp_flag.append(str(hit.get("interp_flag")) if hit.get("interp_flag") is not None else None)
            nearest_center_uncert_km.append(_to_float(hit.get("center_uncert_km")) if _to_float(hit.get("center_uncert_km")) is not None else np.nan)
        min_dist_window.append(
            _min_storm_distance_within_window(
                ib_index=ib_index,
                time=pd.Timestamp(row["time"]),
                lat=float(row["lat0"]),
                lon=float(row["lon0"]),
                window_hours=float(exclusion_window_hours),
                no_storm_implies_far=bool(no_storm_implies_far),
            )
        )

    q["nearest_storm_distance_km"] = nearest_dist
    q["nearest_storm_time_delta_h"] = nearest_dt
    q["nearest_storm_min_dist_window_km"] = min_dist_window
    q["nearest_storm_id"] = nearest_sid
    q["nearest_storm_name"] = nearest_name
    q["nearest_storm_basin"] = nearest_basin
    q["nearest_storm_lat"] = nearest_lat
    q["nearest_storm_lon"] = nearest_lon
    q["nearest_storm_fix_time"] = nearest_fix_time
    q["nearest_storm_dt_query_h"] = nearest_dt_query_h
    q["nearest_storm_dt_fix_h"] = nearest_dt_fix_h
    q["nearest_storm_speed_kmh"] = nearest_speed_kmh
    q["nearest_storm_interp_flag"] = nearest_interp_flag
    q["nearest_storm_center_uncert_km"] = nearest_center_uncert_km
    q["ib_has_fix_window"] = (
        np.isfinite(pd.to_numeric(q["nearest_storm_dt_fix_h"], errors="coerce"))
        & (pd.to_numeric(q["nearest_storm_dt_fix_h"], errors="coerce") <= float(tolerance_hours))
    )
    return q


def _distance_cohort_from_flags(
    dist_km: float,
    *,
    event_km: float,
    near_km: float,
    ib_event: bool,
    ib_far: bool,
) -> str:
    if bool(ib_event):
        return "event"
    if bool(ib_far):
        return "far_nonstorm"
    return _distance_cohort(dist_km, event_km=event_km, near_km=near_km, far_km=np.inf)


def _distance_cohort(d_km: float, *, event_km: float, near_km: float, far_km: float) -> str:
    if not np.isfinite(float(d_km)):
        return "unknown"
    d = float(d_km)
    if d <= float(event_km):
        return "event"
    if d <= float(near_km):
        return "near_storm"
    if d >= float(far_km):
        return "far_nonstorm"
    return "transition"


def _time_confidence_weight(dt_hours: float, td0_hours: float) -> float:
    if not np.isfinite(float(dt_hours)):
        return 0.0
    td = max(float(td0_hours), 1e-6)
    return float(np.exp(-((float(dt_hours) / td) ** 2)))


def _soft_event_weight(*, dist_km: float, dt_hours: float, r0_km: float, td0_hours: float) -> float:
    if not np.isfinite(float(dist_km)):
        return 0.0
    r0 = max(float(r0_km), 1e-6)
    spatial = float(np.exp(-((float(dist_km) / r0) ** 2)))
    return float(spatial * _time_confidence_weight(float(dt_hours), float(td0_hours)))


def _soft_far_weight(*, dist_km: float, dt_hours: float, r_far_km: float, td0_hours: float) -> float:
    if not np.isfinite(float(dist_km)):
        return 0.0
    rf = max(float(r_far_km), 1e-6)
    spatial = float(1.0 - np.exp(-((float(dist_km) / rf) ** 2)))
    return float(spatial * _time_confidence_weight(float(dt_hours), float(td0_hours)))


def _soft_event_weight_series(*, dist_km: pd.Series, dt_hours: pd.Series, r0_km: float, td0_hours: float) -> np.ndarray:
    d = pd.to_numeric(dist_km, errors="coerce").to_numpy(dtype=float)
    dt = pd.to_numeric(dt_hours, errors="coerce").to_numpy(dtype=float)
    out = np.zeros(d.shape, dtype=float)
    valid = np.isfinite(d)
    if not np.any(valid):
        return out
    r0 = max(float(r0_km), 1e-6)
    td = max(float(td0_hours), 1e-6)
    spatial = np.exp(-np.square(d[valid] / r0))
    tconf = np.zeros(spatial.shape, dtype=float)
    dtv = dt[valid]
    dt_ok = np.isfinite(dtv)
    tconf[dt_ok] = np.exp(-np.square(dtv[dt_ok] / td))
    out[valid] = spatial * tconf
    return out


def _soft_far_weight_series(*, dist_km: pd.Series, dt_hours: pd.Series, r_far_km: float, td0_hours: float) -> np.ndarray:
    d = pd.to_numeric(dist_km, errors="coerce").to_numpy(dtype=float)
    dt = pd.to_numeric(dt_hours, errors="coerce").to_numpy(dtype=float)
    out = np.zeros(d.shape, dtype=float)
    valid = np.isfinite(d)
    if not np.any(valid):
        return out
    rf = max(float(r_far_km), 1e-6)
    td = max(float(td0_hours), 1e-6)
    spatial = 1.0 - np.exp(-np.square(d[valid] / rf))
    tconf = np.zeros(spatial.shape, dtype=float)
    dtv = dt[valid]
    dt_ok = np.isfinite(dtv)
    tconf[dt_ok] = np.exp(-np.square(dtv[dt_ok] / td))
    out[valid] = spatial * tconf
    return out


def _lead_hours(lead_bucket: Any) -> float:
    text = str(lead_bucket).strip().lower()
    if text in {"", "none", "nan", "null"}:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _valid_time_from_components(time_value: Any, lead_bucket: Any, *, use_lead_shift: bool = False) -> pd.Timestamp:
    t = pd.to_datetime(time_value, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if not bool(use_lead_shift):
        return pd.Timestamp(t)
    return pd.Timestamp(t) + pd.Timedelta(hours=_lead_hours(lead_bucket))


def _min_storm_distance_within_window(
    *,
    ib_index: Any,
    time: pd.Timestamp,
    lat: float,
    lon: float,
    window_hours: float,
    no_storm_implies_far: bool = True,
) -> float:
    if ib_index is None or pd.isna(time):
        return np.nan
    by_time = getattr(ib_index, "by_time", {}) or {}
    if not by_time:
        return np.nan
    t0 = pd.Timestamp(time).floor("h")
    offsets = int(np.ceil(max(float(window_hours), 0.0)))
    best = np.inf
    seen_points = False
    lon_q = float(normalize_lon_180(lon))
    for h in range(-offsets, offsets + 1):
        g = by_time.get(t0 + pd.Timedelta(hours=h))
        if g is None or g.empty:
            continue
        seen_points = True
        dist = haversine_km_vec(
            float(lat),
            lon_q,
            pd.to_numeric(g["lat"], errors="coerce").to_numpy(dtype=float),
            normalize_lon_180(pd.to_numeric(g["lon"], errors="coerce").to_numpy(dtype=float)),
        )
        if dist.size == 0:
            continue
        dmin = float(np.nanmin(dist))
        if np.isfinite(dmin):
            best = min(best, dmin)
    if np.isfinite(best):
        return float(best)
    if bool(no_storm_implies_far) and (not bool(seen_points)):
        return float(np.inf)
    return np.nan


def _ib_conflict_flag(
    *,
    dist_km: float,
    dt_hours: float,
    far_km: float,
    dt_max_hours: float,
    ib_event: bool,
    ib_far: bool,
) -> str | None:
    d_ok = np.isfinite(float(dist_km))
    t_ok = np.isfinite(float(dt_hours))
    if not d_ok or not t_ok:
        return "missing_match"
    d = float(dist_km)
    t = float(dt_hours)
    if bool(ib_event) and bool(ib_far):
        return "event_far_overlap"
    if d <= float(far_km) and t > float(dt_max_hours):
        return "spatial_near_time_far"
    if t <= float(dt_max_hours) and d > float(far_km):
        return "time_near_spatial_far"
    return None


def _ib_far_quality_tag(
    *,
    dist_km: float,
    min_dist_window_km: float,
    far_km: float,
    far_loose_km: float,
    ib_far: bool,
    ib_far_strict: bool,
    kinematic_clean: bool | None = None,
) -> str:
    if bool(ib_far_strict):
        if kinematic_clean is True:
            return "A_strict_clean"
        if kinematic_clean is False:
            return "B_strict_kinematic_borderline"
        return "B_strict_unknown_kinematics"
    d_ok = np.isfinite(float(dist_km))
    w_ok = np.isfinite(float(min_dist_window_km))
    if not d_ok and not w_ok:
        return "missing_distance_and_window"
    if not d_ok:
        return "missing_spatial_distance"
    if not w_ok:
        return "missing_temporal_window_distance"
    if float(dist_km) < float(far_km):
        return "fails_spatial_far"
    if float(min_dist_window_km) < float(far_loose_km):
        return "fails_temporal_far_window"
    if bool(ib_far):
        return "C_loose_fallback"
    return "C_not_far"


def _build_ibtracs_alignment_diagnostics(
    *,
    samples: pd.DataFrame,
    case_manifest_all: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if samples.empty:
        return pd.DataFrame(), {"n_cases": 0, "available": False}
    frame = samples.copy()
    for col in ("ib_event", "ib_far", "ib_event_strict", "ib_far_strict"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce").fillna(0).astype(int)
    if "ib_dist_km" not in frame.columns:
        frame["ib_dist_km"] = pd.to_numeric(frame.get("nearest_storm_distance_km"), errors="coerce")
    if "ib_dt_hours" not in frame.columns:
        frame["ib_dt_hours"] = pd.to_numeric(frame.get("nearest_storm_time_delta_h"), errors="coerce")
    agg: dict[str, tuple[str, str]] = {
        "lead_bucket": ("lead_bucket", "first"),
        "case_type": ("case_type", "first"),
        "storm_id": ("storm_id", "first"),
        "lat0": ("lat0", "median"),
        "lon0": ("lon0", "median"),
        "time_basis": ("time_basis", "first"),
        "ib_time_basis_used": ("ib_time_basis_used", "first"),
        "internal_storm": ("storm", "max"),
        "internal_near": ("near_storm", "max"),
        "internal_pregen": ("pregen", "max"),
        "ib_sid": ("ib_sid", "first"),
        "ib_name": ("ib_name", "first"),
        "ib_basin": ("ib_basin", "first"),
        "ib_dist_km": ("ib_dist_km", "median"),
        "ib_dt_hours": ("ib_dt_hours", "median"),
        "ib_event": ("ib_event", "max"),
        "ib_far": ("ib_far", "max"),
        "ib_event_strict": ("ib_event_strict", "max"),
        "ib_far_strict": ("ib_far_strict", "max"),
        "ib_event_source": ("ib_event_source", "max"),
        "ib_far_source": ("ib_far_source", "max"),
        "ib_event_valid": ("ib_event_valid", "max"),
        "ib_far_valid": ("ib_far_valid", "max"),
        "ib_kinematic_clean": ("ib_kinematic_clean", "first"),
        "ib_far_quality_tag": ("ib_far_quality_tag", "first"),
        "ib_conflict_flag": ("ib_conflict_flag", "first"),
        "nearest_storm_lat": ("nearest_storm_lat", "median"),
        "nearest_storm_lon": ("nearest_storm_lon", "median"),
        "nearest_storm_fix_time": ("nearest_storm_fix_time", "first"),
        "nearest_storm_distance_km_source": ("nearest_storm_distance_km_source", "median"),
        "nearest_storm_time_delta_h_source": ("nearest_storm_time_delta_h_source", "median"),
        "nearest_storm_dt_fix_h_source": ("nearest_storm_dt_fix_h_source", "median"),
        "nearest_storm_distance_km_valid": ("nearest_storm_distance_km_valid", "median"),
        "nearest_storm_time_delta_h_valid": ("nearest_storm_time_delta_h_valid", "median"),
        "nearest_storm_dt_fix_h_valid": ("nearest_storm_dt_fix_h_valid", "median"),
        "nearest_storm_id_source": ("nearest_storm_id_source", "first"),
        "nearest_storm_id_valid": ("nearest_storm_id_valid", "first"),
    }
    time_col = "t" if "t" in frame.columns else ("time" if "time" in frame.columns else None)
    if time_col:
        agg["t"] = (time_col, "median")
    if "t_source" in frame.columns:
        agg["t_source"] = ("t_source", "median")
    if "t_valid" in frame.columns:
        agg["t_valid"] = ("t_valid", "median")
    if "storm_match_time" in frame.columns:
        agg["storm_match_time"] = ("storm_match_time", "median")
    agg = {k: v for k, v in agg.items() if v[0] in frame.columns}
    grp = frame.groupby("case_id", as_index=False).agg(**agg)
    grp["internal_event"] = (pd.to_numeric(grp["internal_storm"], errors="coerce").fillna(0) > 0).astype(int)
    grp["internal_far"] = (
        (pd.to_numeric(grp["internal_storm"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(grp["internal_near"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(grp["internal_pregen"], errors="coerce").fillna(0) == 0)
    ).astype(int)
    grp["ib_event"] = (pd.to_numeric(grp["ib_event"], errors="coerce").fillna(0) > 0).astype(int)
    grp["ib_far"] = (pd.to_numeric(grp["ib_far"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_event_source" in grp.columns:
        grp["ib_event_source"] = (pd.to_numeric(grp["ib_event_source"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_far_source" in grp.columns:
        grp["ib_far_source"] = (pd.to_numeric(grp["ib_far_source"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_event_valid" in grp.columns:
        grp["ib_event_valid"] = (pd.to_numeric(grp["ib_event_valid"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_far_valid" in grp.columns:
        grp["ib_far_valid"] = (pd.to_numeric(grp["ib_far_valid"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_event_strict" in grp.columns:
        grp["ib_event_strict"] = (pd.to_numeric(grp["ib_event_strict"], errors="coerce").fillna(0) > 0).astype(int)
    if "ib_far_strict" in grp.columns:
        grp["ib_far_strict"] = (pd.to_numeric(grp["ib_far_strict"], errors="coerce").fillna(0) > 0).astype(int)

    case_meta = pd.DataFrame(case_manifest_all) if case_manifest_all else pd.DataFrame()
    if not case_meta.empty and "case_id" in case_meta.columns:
        case_meta = case_meta.copy()
        case_meta["case_id"] = case_meta["case_id"].astype(str)
        keep_cols = [c for c in ("case_id", "center_source", "event_label", "control_tier", "match_quality") if c in case_meta.columns]
        if keep_cols:
            grp = grp.merge(case_meta[keep_cols].drop_duplicates(subset=["case_id"]), on="case_id", how="left")

    worst_internal_event_ib_far = grp.loc[(grp["internal_event"] == 1) & (grp["ib_event"] == 0)].copy()
    worst_internal_event_ib_far = worst_internal_event_ib_far.sort_values(["ib_dist_km", "ib_dt_hours"], ascending=[False, False]).head(20)
    worst_internal_far_ib_event = grp.loc[(grp["internal_far"] == 1) & (grp["ib_event"] == 1)].copy()
    worst_internal_far_ib_event = worst_internal_far_ib_event.sort_values(["ib_dist_km", "ib_dt_hours"], ascending=[True, True]).head(20)

    summary = {
        "available": True,
        "n_cases": int(grp.shape[0]),
        "internal_event_vs_ib_event": pd.crosstab(grp["internal_event"], grp["ib_event"]).to_dict(),
        "internal_far_vs_ib_far": pd.crosstab(grp["internal_far"], grp["ib_far"]).to_dict(),
        "basis_flip_event_case_count": int(
            (
                (
                    pd.to_numeric(grp.get("ib_event_source", pd.Series(0, index=grp.index)), errors="coerce").fillna(0).astype(int)
                    != pd.to_numeric(grp.get("ib_event_valid", pd.Series(0, index=grp.index)), errors="coerce").fillna(0).astype(int)
                )
            ).sum()
        ),
        "basis_flip_far_case_count": int(
            (
                (
                    pd.to_numeric(grp.get("ib_far_source", pd.Series(0, index=grp.index)), errors="coerce").fillna(0).astype(int)
                    != pd.to_numeric(grp.get("ib_far_valid", pd.Series(0, index=grp.index)), errors="coerce").fillna(0).astype(int)
                )
            ).sum()
        ),
        "basis_flip_storm_id_case_count": int(
            (
                grp.get("nearest_storm_id_source", pd.Series(index=grp.index)).fillna("").astype(str)
                != grp.get("nearest_storm_id_valid", pd.Series(index=grp.index)).fillna("").astype(str)
            ).sum()
        ),
        "ib_conflict_flag_counts": grp.get("ib_conflict_flag", pd.Series(dtype=object)).fillna("none").value_counts(dropna=False).to_dict(),
        "ib_event_case_count": int((grp["ib_event"] == 1).sum()),
        "ib_far_case_count": int((grp["ib_far"] == 1).sum()),
        "ib_event_strict_case_count": int((grp.get("ib_event_strict", pd.Series(dtype=int)) == 1).sum()) if "ib_event_strict" in grp.columns else 0,
        "ib_far_strict_case_count": int((grp.get("ib_far_strict", pd.Series(dtype=int)) == 1).sum()) if "ib_far_strict" in grp.columns else 0,
        "internal_event_case_count": int((grp["internal_event"] == 1).sum()),
        "internal_far_case_count": int((grp["internal_far"] == 1).sum()),
        "worst_internal_event_ib_far": worst_internal_event_ib_far[
            [c for c in ("case_id", "lead_bucket", "t_valid", "lat0", "lon0", "ib_sid", "ib_dist_km", "ib_dt_hours", "center_source") if c in worst_internal_event_ib_far.columns]
        ].to_dict(orient="records"),
        "worst_internal_far_ib_event": worst_internal_far_ib_event[
            [c for c in ("case_id", "lead_bucket", "t_valid", "lat0", "lon0", "ib_sid", "ib_dist_km", "ib_dt_hours", "center_source") if c in worst_internal_far_ib_event.columns]
        ].to_dict(orient="records"),
    }
    return grp.reset_index(drop=True), summary


def _build_ibtracs_contract_cases(contract_cases: pd.DataFrame) -> pd.DataFrame:
    if contract_cases is None or contract_cases.empty:
        return pd.DataFrame()
    out = contract_cases.copy()
    out["case_id"] = out.get("case_id", pd.Series(index=out.index)).astype(str)
    out["t_source"] = pd.to_datetime(out.get("t_source"), errors="coerce")
    out["t_valid"] = pd.to_datetime(out.get("t_valid"), errors="coerce")
    out["storm_match_time"] = pd.to_datetime(out.get("storm_match_time"), errors="coerce")
    out["ib_time_basis_used"] = out.get("ib_time_basis_used", out.get("time_basis", pd.Series(index=out.index))).fillna("source").astype(str)
    for col in (
        "ib_event",
        "ib_far",
        "ib_event_strict",
        "ib_far_strict",
        "ib_event_source",
        "ib_far_source",
        "ib_event_valid",
        "ib_far_valid",
    ):
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(0) > 0).astype(int)
    if "nearest_storm_lat" in out.columns and "nearest_storm_lon" in out.columns:
        lat0 = pd.to_numeric(out.get("lat0"), errors="coerce").to_numpy(dtype=float)
        lon0 = pd.to_numeric(out.get("lon0"), errors="coerce").to_numpy(dtype=float)
        slat = pd.to_numeric(out.get("nearest_storm_lat"), errors="coerce").to_numpy(dtype=float)
        slon = pd.to_numeric(out.get("nearest_storm_lon"), errors="coerce").to_numpy(dtype=float)
        dist = np.full(out.shape[0], np.nan, dtype=float)
        mask = np.isfinite(lat0) & np.isfinite(lon0) & np.isfinite(slat) & np.isfinite(slon)
        if np.any(mask):
            r = 6_371.0
            p1 = np.deg2rad(lat0[mask])
            p2 = np.deg2rad(slat[mask])
            dp = p2 - p1
            dl = np.deg2rad(slon[mask] - lon0[mask])
            a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None)))
            dist[mask] = r * c
        out["center_to_storm_dist_km"] = dist
    else:
        out["center_to_storm_dist_km"] = np.nan
    return out


def _summary_numeric(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"n": 0}
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p05": float(np.quantile(a, 0.05)),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _build_time_basis_audit(contract_cases: pd.DataFrame) -> dict[str, Any]:
    if contract_cases is None or contract_cases.empty:
        return {"available": False, "n_cases": 0}
    frame = contract_cases.copy()
    event_src = pd.to_numeric(frame.get("ib_event_source"), errors="coerce").fillna(0).astype(int)
    event_val = pd.to_numeric(frame.get("ib_event_valid"), errors="coerce").fillna(0).astype(int)
    far_src = pd.to_numeric(frame.get("ib_far_source"), errors="coerce").fillna(0).astype(int)
    far_val = pd.to_numeric(frame.get("ib_far_valid"), errors="coerce").fillna(0).astype(int)
    sid_src = frame.get("nearest_storm_id_source", pd.Series(index=frame.index)).fillna("").astype(str)
    sid_val = frame.get("nearest_storm_id_valid", pd.Series(index=frame.index)).fillna("").astype(str)
    flip_event = (event_src != event_val)
    flip_far = (far_src != far_val)
    flip_sid = (sid_src != sid_val)
    return {
        "available": True,
        "n_cases": int(frame.shape[0]),
        "time_basis_used_counts": frame.get("ib_time_basis_used", pd.Series(index=frame.index)).fillna("unknown").astype(str).value_counts(dropna=False).to_dict(),
        "source_event_case_count": int((event_src == 1).sum()),
        "valid_event_case_count": int((event_val == 1).sum()),
        "source_far_case_count": int((far_src == 1).sum()),
        "valid_far_case_count": int((far_val == 1).sum()),
        "basis_flip_event_case_count": int(flip_event.sum()),
        "basis_flip_far_case_count": int(flip_far.sum()),
        "basis_flip_storm_id_case_count": int(flip_sid.sum()),
        "basis_flip_any_label_case_count": int((flip_event | flip_far).sum()),
        "basis_flip_label_rate": float((flip_event | flip_far).mean()) if frame.shape[0] > 0 else 0.0,
        "dt_to_fix_hours_source_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_dt_fix_h_source"), errors="coerce")),
        "dt_to_fix_hours_valid_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_dt_fix_h_valid"), errors="coerce")),
        "center_uncert_km_source_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_center_uncert_km_source"), errors="coerce")),
        "center_uncert_km_valid_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_center_uncert_km_valid"), errors="coerce")),
        "speed_kmh_source_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_speed_kmh_source"), errors="coerce")),
        "speed_kmh_valid_summary": _summary_numeric(pd.to_numeric(frame.get("nearest_storm_speed_kmh_valid"), errors="coerce")),
    }


def _build_center_definition_audit(contract_cases: pd.DataFrame) -> dict[str, Any]:
    if contract_cases is None or contract_cases.empty:
        return {"available": False, "n_cases": 0}
    frame = contract_cases.copy()
    dist = pd.to_numeric(frame.get("center_to_storm_dist_km"), errors="coerce")
    out: dict[str, Any] = {
        "available": True,
        "n_cases": int(frame.shape[0]),
        "center_to_storm_dist_km_summary": _summary_numeric(dist),
        "center_mismatch_gt200km_rate": float((dist > 200.0).mean()) if dist.notna().any() else 0.0,
    }
    if "center_source" in frame.columns:
        by_source: dict[str, Any] = {}
        for src, grp in frame.groupby(frame["center_source"].fillna("unknown").astype(str), dropna=False):
            d = pd.to_numeric(grp.get("center_to_storm_dist_km"), errors="coerce")
            by_source[str(src)] = {
                "n_cases": int(grp.shape[0]),
                "dist_km_summary": _summary_numeric(d),
                "mismatch_gt200km_rate": float((d > 200.0).mean()) if d.notna().any() else 0.0,
            }
        out["by_center_source"] = by_source
    return out


def _build_cohort_audit(contract_cases: pd.DataFrame) -> dict[str, Any]:
    if contract_cases is None or contract_cases.empty:
        return {"available": False, "n_cases": 0}
    frame = contract_cases.copy()
    lead = frame.get("lead_bucket", pd.Series(index=frame.index)).fillna("unknown").astype(str)
    event = (pd.to_numeric(frame.get("ib_event_strict"), errors="coerce").fillna(0) > 0)
    far = (pd.to_numeric(frame.get("ib_far_strict"), errors="coerce").fillna(0) > 0)
    by_lead: dict[str, Any] = {}
    for k in sorted(lead.dropna().unique().tolist()):
        m = lead == str(k)
        by_lead[str(k)] = {
            "n_cases": int(m.sum()),
            "ib_event_strict_cases": int((event & m).sum()),
            "ib_far_strict_cases": int((far & m).sum()),
        }
    return {
        "available": True,
        "n_cases": int(frame.shape[0]),
        "ib_event_strict_case_count": int(event.sum()),
        "ib_far_strict_case_count": int(far.sum()),
        "ib_far_quality_tag_counts": frame.get("ib_far_quality_tag", pd.Series(index=frame.index)).fillna("missing").astype(str).value_counts(dropna=False).to_dict(),
        "by_lead": by_lead,
    }


def _build_strict_coverage_by_week(contract_cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "split_date",
        "n_train_cases",
        "n_test_cases",
        "n_buffer_cases",
        "strict_event_test_cases",
        "strict_far_test_cases",
        "strict_event_test_by_lead",
        "strict_far_test_by_lead",
    ]
    if contract_cases is None or contract_cases.empty:
        return pd.DataFrame(columns=cols)
    frame = contract_cases.copy()
    t = pd.to_datetime(frame.get("storm_match_time"), errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(frame.get("t_source"), errors="coerce")
    frame["_t"] = t
    frame = frame.dropna(subset=["case_id", "_t"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=cols)
    case = (
        frame.sort_values("_t")
        .drop_duplicates(subset=["case_id"], keep="first")
        .copy()
    )
    case["lead_bucket"] = case.get("lead_bucket", pd.Series(index=case.index)).fillna("none").astype(str)
    case["ib_event_strict"] = (pd.to_numeric(case.get("ib_event_strict"), errors="coerce").fillna(0) > 0)
    case["ib_far_strict"] = (pd.to_numeric(case.get("ib_far_strict"), errors="coerce").fillna(0) > 0)

    start = pd.Timestamp(case["_t"].min()).floor("D")
    end = pd.Timestamp(case["_t"].max()).floor("D")
    if end < start:
        return pd.DataFrame(columns=cols)
    split_dates = pd.date_range(start=start, end=end, freq="7D")
    if split_dates.empty:
        split_dates = pd.DatetimeIndex([start])

    rows: list[dict[str, Any]] = []
    for split_dt in split_dates:
        train_mask = case["_t"] < split_dt
        test_mask = case["_t"] >= split_dt
        train = case.loc[train_mask].copy()
        test = case.loc[test_mask].copy()
        ev = test.loc[test["ib_event_strict"]].copy()
        fr = test.loc[test["ib_far_strict"]].copy()
        ev_by = ev.groupby("lead_bucket")["case_id"].nunique().to_dict()
        fr_by = fr.groupby("lead_bucket")["case_id"].nunique().to_dict()
        rows.append(
            {
                "split_date": str(pd.Timestamp(split_dt).date()),
                "n_train_cases": int(train["case_id"].nunique()),
                "n_test_cases": int(test["case_id"].nunique()),
                "n_buffer_cases": 0,
                "strict_event_test_cases": int(ev["case_id"].nunique()),
                "strict_far_test_cases": int(fr["case_id"].nunique()),
                "strict_event_test_by_lead": {str(k): int(v) for k, v in sorted(ev_by.items())},
                "strict_far_test_by_lead": {str(k): int(v) for k, v in sorted(fr_by.items())},
            }
        )
    return pd.DataFrame(rows)


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if key not in row or row.get(key) is None:
            continue
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())

