from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from gka.weather.ibtracs import load_ibtracs_points, match_tracks_to_ibtracs
from gka.weather.polar import summarize_polar_features
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
        "--event-source",
        choices=["vortex_or_labels", "vortex", "labels", "ibtracs"],
        default="vortex_or_labels",
        help="How event centers are discovered before tile extraction",
    )
    parser.add_argument(
        "--ibtracs-csv",
        default=None,
        help="Optional IBTrACS CSV path used when --event-source ibtracs or for discovery validation",
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
        default=0,
        help="Fail if used control cases per lead bucket falls below this threshold",
    )
    parser.add_argument(
        "--min-far-nonstorm-per-lead-bucket",
        type=int,
        default=0,
        help="Fail if far-nonstorm control cases per lead bucket falls below this threshold",
    )
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
    event_source_requested = str(args.event_source)
    event_source_effective = str(event_source_requested)

    events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    control_stats: dict[str, Any] = {}
    vortex_candidates_df = pd.DataFrame()
    storm_tracks_df = pd.DataFrame()
    track_summary_df = pd.DataFrame()
    ibtracs_match: dict[str, Any] | None = None
    all_cases: list[dict[str, Any]] = []
    if cohort in {"all", "events"}:
        if event_source_requested == "ibtracs":
            if not args.ibtracs_csv:
                raise ValueError("--event-source ibtracs requires --ibtracs-csv")
            events = _collect_ibtracs_centers(
                ibtracs_csv=str(args.ibtracs_csv),
                lead_buckets=lead_buckets,
                max_events_per_lead=int(args.max_events_per_lead),
                lon_min=float(args.lon_min),
                lon_max=float(args.lon_max),
                rng=rng,
            )
        elif event_source_requested in {"vortex", "vortex_or_labels"}:
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
            if (not events) and event_source_requested == "vortex_or_labels":
                event_source_effective = "labels_fallback"
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
        if args.ibtracs_csv and (not storm_tracks_df.empty):
            try:
                ib_points = load_ibtracs_points(
                    args.ibtracs_csv,
                    lon_min=float(args.lon_min),
                    lon_max=float(args.lon_max),
                )
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
        all_cases = events + controls
    elif cohort == "events":
        all_cases = events
    elif cohort == "background":
        controls = _collect_background_centers(
            dataset=dataset,
            lead_buckets=lead_buckets,
            max_per_lead=int(args.background_per_lead or args.max_events_per_lead),
            rng=rng,
            scan_batch_rows=int(args.scan_batch_rows),
            max_batches_per_lead=int(args.background_max_batches_per_lead),
        )
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

    control_coverage = _compute_control_coverage(samples=samples)
    min_controls = int(args.min_controls_per_lead_bucket)
    min_far = int(args.min_far_nonstorm_per_lead_bucket)
    if min_controls > 0:
        missing = {
            lead: cnt
            for lead, cnt in control_coverage["control_cases_by_lead"].items()
            if int(cnt) < min_controls
        }
        if missing:
            raise ValueError(
                "insufficient_control_coverage:"
                + ",".join(f"{lead}:{cnt}<{min_controls}" for lead, cnt in sorted(missing.items()))
            )
    if min_far > 0:
        missing = {
            lead: cnt
            for lead, cnt in control_coverage["far_nonstorm_controls_by_lead"].items()
            if int(cnt) < min_far
        }
        if missing:
            raise ValueError(
                "insufficient_far_nonstorm_coverage:"
                + ",".join(f"{lead}:{cnt}<{min_far}" for lead, cnt in sorted(missing.items()))
            )

    samples.to_parquet(out_dir / "samples.parquet", index=False)
    eta_long = _build_eta_long(samples)
    eta_long.to_parquet(out_dir / "eta_long.parquet", index=False)
    polar_features = pd.DataFrame(polar_rows)
    if not polar_features.empty:
        polar_features["t"] = pd.to_datetime(polar_features["t"], errors="coerce")
        polar_features = polar_features.sort_values(["case_id", "t", "L"]).reset_index(drop=True)
    polar_features.to_parquet(out_dir / "polar_features.parquet", index=False)
    if not vortex_candidates_df.empty:
        vortex_candidates_df.to_parquet(out_dir / "vortex_candidates.parquet", index=False)
    if not storm_tracks_df.empty:
        storm_tracks_df.to_parquet(out_dir / "storm_tracks.parquet", index=False)
    if not track_summary_df.empty:
        track_summary_df.to_parquet(out_dir / "storm_track_summary.parquet", index=False)
    if ibtracs_match is not None:
        (out_dir / "ibtracs_match.json").write_text(json.dumps(ibtracs_match, indent=2, sort_keys=True), encoding="utf-8")

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
            "match_lat_bin_deg": float(args.match_lat_bin_deg),
            "match_lon_bin_deg": float(args.match_lon_bin_deg),
            "allow_nonexact_controls": bool(args.allow_nonexact_controls),
            "require_physical_match": bool(args.require_physical_match),
            "physical_speed_bin_ms": float(args.physical_speed_bin_ms),
            "physical_zeta_bin": float(args.physical_zeta_bin),
            "min_distinct_scales_per_case": int(args.min_distinct_scales_per_case),
            "min_controls_per_lead_bucket": int(args.min_controls_per_lead_bucket),
            "min_far_nonstorm_per_lead_bucket": int(args.min_far_nonstorm_per_lead_bucket),
            "event_source_requested": str(event_source_requested),
            "event_source_effective": str(event_source_effective),
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
        "physical_speed_bin_ms": float(args.physical_speed_bin_ms),
        "physical_zeta_bin": float(args.physical_zeta_bin),
        "min_distinct_scales_per_case": int(args.min_distinct_scales_per_case),
        "min_controls_per_lead_bucket": int(args.min_controls_per_lead_bucket),
        "min_far_nonstorm_per_lead_bucket": int(args.min_far_nonstorm_per_lead_bucket),
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
        "control_stats": control_stats,
        "control_coverage": control_coverage,
        "n_vortex_candidates": int(vortex_candidates_df.shape[0]),
        "n_storm_track_points": int(storm_tracks_df.shape[0]),
        "n_tracks": int(storm_tracks_df["track_id"].nunique()) if not storm_tracks_df.empty and "track_id" in storm_tracks_df.columns else 0,
        "anomaly_commutation": _summarize_anomaly_audit(anomaly_audit_rows),
        "scale_audit_path": str((out_dir / "scale_audit.json").resolve()),
        "eta_long_path": str((out_dir / "eta_long.parquet").resolve()),
        "polar_features_path": str((out_dir / "polar_features.parquet").resolve()),
        "vortex_candidates_path": str((out_dir / "vortex_candidates.parquet").resolve()) if not vortex_candidates_df.empty else None,
        "storm_tracks_path": str((out_dir / "storm_tracks.parquet").resolve()) if not storm_tracks_df.empty else None,
        "storm_track_summary_path": str((out_dir / "storm_track_summary.parquet").resolve()) if not track_summary_df.empty else None,
        "ibtracs_match": ibtracs_match,
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
    ibtracs_csv: str,
    lead_buckets: list[str],
    max_events_per_lead: int,
    lon_min: float,
    lon_max: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    ib = load_ibtracs_points(
        ibtracs_csv,
        lon_min=float(lon_min),
        lon_max=float(lon_max),
    )
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
        for event in lead_events:
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
                    "case_id": str(event["case_id"]).replace("storm_", "bgm_"),
                    "case_type": "control",
                    "event_label": "background_matched",
                    "storm_id": f"background_matched_{lead}_{len(controls):04d}",
                    "lead_bucket": str(lead),
                    "time0": pd.Timestamp(pick["time"]),
                    "lat0": float(pick["lat"]),
                    "lon0": float(pick["lon"]),
                    "dist_from_event_deg": dist_deg,
                    "dist_from_event_bin_deg": dist_bin,
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
    filt = (
        _lead_filter(lead)
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
) -> list[dict[str, Any]]:
    cols = ["time", "lat", "lon", "lead_partition"]
    out: list[dict[str, Any]] = []
    for lead in lead_buckets:
        filt = (
            _lead_filter(lead)
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
                # Reservoir-style replacement across batches by random keep.
                if t_val not in picks_by_time or bool(rng.random() < 0.5):
                    picks_by_time[pd.Timestamp(t_val)] = {
                        "time": pd.Timestamp(row["time"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
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
                    "storm_id": f"background_{lead}_{i:04d}",
                    "lead_bucket": str(lead),
                    "time0": pd.Timestamp(row["time"]),
                    "lat0": float(row["lat"]),
                    "lon0": float(row["lon"]),
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
            if bool(polar_enable):
                slice_df = subset_by_time.get(t_val)
                if slice_df is not None and not slice_df.empty:
                    try:
                        polar_orig = summarize_polar_features(
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
                        polar_mirror = summarize_polar_features(
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
    pivot = samples.pivot_table(
        index=[
            "case_id",
            "case_type",
            "control_tier",
            "lead_bucket",
            "t",
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
    return pivot.dropna(subset=["eta"]).reset_index(drop=True)


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
