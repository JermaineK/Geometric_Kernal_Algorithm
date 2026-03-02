from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch
from scipy.stats import kendalltau, ks_2samp

from gka.core.pipeline import run_pipeline
from gka.domains import register_builtin_adapters
from gka.weather.contract import (
    strict_coverage_ok as contract_strict_coverage_ok,
    strict_ib_case_flags as contract_strict_ib_case_flags,
    strict_ib_coverage as contract_strict_ib_coverage,
)
from gka.utils.time import utc_now_iso

NullTransform = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]

MODE_TO_COLUMN: dict[str, str | None] = {
    "current": None,
    "vectors_only": "O_vector",
    "scalars_only": "O_scalar",
    "raw": "O_raw",
    "local_frame": "O_local_frame",
    "vorticity": "O_vorticity",
    "meanflow_removed": "O_meanflow",
    "anomaly_lat_hour": "O_lat_hour",
    "anomaly_lat_day": "O_lat_day",
    "polar_spiral": "O_polar_spiral",
    "polar_chiral": "O_polar_chiral",
    "polar_left": "O_polar_left",
    "polar_right": "O_polar_right",
    "polar_odd_ratio": "O_polar_odd_ratio",
    "polar_eta": "O_polar_eta",
}


def _mode_to_column(mode: str) -> str | None:
    key = str(mode or "").strip()
    if key in MODE_TO_COLUMN:
        return MODE_TO_COLUMN[key]
    alias = {
        "none": "O_raw",
        "lat_day": "O_lat_day",
        "lat_hour": "O_lat_hour",
    }
    return alias.get(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate weather real-data minipilot with blocked splits and null controls")
    parser.add_argument("--dataset", required=True, help="Canonical tile dataset directory")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--split-date", default="2025-05-01", help="Blocked split date (train < split, test >= split)")
    parser.add_argument(
        "--time-buffer-hours",
        type=float,
        default=72.0,
        help="Buffer around split boundary; cases in buffer are excluded from both train/test",
    )
    parser.add_argument(
        "--storm-id-col",
        default="auto",
        help="Optional storm identity column for leakage audit (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--p-lock-threshold", type=float, default=0.5, help="Parity-lock threshold")
    parser.add_argument(
        "--enable-time-frequency-knee",
        action="store_true",
        help="Enable optional time-frequency knee detector on per-case eta time series",
    )
    parser.add_argument(
        "--tf-knee-bic-delta-min",
        type=float,
        default=6.0,
        help="Minimum delta-BIC for accepting time-frequency knee candidates",
    )
    parser.add_argument(
        "--slowtick-delta-min",
        type=float,
        default=0.05,
        help="Minimum effect-size delta (observed - null median) for slowtick survival",
    )
    parser.add_argument(
        "--slowtick-p-max",
        type=float,
        default=0.10,
        help="Maximum empirical p-value for slowtick survival",
    )
    parser.add_argument(
        "--axis-sensitivity-json",
        default=None,
        help="Optional path to lon0_sensitivity.json for axis-robust parity gating",
    )
    parser.add_argument(
        "--axis-std-threshold",
        type=float,
        default=0.02,
        help="Max std of event-background parity contrast across lon0 sweep to be considered axis-robust",
    )
    parser.add_argument(
        "--min-far-nonstorm-test-cases",
        type=int,
        default=10,
        help="Fail fast if test split has fewer far-nonstorm cases than this threshold",
    )
    parser.add_argument(
        "--min-far-nonstorm-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum far-nonstorm test case count (0 disables per-lead gate)",
    )
    parser.add_argument(
        "--min-train-cases",
        type=int,
        default=15,
        help="Minimum train cases required after blocked split + buffer",
    )
    parser.add_argument(
        "--min-test-cases",
        type=int,
        default=15,
        help="Minimum test cases required after blocked split + buffer",
    )
    parser.add_argument(
        "--min-event-test-cases",
        type=int,
        default=10,
        help="Minimum event/near/pregen cases required in test split",
    )
    parser.add_argument(
        "--min-event-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum event/near/pregen test case count (0 disables per-lead gate)",
    )
    parser.add_argument(
        "--parity-claim-mode",
        choices=[
            "auto",
            "current",
            "vectors_only",
            "scalars_only",
            "raw",
            "local_frame",
            "vorticity",
            "meanflow_removed",
            "anomaly_lat_hour",
            "anomaly_lat_day",
            "polar_spiral",
            "polar_chiral",
            "polar_left",
            "polar_right",
            "polar_odd_ratio",
            "polar_eta",
        ],
        default="auto",
        help="Parity observable mode used for final claim metrics (auto uses ablations + gates)",
    )
    parser.add_argument(
        "--parity-confound-max-ratio",
        type=float,
        default=0.6,
        help="Maximum allowed confound ratio far_nonstorm/events for interpretation",
    )
    parser.add_argument(
        "--parity-event-minus-far-min",
        type=float,
        default=0.15,
        help="Minimum required margin between event and far-nonstorm parity rates",
    )
    parser.add_argument(
        "--parity-far-max",
        type=float,
        default=0.35,
        help="Maximum allowed far-nonstorm parity signal rate",
    )
    parser.add_argument(
        "--parity-use-alignment-fraction",
        dest="parity_use_alignment_fraction",
        action="store_true",
        help="Use alignment-fraction metrics (A_align_event/A_align_far) for parity confound gating when available.",
    )
    parser.add_argument(
        "--parity-use-raw-rate",
        dest="parity_use_alignment_fraction",
        action="store_false",
        help="Disable alignment-fraction gating and use raw parity signal rates.",
    )
    parser.set_defaults(parity_use_alignment_fraction=True)
    parser.add_argument(
        "--alignment-eta-threshold",
        type=float,
        default=0.10,
        help="Per-time eta threshold used to compute alignment fraction metrics.",
    )
    parser.add_argument(
        "--alignment-block-hours",
        type=float,
        default=3.0,
        help="Time block size used for alignment-fraction summaries.",
    )
    parser.add_argument(
        "--claim-min-far-per-lead",
        type=int,
        default=3,
        help="Minimum far_nonstorm test cases per lead to include that lead in claim-mode metrics (0 disables lead filtering)",
    )
    parser.add_argument(
        "--anomaly-agreement-mean-min",
        type=float,
        default=0.85,
        help="Minimum mean decision agreement across anomaly modes vs none",
    )
    parser.add_argument(
        "--anomaly-agreement-min-min",
        type=float,
        default=0.80,
        help="Minimum worst-case decision agreement across anomaly modes vs none",
    )
    parser.add_argument(
        "--anomaly-case-stability-min",
        type=float,
        default=0.67,
        help="Minimum per-case cross-mode agreement required for stable canonical anomaly mode selection",
    )
    parser.add_argument(
        "--anomaly-mode-agreement-min",
        type=float,
        default=0.70,
        help="Minimum mode-level agreement vs none required when ranking canonical anomaly modes",
    )
    parser.add_argument(
        "--anomaly-selection-far-tolerance",
        type=float,
        default=0.10,
        help="Canonical anomaly mode far-rate may exceed best mode by at most this tolerance",
    )
    parser.add_argument(
        "--null-collapse-min-drop",
        type=float,
        default=0.50,
        help="Minimum relative event-parity drop required under geometry nulls",
    )
    parser.add_argument(
        "--null-collapse-min-abs-drop",
        type=float,
        default=0.15,
        help="Minimum absolute event-parity drop required under geometry nulls",
    )
    parser.add_argument(
        "--mode-invariant-event-minus-far-min",
        type=float,
        default=0.15,
        help="Minimum event-minus-far margin required under all-modes parity consensus for interpretability",
    )
    parser.add_argument(
        "--mode-invariant-far-max",
        type=float,
        default=0.35,
        help="Maximum far_nonstorm parity rate allowed under all-modes parity consensus",
    )
    parser.add_argument(
        "--angular-witness-margin-min",
        type=float,
        default=0.10,
        help="Minimum event-minus-far margin for angular witness rate",
    )
    parser.add_argument(
        "--angular-witness-null-drop-min",
        type=float,
        default=0.30,
        help="Minimum relative drop of angular witness under theta_roll null",
    )
    parser.add_argument(
        "--angular-witness-null-abs-drop-min",
        type=float,
        default=0.05,
        help="Minimum absolute drop of angular witness under theta_roll null",
    )
    parser.add_argument(
        "--angular-witness-d-min",
        type=float,
        default=0.40,
        help="Minimum Cohen's d effect size for angular witness event-vs-far separation",
    )
    parser.add_argument(
        "--angular-witness-p-max",
        type=float,
        default=0.10,
        help="Maximum one-sided permutation p-value for angular witness separation",
    )
    parser.add_argument(
        "--angular-witness-null-margin-max",
        type=float,
        default=0.05,
        help="Maximum allowed null margin for angular witness (theta_roll/center_jitter)",
    )
    parser.add_argument(
        "--angular-witness-permutation-n",
        type=int,
        default=500,
        help="Permutation count for angular witness p-value estimation",
    )
    parser.add_argument(
        "--ibtracs-strict-radius-km",
        type=float,
        default=500.0,
        help="Radius threshold (km) for strict IBTrACS-aligned event slice in evaluation overlay",
    )
    parser.add_argument(
        "--ibtracs-strict-far-min-km",
        type=float,
        default=1500.0,
        help="Minimum distance (km) for strict far-control slice in IBTrACS overlay",
    )
    parser.add_argument(
        "--ibtracs-strict-time-hours",
        type=float,
        default=3.0,
        help="Maximum |time delta| hours for strict IBTrACS event slice when time-delta column is available",
    )
    parser.add_argument(
        "--ibtracs-strict-use-flags",
        action="store_true",
        help="Prefer ib_event_strict/ib_far_strict case flags (when present) for IBTrACS strict overlay masks",
    )
    parser.add_argument(
        "--ibtracs-selected-time-basis",
        choices=["auto", "source", "valid"],
        default="auto",
        help="Time basis used for claim-time IBTrACS alignment gate selection (auto picks dominant basis in samples)",
    )
    parser.add_argument(
        "--ibtracs-alignment-min-event-cases",
        type=int,
        default=5,
        help="Minimum strict IBTrACS event cases required for alignment gate",
    )
    parser.add_argument(
        "--ibtracs-alignment-min-far-cases",
        type=int,
        default=5,
        help="Minimum strict IBTrACS far-control cases required for alignment gate",
    )
    parser.add_argument(
        "--ibtracs-alignment-margin-min",
        type=float,
        default=0.0,
        help="Minimum strict IBTrACS event-minus-far parity margin required for alignment gate",
    )
    parser.add_argument(
        "--ibtracs-alignment-confound-max",
        type=float,
        default=1.0,
        help="Maximum strict IBTrACS confound ratio allowed for alignment gate",
    )
    parser.add_argument(
        "--claim-contract-schema-version",
        type=int,
        default=1,
        help="Schema version written to claim_contract.json",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional hard cap on number of case_ids used in this evaluation (0 disables cap)",
    )
    parser.add_argument(
        "--case-shards",
        type=int,
        default=1,
        help="Number of deterministic case shards for resumable evaluation",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=1,
        help="1-based shard index when --case-shards > 1",
    )
    parser.add_argument(
        "--shard-stratify-by",
        nargs="+",
        default=["case_type", "lead_bucket", "time_basis", "storm_id"],
        help="Case-level fields used to stratify shard assignment; supports comma-separated values",
    )
    parser.add_argument(
        "--claim-far-quality-tags",
        nargs="+",
        default=["A_strict_clean"],
        help="Allowed ib_far_quality_tag values for far controls in claim-mode filtering",
    )
    parser.add_argument(
        "--parity-odd-inner-outer-min",
        type=float,
        default=1.10,
        help="Minimum odd-energy inner/outer ring ratio required for parity pass",
    )
    parser.add_argument(
        "--parity-tangential-radial-min",
        type=float,
        default=1.10,
        help="Minimum tangential/radial odd-energy ratio required for parity pass",
    )
    parser.add_argument(
        "--parity-inner-ring-quantile",
        type=float,
        default=0.5,
        help="Quantile split on L used to define inner vs outer ring for parity localization",
    )
    parser.add_argument(
        "--geometry-required-checks",
        nargs="+",
        default=[
            "theta_roll",
            "center_swap",
            "storm_track_reassignment",
            "time_permutation_within_lead",
            "theta_scramble_within_ring",
        ],
        help="Geometry null checks required by the collapse gate",
    )
    parser.add_argument(
        "--null-transforms",
        nargs="+",
        default=[],
        help=(
            "Optional subset of null transform names to execute. "
            "Default runs full built-in null suite."
        ),
    )
    parser.add_argument(
        "--null-no-signal-max-rate",
        type=float,
        default=0.05,
        help="Maximum null event parity rate allowed when observed event parity is effectively zero",
    )
    parser.add_argument(
        "--strict-coverage-time-basis",
        choices=["selected", "source", "valid"],
        default="selected",
        help="Time basis used for strict IBTrACS event/far coverage checks",
    )
    parser.add_argument(
        "--canonical-time-basis",
        choices=["auto", "selected", "source", "valid"],
        default="auto",
        help=(
            "Canonical basis for strict cohorts + calibration + IB alignment. "
            "When set (not auto), enforces one basis across contract-critical paths."
        ),
    )
    parser.add_argument(
        "--strict-coverage-use-flags",
        dest="strict_coverage_use_flags",
        action="store_true",
        help="Use ib_event_strict/ib_far_strict flags for strict coverage checks when available",
    )
    parser.add_argument(
        "--strict-coverage-no-flags",
        dest="strict_coverage_use_flags",
        action="store_false",
        help="Disable strict-flag usage and rely on distance/time columns for strict coverage checks",
    )
    parser.set_defaults(strict_coverage_use_flags=True)
    parser.add_argument(
        "--min-strict-event-test-cases",
        type=int,
        default=0,
        help="Minimum strict-IB event test cases required for claim-valid shard status (0 uses --min-event-test-cases)",
    )
    parser.add_argument(
        "--min-strict-far-test-cases",
        type=int,
        default=0,
        help="Minimum strict-IB far test cases required for claim-valid shard status (0 uses --min-far-nonstorm-test-cases)",
    )
    parser.add_argument(
        "--min-ib-event-train",
        "--min-strict-event-train-cases",
        dest="min_strict_event_train_cases",
        type=int,
        default=0,
        help="Minimum strict-IB event train cases required for calibration/claim path (0 uses strict test minimum)",
    )
    parser.add_argument(
        "--min-ib-far-train",
        "--min-strict-far-train-cases",
        dest="min_strict_far_train_cases",
        type=int,
        default=0,
        help="Minimum strict-IB far train cases required for calibration/claim path (0 uses strict test minimum)",
    )
    parser.add_argument(
        "--min-strict-event-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum strict-IB event test cases for claim-valid shard status",
    )
    parser.add_argument(
        "--min-strict-far-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum strict-IB far test cases for claim-valid shard status",
    )
    parser.add_argument(
        "--min-strict-event-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum strict-IB event train cases for claim-valid shard status",
    )
    parser.add_argument(
        "--min-strict-far-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead minimum strict-IB far train cases for claim-valid shard status",
    )
    parser.add_argument(
        "--calibrate-parity-thresholds",
        action="store_true",
        help="Calibrate storm-local parity thresholds on train split and freeze for test evaluation",
    )
    parser.add_argument(
        "--parity-calibration-event-min-floor",
        type=float,
        default=0.05,
        help="Minimum train event parity rate allowed for calibrated thresholds; lower-rate solutions are rejected.",
    )
    parser.add_argument(
        "--parity-calibration-constrain-min-thresholds",
        action="store_true",
        help="Disallow calibrated odd/tangential thresholds below CLI defaults to avoid no-op fits.",
    )
    parser.add_argument(
        "--strict-far-min-row-quality-frac",
        type=float,
        default=0.0,
        help="Minimum per-case fraction of rows carrying allowed far-quality tags for strict far inclusion.",
    )
    parser.add_argument(
        "--strict-far-min-kinematic-clean-frac",
        type=float,
        default=0.0,
        help="Minimum per-case fraction of kinematically clean rows for strict far inclusion.",
    )
    parser.add_argument(
        "--strict-far-min-any-storm-km",
        type=float,
        default=0.0,
        help="Minimum per-case min-distance-to-any-storm (basis-aware) required for strict far inclusion.",
    )
    parser.add_argument(
        "--strict-far-min-nearest-storm-km",
        type=float,
        default=0.0,
        help="Minimum per-case nearest-storm distance (basis-aware) required for strict far inclusion.",
    )
    parser.add_argument(
        "--knee-weather-delta-bic-min",
        type=float,
        default=6.0,
        help="Minimum delta-BIC for relaxed weather knee acceptance",
    )
    parser.add_argument(
        "--knee-weather-resid-improvement-min",
        type=float,
        default=0.75,
        help="Minimum residual improvement for relaxed weather knee acceptance",
    )
    parser.add_argument(
        "--knee-weather-slope-delta-min",
        type=float,
        default=0.20,
        help="Minimum |slope_right - slope_left| for relaxed weather knee acceptance",
    )
    parser.add_argument(
        "--knee-weather-min-consistent-modes",
        type=int,
        default=2,
        help="Minimum anomaly/observable modes agreeing on knee location for weather acceptance",
    )
    parser.add_argument(
        "--knee-weather-knee-l-rel-tol",
        type=float,
        default=0.35,
        help="Relative tolerance on knee L consistency across modes for weather acceptance",
    )
    return parser.parse_args()


def _select_case_subset(
    samples: pd.DataFrame,
    *,
    max_cases: int,
    case_shards: int,
    case_shard_index: int,
    stratify_by: tuple[str, ...] = (),
    storm_id_col: str | None = None,
    split_date: pd.Timestamp | None = None,
    time_buffer_hours: float = 0.0,
    strict_time_basis: str = "selected",
    strict_use_flags: bool = True,
    far_quality_tags: tuple[str, ...] = (),
    min_event_total: int = 0,
    min_far_total: int = 0,
    min_event_train_total: int = 0,
    min_far_train_total: int = 0,
    min_train: int = 0,
    min_test: int = 0,
    strict_far_min_row_quality_frac: float = 0.0,
    strict_far_min_kinematic_clean_frac: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if samples.empty or "case_id" not in samples.columns:
        return samples.copy(), {
            "enabled": False,
            "reason": "empty_or_missing_case_id",
            "n_cases_selected": 0,
        }
    all_cases = sorted(samples["case_id"].astype(str).dropna().unique().tolist())
    if not all_cases:
        return samples.copy(), {
            "enabled": False,
            "reason": "no_cases",
            "n_cases_selected": 0,
        }
    n_shards = int(max(1, case_shards))
    shard_idx = int(max(1, case_shard_index))
    shard_idx = int(min(shard_idx, n_shards))

    selected_cases: list[str]
    if n_shards <= 1:
        selected_cases = list(all_cases)
    else:
        selected_cases = []
        fields = [str(v).strip() for v in (stratify_by or ()) if str(v).strip()]
        if not fields:
            fields = []
        case_frame = (
            samples[["case_id"]]
            .drop_duplicates(subset=["case_id"], keep="first")
            .copy()
        )
        case_frame["case_id"] = case_frame["case_id"].astype(str)

        if "case_type" in fields:
            src = samples.groupby("case_id", dropna=False)["case_type"].first().astype(str)
            case_frame["case_type"] = case_frame["case_id"].map(src).fillna("unknown")
        if "lead" in fields or "lead_bucket" in fields:
            src = samples.groupby("case_id", dropna=False)["lead_bucket"].first().astype(str)
            case_frame["lead_bucket"] = case_frame["case_id"].map(src).fillna("none")
        if "time_basis" in fields:
            basis_col = None
            if "ib_time_basis_used" in samples.columns:
                basis_col = "ib_time_basis_used"
            elif "time_basis" in samples.columns:
                basis_col = "time_basis"
            if basis_col is not None:
                src = (
                    samples[["case_id", basis_col]]
                    .copy()
                    .dropna(subset=[basis_col])
                    .groupby("case_id", dropna=False)[basis_col]
                    .first()
                    .astype(str)
                )
                case_frame["time_basis"] = case_frame["case_id"].map(src).fillna("unknown")
            else:
                case_frame["time_basis"] = "unknown"
        if "storm_id" in fields:
            sid_col = str(storm_id_col) if storm_id_col else ""
            if sid_col and sid_col in samples.columns:
                src = samples.groupby("case_id", dropna=False)[sid_col].first().astype(str)
                case_frame["storm_id"] = case_frame["case_id"].map(src).fillna("")
            else:
                case_frame["storm_id"] = ""

        if fields:
            key_cols: list[str] = []
            for f in fields:
                if f == "lead":
                    f = "lead_bucket"
                if f in case_frame.columns:
                    key_cols.append(str(f))
            if key_cols:
                case_frame["_group"] = case_frame[key_cols].astype(str).agg("|".join, axis=1)
            else:
                case_frame["_group"] = "all"
        else:
            case_frame["_group"] = "all"

        assign: dict[str, int] = {}
        for _, grp in case_frame.groupby("_group", dropna=False):
            cids = grp["case_id"].astype(str).tolist()
            cids = sorted(cids, key=lambda cid: int(hashlib.sha1(cid.encode("utf-8")).hexdigest()[:12], 16))
            for pos, cid in enumerate(cids):
                assign[str(cid)] = int((pos % n_shards) + 1)
        for cid in all_cases:
            if int(assign.get(str(cid), 1)) == int(shard_idx):
                selected_cases.append(str(cid))
    selected_cases = sorted(selected_cases)
    if int(max_cases) > 0:
        max_n = int(max_cases)
        candidate_df = samples.loc[samples["case_id"].astype(str).isin(set(selected_cases))].copy()
        if (
            split_date is not None
            and (not candidate_df.empty)
            and ("case_id" in candidate_df.columns)
        ):
            try:
                strict_case = _strict_ib_case_flags(
                    samples=candidate_df,
                    time_basis=str(strict_time_basis),
                    use_flags=bool(strict_use_flags),
                    far_quality_tags=tuple(far_quality_tags),
                    min_far_quality_fraction=float(strict_far_min_row_quality_frac),
                    min_far_kinematic_clean_fraction=float(strict_far_min_kinematic_clean_frac),
                    strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
                    strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
                )
                strict_case = strict_case.set_index("case_id", drop=False)
                cand_meta = _build_case_meta(candidate_df, storm_id_col=storm_id_col)
                split = _blocked_time_split(
                    case_meta=cand_meta,
                    split_date=pd.Timestamp(split_date),
                    buffer_hours=float(time_buffer_hours),
                )
                train_ids = sorted([str(v) for v in split.get("train_cases", set())])
                test_ids = sorted([str(v) for v in split.get("test_cases", set())])
                remaining = sorted(selected_cases, key=lambda cid: int(hashlib.sha1(cid.encode("utf-8")).hexdigest()[:12], 16))

                test_event = [
                    cid
                    for cid in test_ids
                    if (cid in strict_case.index) and bool(strict_case.loc[cid, "event_flag"])
                ]
                test_far = [
                    cid
                    for cid in test_ids
                    if (cid in strict_case.index) and bool(strict_case.loc[cid, "far_flag"])
                ]
                train_event = [
                    cid
                    for cid in train_ids
                    if (cid in strict_case.index) and bool(strict_case.loc[cid, "event_flag"])
                ]
                train_far = [
                    cid
                    for cid in train_ids
                    if (cid in strict_case.index) and bool(strict_case.loc[cid, "far_flag"])
                ]
                selected_priority: list[str] = []
                selected_set: set[str] = set()

                need_event = int(max(0, min_event_total))
                need_far = int(max(0, min_far_total))
                need_event_train = int(max(0, min_event_train_total))
                need_far_train = int(max(0, min_far_train_total))
                for cid in test_event[:need_event]:
                    if cid not in selected_set and len(selected_priority) < max_n:
                        selected_priority.append(cid)
                        selected_set.add(cid)
                for cid in test_far[:need_far]:
                    if cid not in selected_set and len(selected_priority) < max_n:
                        selected_priority.append(cid)
                        selected_set.add(cid)
                for cid in train_event[:need_event_train]:
                    if cid not in selected_set and len(selected_priority) < max_n:
                        selected_priority.append(cid)
                        selected_set.add(cid)
                for cid in train_far[:need_far_train]:
                    if cid not in selected_set and len(selected_priority) < max_n:
                        selected_priority.append(cid)
                        selected_set.add(cid)

                min_test_need = int(max(0, min_test))
                min_train_need = int(max(0, min_train))
                for cid in test_ids:
                    if len(selected_priority) >= max_n:
                        break
                    if cid in selected_set:
                        continue
                    current_test = sum(1 for v in selected_priority if v in set(test_ids))
                    if current_test < min_test_need:
                        selected_priority.append(cid)
                        selected_set.add(cid)
                for cid in train_ids:
                    if len(selected_priority) >= max_n:
                        break
                    if cid in selected_set:
                        continue
                    current_train = sum(1 for v in selected_priority if v in set(train_ids))
                    if current_train < min_train_need:
                        selected_priority.append(cid)
                        selected_set.add(cid)

                for cid in remaining:
                    if len(selected_priority) >= max_n:
                        break
                    if cid in selected_set:
                        continue
                    selected_priority.append(cid)
                    selected_set.add(cid)

                selected_cases = sorted(selected_priority)
            except Exception:
                selected_cases = selected_cases[:max_n]
        else:
            selected_cases = selected_cases[:max_n]

    out = samples.loc[samples["case_id"].astype(str).isin(set(selected_cases))].copy()
    info = {
        "enabled": bool((n_shards > 1) or (int(max_cases) > 0)),
        "case_shards": int(n_shards),
        "case_shard_index": int(shard_idx),
        "shard_stratify_by": [str(v) for v in (stratify_by or ())],
        "max_cases": int(max_cases),
        "n_cases_total": int(len(all_cases)),
        "n_cases_selected": int(len(selected_cases)),
        "n_rows_selected": int(out.shape[0]),
        "subset_contract_preserving": bool(int(max_cases) > 0 and split_date is not None),
        "strict_time_basis": str(strict_time_basis),
        "strict_use_flags": bool(strict_use_flags),
        "claim_far_quality_tags": [str(v) for v in (far_quality_tags or ())],
        "min_event_total": int(min_event_total),
        "min_far_total": int(min_far_total),
        "min_event_train_total": int(min_event_train_total),
        "min_far_train_total": int(min_far_train_total),
        "min_train": int(min_train),
        "min_test": int(min_test),
    }
    return out, info


def main() -> int:
    args = parse_args()
    register_builtin_adapters()
    dataset_dir = Path(args.dataset)
    config_path = Path(args.config)
    out_path = Path(args.out)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    samples_all = pd.read_parquet(dataset_dir / "samples.parquet")
    samples_all["t"] = pd.to_datetime(samples_all["t"], errors="coerce")
    if "onset_time" in samples_all.columns:
        samples_all["onset_time"] = pd.to_datetime(samples_all["onset_time"], errors="coerce")
    shard_stratify_tokens: list[str] = []
    for tok in (args.shard_stratify_by or []):
        for part in str(tok).split(","):
            part = str(part).strip()
            if part:
                shard_stratify_tokens.append(part)
    claim_far_quality_tags = [
        str(v).strip()
        for tok in (args.claim_far_quality_tags or [])
        for v in str(tok).split(",")
        if str(v).strip()
    ]
    geometry_required_checks = [
        str(v).strip()
        for tok in (args.geometry_required_checks or [])
        for v in str(tok).split(",")
        if str(v).strip()
    ]
    strict_time_basis = str(args.strict_coverage_time_basis).strip().lower()
    if strict_time_basis not in {"selected", "source", "valid"}:
        strict_time_basis = "selected"
    canonical_time_basis = str(args.canonical_time_basis).strip().lower()
    if canonical_time_basis not in {"auto", "selected", "source", "valid"}:
        canonical_time_basis = "auto"
    if canonical_time_basis != "auto":
        strict_time_basis = str(canonical_time_basis)
    strict_use_flags = bool(args.strict_coverage_use_flags)
    strict_min_event_test_total = int(args.min_strict_event_test_cases) if int(args.min_strict_event_test_cases) > 0 else int(args.min_event_test_cases)
    strict_min_far_test_total = int(args.min_strict_far_test_cases) if int(args.min_strict_far_test_cases) > 0 else int(args.min_far_nonstorm_test_cases)
    strict_min_event_test_per_lead = int(args.min_strict_event_test_per_lead) if int(args.min_strict_event_test_per_lead) > 0 else int(args.min_event_test_per_lead)
    strict_min_far_test_per_lead = int(args.min_strict_far_test_per_lead) if int(args.min_strict_far_test_per_lead) > 0 else int(args.min_far_nonstorm_test_per_lead)
    strict_min_event_train_total = int(args.min_strict_event_train_cases) if int(args.min_strict_event_train_cases) > 0 else int(strict_min_event_test_total)
    strict_min_far_train_total = int(args.min_strict_far_train_cases) if int(args.min_strict_far_train_cases) > 0 else int(strict_min_far_test_total)
    strict_min_event_train_per_lead = int(args.min_strict_event_train_per_lead) if int(args.min_strict_event_train_per_lead) > 0 else int(strict_min_event_test_per_lead)
    strict_min_far_train_per_lead = int(args.min_strict_far_train_per_lead) if int(args.min_strict_far_train_per_lead) > 0 else int(strict_min_far_test_per_lead)

    initial_storm_id_col = _resolve_storm_id_col(samples_all, requested=args.storm_id_col)
    shard_manifest = _build_evaluation_shard_manifest(
        samples=samples_all,
        split_date=pd.Timestamp(args.split_date),
        time_buffer_hours=float(args.time_buffer_hours),
        case_shards=int(args.case_shards),
        max_cases=int(args.max_cases),
        stratify_by=tuple(shard_stratify_tokens),
        storm_id_col=initial_storm_id_col,
        strict_time_basis=strict_time_basis,
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_event_test_total=int(strict_min_event_test_total),
        min_far_test_total=int(strict_min_far_test_total),
        min_event_test_per_lead=int(strict_min_event_test_per_lead),
        min_far_test_per_lead=int(strict_min_far_test_per_lead),
        min_event_train_total=int(strict_min_event_train_total),
        min_far_train_total=int(strict_min_far_train_total),
        min_event_train_per_lead=int(strict_min_event_train_per_lead),
        min_far_train_per_lead=int(strict_min_far_train_per_lead),
        min_train_cases=int(args.min_train_cases),
        min_test_cases=int(args.min_test_cases),
        current_shard_index=int(args.case_shard_index),
        strict_far_min_row_quality_frac=float(args.strict_far_min_row_quality_frac),
        strict_far_min_kinematic_clean_frac=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )

    samples, case_subset_info = _select_case_subset(
        samples_all,
        max_cases=int(args.max_cases),
        case_shards=int(args.case_shards),
        case_shard_index=int(args.case_shard_index),
        stratify_by=tuple(shard_stratify_tokens),
        storm_id_col=initial_storm_id_col,
        split_date=pd.Timestamp(args.split_date),
        time_buffer_hours=float(args.time_buffer_hours),
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_event_total=int(strict_min_event_test_total),
        min_far_total=int(strict_min_far_test_total),
        min_event_train_total=int(strict_min_event_train_total),
        min_far_train_total=int(strict_min_far_train_total),
        min_train=int(args.min_train_cases),
        min_test=int(args.min_test_cases),
        strict_far_min_row_quality_frac=float(args.strict_far_min_row_quality_frac),
        strict_far_min_kinematic_clean_frac=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    if samples.empty:
        raise ValueError("no_rows_after_case_subset")
    split_date = pd.Timestamp(args.split_date)
    rng = np.random.default_rng(int(args.seed))

    lead_per_case = samples.groupby("case_id")["lead_bucket"].nunique(dropna=False)
    mixed = lead_per_case[lead_per_case > 1]
    if not mixed.empty:
        raise ValueError(
            f"Lead leakage risk: {mixed.shape[0]} case_id values mix lead_bucket values"
        )

    storm_id_col = _resolve_storm_id_col(samples, requested=args.storm_id_col)
    case_meta = _build_case_meta(samples, storm_id_col=storm_id_col)
    split = _blocked_time_split(case_meta, split_date=split_date, buffer_hours=float(args.time_buffer_hours))
    train_cases = split["train_cases"]
    test_cases = split["test_cases"]
    buffer_cases = split["buffer_cases"]
    _enforce_split_case_counts(
        train_cases=train_cases,
        test_cases=test_cases,
        min_train=int(args.min_train_cases),
        min_test=int(args.min_test_cases),
    )
    test_cov = _test_case_coverage(case_meta=case_meta, test_cases=test_cases)
    _enforce_test_coverage(
        coverage=test_cov,
        min_event_total=int(args.min_event_test_cases),
        min_event_per_lead=int(args.min_event_test_per_lead),
        min_total=int(args.min_far_nonstorm_test_cases),
        min_per_lead=int(args.min_far_nonstorm_test_per_lead),
    )
    strict_cov_test = _strict_ib_coverage(
        samples=samples,
        case_ids=test_cases,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    strict_cov_train = _strict_ib_coverage(
        samples=samples,
        case_ids=train_cases,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    strict_test_ok, strict_test_reasons = _strict_coverage_ok(
        coverage=strict_cov_test,
        min_event_total=int(strict_min_event_test_total),
        min_far_total=int(strict_min_far_test_total),
        min_event_per_lead=int(strict_min_event_test_per_lead),
        min_far_per_lead=int(strict_min_far_test_per_lead),
    )
    strict_train_ok, strict_train_reasons = _strict_coverage_ok(
        coverage=strict_cov_train,
        min_event_total=int(strict_min_event_train_total),
        min_far_total=int(strict_min_far_train_total),
        min_event_per_lead=int(strict_min_event_train_per_lead),
        min_far_per_lead=int(strict_min_far_train_per_lead),
    )
    strict_cov_ok = bool(strict_test_ok and strict_train_ok)
    strict_cov_reasons = [f"test:{v}" for v in strict_test_reasons] + [f"train:{v}" for v in strict_train_reasons]
    if not bool(strict_cov_ok):
        raise ValueError("insufficient_strict_ib_coverage:" + ";".join([str(v) for v in strict_cov_reasons]))
    train_df = samples.loc[samples["case_id"].astype(str).isin(train_cases)].copy()
    test_df = samples.loc[samples["case_id"].astype(str).isin(test_cases)].copy()
    axis_meta = _load_axis_robust_meta(
        dataset_dir=dataset_dir,
        axis_json_arg=args.axis_sensitivity_json,
        std_threshold=float(args.axis_std_threshold),
    )

    train_case_metrics = _evaluate_cases(
        train_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed),
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
        parity_tangential_radial_min=float(args.parity_tangential_radial_min),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
    )
    test_case_metrics = _evaluate_cases(
        test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 1,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
        parity_tangential_radial_min=float(args.parity_tangential_radial_min),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
    )
    train_case_metrics = _apply_axis_robust_gating(train_case_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
    test_case_metrics = _apply_axis_robust_gating(test_case_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))

    summary = {
        "generated_at_utc": utc_now_iso(),
        "dataset": str(dataset_dir.resolve()),
        "config": str(config_path.resolve()),
        "split_date": str(split_date.date()),
        "time_buffer_hours": float(args.time_buffer_hours),
        "storm_id_col": storm_id_col,
        "time_frequency_knee_enabled": bool(args.enable_time_frequency_knee),
        "tf_knee_bic_delta_min": float(args.tf_knee_bic_delta_min),
        "slowtick_delta_min": float(args.slowtick_delta_min),
        "slowtick_p_max": float(args.slowtick_p_max),
        "axis_robust": axis_meta,
        "n_rows_total": int(samples.shape[0]),
        "n_cases_total": int(samples["case_id"].nunique()),
        "n_rows_dataset_full": int(samples_all.shape[0]),
        "n_cases_dataset_full": int(samples_all["case_id"].nunique()) if "case_id" in samples_all.columns else 0,
        "case_subset": case_subset_info,
        "evaluation_shard_manifest": shard_manifest,
        "n_cases_train": int(len(train_cases)),
        "n_cases_test": int(len(test_cases)),
        "n_cases_buffer_excluded": int(len(buffer_cases)),
        "test_coverage": test_cov,
        "strict_test_coverage": strict_cov_test,
        "strict_train_coverage": strict_cov_train,
        "strict_coverage_gate": {
            "passed": bool(strict_cov_ok),
            "reasons": [str(v) for v in strict_cov_reasons],
            "time_basis": str(strict_time_basis),
            "use_flags": bool(strict_use_flags),
            "thresholds": {
                "min_event_total": int(strict_min_event_test_total),
                "min_far_total": int(strict_min_far_test_total),
                "min_event_per_lead": int(strict_min_event_test_per_lead),
                "min_far_per_lead": int(strict_min_far_test_per_lead),
                "test": {
                    "min_event_total": int(strict_min_event_test_total),
                    "min_far_total": int(strict_min_far_test_total),
                    "min_event_per_lead": int(strict_min_event_test_per_lead),
                    "min_far_per_lead": int(strict_min_far_test_per_lead),
                },
                "train": {
                    "min_event_total": int(strict_min_event_train_total),
                    "min_far_total": int(strict_min_far_train_total),
                    "min_event_per_lead": int(strict_min_event_train_per_lead),
                    "min_far_per_lead": int(strict_min_far_train_per_lead),
                },
            },
        },
        "far_nonstorm_coverage_test": test_cov.get("far_nonstorm", {}),
        "event_coverage_test": test_cov.get("events", {}),
        "min_train_cases": int(args.min_train_cases),
        "min_test_cases": int(args.min_test_cases),
        "min_event_test_cases": int(args.min_event_test_cases),
        "min_event_test_per_lead": int(args.min_event_test_per_lead),
        "min_far_nonstorm_test_cases": int(args.min_far_nonstorm_test_cases),
        "min_far_nonstorm_test_per_lead": int(args.min_far_nonstorm_test_per_lead),
        "min_strict_event_test_cases": int(strict_min_event_test_total),
        "min_strict_far_test_cases": int(strict_min_far_test_total),
        "min_strict_event_test_per_lead": int(strict_min_event_test_per_lead),
        "min_strict_far_test_per_lead": int(strict_min_far_test_per_lead),
        "min_strict_event_train_cases": int(strict_min_event_train_total),
        "min_strict_far_train_cases": int(strict_min_far_train_total),
        "min_strict_event_train_per_lead": int(strict_min_event_train_per_lead),
        "min_strict_far_train_per_lead": int(strict_min_far_train_per_lead),
        "claim_far_quality_tags": [str(v) for v in claim_far_quality_tags],
        "geometry_required_checks": [str(v) for v in geometry_required_checks],
        "canonical_time_basis": str(canonical_time_basis),
        "canonical_time_basis_enforced": bool(canonical_time_basis != "auto"),
        "parity_odd_inner_outer_min": float(args.parity_odd_inner_outer_min),
        "parity_tangential_radial_min": float(args.parity_tangential_radial_min),
        "parity_inner_ring_quantile": float(args.parity_inner_ring_quantile),
        "split_audit": _build_split_audit(
            case_meta=case_meta,
            train_cases=train_cases,
            test_cases=test_cases,
            buffer_cases=buffer_cases,
            split_date=split_date,
            buffer_hours=float(args.time_buffer_hours),
            storm_id_col=storm_id_col,
        ),
        "train": _summarize_case_metrics_by_type(train_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "test": _summarize_case_metrics_by_type(test_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "strata": {
            "train": _summarize_case_metrics_by_strata(train_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
            "test": _summarize_case_metrics_by_strata(test_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        },
        "control_tiers": {
            "train": _summarize_case_metrics_by_control_tier(train_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
            "test": _summarize_case_metrics_by_control_tier(test_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        },
        "center_sources": {
            "train": _summarize_case_metrics_by_center_source(train_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
            "test": _summarize_case_metrics_by_center_source(test_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        },
        "knee_detector_comparison": {
            "train": _build_knee_detector_comparison(train_case_metrics),
            "test": _build_knee_detector_comparison(test_case_metrics),
        },
        "knee_trace": {
            "train": _build_knee_trace(train_case_metrics),
            "test": _build_knee_trace(test_case_metrics),
        },
        "size_knee_diagnostic": _build_size_knee_diagnostic(test_df, top_k=10),
    }
    summary["test"]["knee_hit_rate_pre_onset_storm"] = summary["test"].get("storm", {}).get("knee_rate", 0.0)
    summary["test"]["false_knee_rate_controls"] = summary["test"].get("control", {}).get("knee_rate", 0.0)

    null_transforms: dict[str, NullTransform] = {
        "direction_randomization": _direction_randomization,
        "spatial_shuffle": _spatial_shuffle_control,
        "time_permutation_within_lead": _time_permutation_within_lead,
        "time_permutation_within_track_phase": _time_permutation_within_lead,
        "lat_band_shuffle_within_month_hour": _lat_band_shuffle_within_month_hour,
        "fake_mirror_pairing": _fake_mirror_pairing,
        "latitude_mirror_pairing": _latitude_mirror_pairing,
        "circular_lon_shift_pairing": _circular_lon_shift_pairing,
        "theta_roll": _theta_roll_polar,
        "theta_scramble_within_ring": _theta_scramble_within_ring,
        "radial_shuffle": _radial_shuffle_polar,
        "radial_scramble": _radial_scramble_polar,
        "mirror_axis_jitter": _mirror_axis_jitter,
        "center_jitter": _center_jitter_polar,
        "center_swap": _center_swap_polar,
        "storm_track_reassignment": _storm_track_reassignment,
        "storm_track_reassignment_phase_matched": _storm_track_reassignment,
    }
    requested_nulls = {
        str(v).strip()
        for tok in (args.null_transforms or [])
        for v in str(tok).split(",")
        if str(v).strip()
    }
    if requested_nulls:
        null_transforms = {k: v for k, v in null_transforms.items() if k in requested_nulls}
    null_controls: dict[str, dict[str, Any]] = {}
    null_case_metrics: dict[str, pd.DataFrame] = {}
    null_strata: dict[str, dict[str, Any]] = {}
    for i, (name, transform) in enumerate(null_transforms.items()):
        transformed = transform(test_df, rng=np.random.default_rng(int(args.seed) + 101 + i))
        null_controls[name] = _run_partition_pipeline(
            transformed,
            template_dataset=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 201 + i,
        )
        metrics = _evaluate_cases(
            transformed,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 301 + i,
            enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
            tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
            parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
            parity_tangential_radial_min=float(args.parity_tangential_radial_min),
            parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
        )
        metrics = _apply_axis_robust_gating(metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
        null_case_metrics[name] = metrics
        null_strata[name] = _summarize_case_metrics_by_strata(metrics, p_lock_threshold=float(args.p_lock_threshold))
    summary["null_controls"] = null_controls
    summary["null_strata"] = null_strata
    geom_audit_keys = (
        "theta_roll",
        "theta_scramble_within_ring",
        "center_swap",
        "storm_track_reassignment",
        "time_permutation_within_lead",
        "lat_band_shuffle_within_month_hour",
        "radial_shuffle",
        "radial_scramble",
        "mirror_axis_jitter",
        "center_jitter",
    )
    geometry_null_audit = _build_geometry_null_audit(
        samples=test_df,
        transforms={k: null_transforms[k] for k in geom_audit_keys if k in null_transforms},
        seed=int(args.seed) + 13000,
    )
    summary["geometry_null_audit"] = geometry_null_audit
    summary["slowtick_report"] = _build_slowtick_report(
        test_df=test_df,
        template_dataset=dataset_dir,
        config_path=config_path,
        null_transforms=null_transforms,
        seed=int(args.seed) + 1001,
        delta_min=float(args.slowtick_delta_min),
        p_max=float(args.slowtick_p_max),
    )
    parity_ablation = _build_parity_ablation_report(
        test_df=test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 5000,
        p_lock_threshold=float(args.p_lock_threshold),
        axis_robust=bool(axis_meta.get("axis_robust", False)),
        parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
        parity_tangential_radial_min=float(args.parity_tangential_radial_min),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
    )
    summary["parity_ablation"] = parity_ablation
    anomaly_ablation, anomaly_mode_case_metrics = _build_anomaly_mode_ablation(
        test_df=test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 7000,
        p_lock_threshold=float(args.p_lock_threshold),
        axis_robust=bool(axis_meta.get("axis_robust", False)),
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
        parity_tangential_radial_min=float(args.parity_tangential_radial_min),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
    )
    summary["anomaly_mode_ablation"] = anomaly_ablation

    anomaly_selection_train = _select_canonical_anomaly_mode_by_lead(
        train_df=train_df,
        seed=int(args.seed) + 7100,
        far_tolerance=float(args.anomaly_selection_far_tolerance),
    )
    summary["anomaly_mode_selection"] = anomaly_selection_train

    anomaly_mode_priority = _rank_anomaly_modes_for_canonical(
        anomaly_ablation,
        mode_agreement_min=float(args.anomaly_mode_agreement_min),
    )
    anomaly_selection_casewise = _build_casewise_anomaly_mode_selection(
        mode_case_metrics=anomaly_mode_case_metrics,
        mode_priority=anomaly_mode_priority,
        stability_min=float(args.anomaly_case_stability_min),
    )
    summary["anomaly_mode_selection_casewise"] = anomaly_selection_casewise["summary"]

    anomaly_mode_gate = _build_anomaly_mode_gate(
        anomaly_ablation=anomaly_ablation,
        anomaly_selection_summary=anomaly_selection_train,
        agreement_mean_min=float(args.anomaly_agreement_mean_min),
        agreement_min_min=float(args.anomaly_agreement_min_min),
    )
    summary["anomaly_mode_gate"] = anomaly_mode_gate

    summary["mode_invariant_parity"] = _build_mode_invariant_parity_summary(anomaly_selection_casewise["per_case"])
    summary["anomaly_mode_distribution_audit"] = _build_anomaly_mode_distribution_audit(anomaly_selection_casewise["per_case"])
    mode_invariant_gate = _build_mode_invariant_claim_gate(
        selection_df=anomaly_selection_casewise["per_case"],
        event_minus_far_min=float(args.mode_invariant_event_minus_far_min),
        far_max=float(args.mode_invariant_far_max),
    )
    summary["mode_invariant_gate"] = mode_invariant_gate
    mode_margin_gate = _build_mode_margin_gate(
        anomaly_ablation=anomaly_ablation,
        event_minus_far_min=float(args.mode_invariant_event_minus_far_min),
        far_max=float(args.mode_invariant_far_max),
    )
    summary["mode_margin_gate"] = mode_margin_gate

    canonical_case_metrics = _assemble_case_metrics_from_mode_selection(
        mode_case_metrics=anomaly_mode_case_metrics,
        selection_df=anomaly_selection_casewise["per_case"],
        mode_col="canonical_mode",
        decision_col="canonical_parity_signal_pass",
    )
    ensemble_case_metrics = _assemble_case_metrics_from_mode_selection(
        mode_case_metrics=anomaly_mode_case_metrics,
        selection_df=anomaly_selection_casewise["per_case"],
        mode_col="canonical_mode",
        decision_col="ensemble_parity_signal_pass",
    )
    summary["claim_anomaly_canonical"] = {
        "test": _summarize_case_metrics_by_type(canonical_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "strata": _summarize_case_metrics_by_strata(canonical_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
    }
    summary["claim_anomaly_ensemble"] = {
        "test": _summarize_case_metrics_by_type(ensemble_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "strata": _summarize_case_metrics_by_strata(ensemble_case_metrics, p_lock_threshold=float(args.p_lock_threshold)),
    }

    claim_retention_train_stages: dict[str, dict[str, Any]] = {
        "pre_claim_input": _strict_case_counts_from_samples(
            samples=train_df,
            strict_time_basis=str(strict_time_basis),
            strict_use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
    }
    claim_retention_test_stages: dict[str, dict[str, Any]] = {
        "pre_claim_input": _strict_case_counts_from_samples(
            samples=test_df,
            strict_time_basis=str(strict_time_basis),
            strict_use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
    }

    requested_claim_mode = str(args.parity_claim_mode or "auto")
    if requested_claim_mode == "auto":
        base_claim_mode = _select_claim_mode(
            requested_mode=requested_claim_mode,
            parity_ablation=parity_ablation,
            anomaly_mode_gate=anomaly_mode_gate,
        )
        summary["claim_mode_base_auto"] = base_claim_mode
        lead_mode_map = {
            str(k): str(v)
            for k, v in (anomaly_selection_train.get("canonical_mode_by_lead", {}) or {}).items()
            if isinstance(v, str) and _mode_to_column(v) is not None
        }
        default_mode = str(anomaly_selection_train.get("default_mode", "none"))
        if lead_mode_map and base_claim_mode in {"current", "raw", "anomaly_lat_hour", "anomaly_lat_day"}:
            claim_mode = "canonical_by_lead"
            claim_train_df = _with_observable_mode_by_lead(
                train_df,
                mode_by_lead=lead_mode_map,
                default_mode=default_mode,
            )
            claim_test_df = _with_observable_mode_by_lead(
                test_df,
                mode_by_lead=lead_mode_map,
                default_mode=default_mode,
            )
        else:
            claim_mode = base_claim_mode
            claim_train_df = _with_observable_mode(train_df, mode=claim_mode)
            claim_test_df = _with_observable_mode(test_df, mode=claim_mode)
    else:
        claim_mode = requested_claim_mode
        claim_train_df = _with_observable_mode(train_df, mode=claim_mode)
        claim_test_df = _with_observable_mode(test_df, mode=claim_mode)

    claim_retention_train_stages["after_mode_application"] = _strict_case_counts_from_samples(
        samples=claim_train_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_retention_test_stages["after_mode_application"] = _strict_case_counts_from_samples(
        samples=claim_test_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )

    claim_train_df, claim_test_df, claim_lead_filter = _filter_claim_leads_by_far_coverage(
        train_df=claim_train_df,
        test_df=claim_test_df,
        min_far_per_lead=int(args.claim_min_far_per_lead),
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_retention_train_stages["after_lead_filter"] = _strict_case_counts_from_samples(
        samples=claim_train_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_retention_test_stages["after_lead_filter"] = _strict_case_counts_from_samples(
        samples=claim_test_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_train_df, claim_test_df, strict_far_claim_filter = _filter_claim_to_strict_far_controls(
        train_df=claim_train_df,
        test_df=claim_test_df,
        allowed_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
    )
    claim_retention_train_stages["after_strict_far_filter"] = _strict_case_counts_from_samples(
        samples=claim_train_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_retention_test_stages["after_strict_far_filter"] = _strict_case_counts_from_samples(
        samples=claim_test_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    parity_thresholds_applied = {
        "odd_inner_outer_min": float(args.parity_odd_inner_outer_min),
        "tangential_radial_min": float(args.parity_tangential_radial_min),
        "inner_ring_quantile": float(args.parity_inner_ring_quantile),
    }
    parity_threshold_calibration: dict[str, Any] = {
        "enabled": bool(args.calibrate_parity_thresholds),
        "available": False,
        "reason": "disabled",
        "thresholds": dict(parity_thresholds_applied),
    }
    strict_claim_cohorts: dict[str, Any] = {
        "calibration_train": {},
        "claim_train": {},
        "claim_test": {},
    }
    parity_feature_distributions = _parity_feature_distribution_table(pd.DataFrame())
    if bool(args.calibrate_parity_thresholds):
        calib_train_metrics = _evaluate_cases(
            claim_train_df,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 8900,
            enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
            tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
            parity_odd_inner_outer_min=0.0,
            parity_tangential_radial_min=0.0,
            parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
        )
        calib_train_metrics = _apply_axis_robust_gating(
            calib_train_metrics, axis_robust=bool(axis_meta.get("axis_robust", False))
        )
        calib_train_metrics, strict_claim_cohorts["calibration_train"] = _apply_strict_claim_strata(
            case_metrics=calib_train_metrics,
            samples=claim_train_df,
            strict_time_basis=str(strict_time_basis),
            strict_use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        parity_feature_distributions = _parity_feature_distribution_table(calib_train_metrics)
        calib = _calibrate_parity_thresholds_from_train(
            train_case_metrics=calib_train_metrics,
            default_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
            default_tangential_radial_min=float(args.parity_tangential_radial_min),
            event_min_floor=float(args.parity_calibration_event_min_floor),
            constrain_min_thresholds=bool(args.parity_calibration_constrain_min_thresholds),
        )
        parity_threshold_calibration = {
            **calib,
            "enabled": True,
            "inner_ring_quantile": float(args.parity_inner_ring_quantile),
            "strict_train_cohorts": strict_claim_cohorts.get("calibration_train", {}),
        }
        if bool(calib.get("available", False)):
            thr = calib.get("thresholds", {}) or {}
            parity_thresholds_applied["odd_inner_outer_min"] = float(
                _to_float(thr.get("odd_inner_outer_min")) or parity_thresholds_applied["odd_inner_outer_min"]
            )
            parity_thresholds_applied["tangential_radial_min"] = float(
                _to_float(thr.get("tangential_radial_min")) or parity_thresholds_applied["tangential_radial_min"]
            )

    claim_retention_train_stages["after_case_reduction_calibration"] = {
        **(strict_claim_cohorts.get("calibration_train", {}) or {}),
        "n_rows": int(claim_train_df.shape[0]),
    }

    summary["claim_mode_applied"] = claim_mode
    summary["claim_lead_filter"] = claim_lead_filter
    summary["claim_strict_far_filter"] = strict_far_claim_filter
    summary["claim_strict_cohorts"] = strict_claim_cohorts
    summary["parity_threshold_calibration"] = parity_threshold_calibration
    summary["parity_thresholds_applied"] = parity_thresholds_applied
    claim_train_metrics = _evaluate_cases(
        claim_train_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 9000,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        parity_odd_inner_outer_min=float(parity_thresholds_applied["odd_inner_outer_min"]),
        parity_tangential_radial_min=float(parity_thresholds_applied["tangential_radial_min"]),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
        alignment_eta_threshold=float(args.alignment_eta_threshold),
        alignment_block_hours=float(args.alignment_block_hours),
    )
    claim_test_metrics = _evaluate_cases(
        claim_test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 9001,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        parity_odd_inner_outer_min=float(parity_thresholds_applied["odd_inner_outer_min"]),
        parity_tangential_radial_min=float(parity_thresholds_applied["tangential_radial_min"]),
        parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
        alignment_eta_threshold=float(args.alignment_eta_threshold),
        alignment_block_hours=float(args.alignment_block_hours),
    )
    claim_train_metrics = _apply_axis_robust_gating(claim_train_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
    claim_test_metrics = _apply_axis_robust_gating(claim_test_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
    claim_train_metrics, strict_claim_cohorts["claim_train"] = _apply_strict_claim_strata(
        case_metrics=claim_train_metrics,
        samples=claim_train_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_test_metrics, strict_claim_cohorts["claim_test"] = _apply_strict_claim_strata(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        strict_time_basis=str(strict_time_basis),
        strict_use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(str(v) for v in claim_far_quality_tags),
        min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
    )
    claim_retention_train_stages["after_case_reduction_claim"] = {
        **(strict_claim_cohorts.get("claim_train", {}) or {}),
        "n_rows": int(claim_train_df.shape[0]),
    }
    claim_retention_test_stages["after_case_reduction_claim"] = {
        **(strict_claim_cohorts.get("claim_test", {}) or {}),
        "n_rows": int(claim_test_df.shape[0]),
    }
    claim_retention_audit = _build_claim_retention_audit(
        train_stages=claim_retention_train_stages,
        test_stages=claim_retention_test_stages,
    )
    strict_train_events = int((strict_cov_train or {}).get("event_total", 0) or 0)
    strict_train_far = int((strict_cov_train or {}).get("far_total", 0) or 0)
    calib_ev = int((strict_claim_cohorts.get("calibration_train", {}) or {}).get("n_cases_event", 0) or 0)
    calib_far = int((strict_claim_cohorts.get("calibration_train", {}) or {}).get("n_cases_far", 0) or 0)
    claim_path_starvation = bool(strict_train_events > 0 and strict_train_far > 0 and (calib_ev <= 0 or calib_far <= 0))
    claim_retention_audit["starvation"] = {
        "claim_path_event_starvation": bool(claim_path_starvation),
        "strict_train_event_total": int(strict_train_events),
        "strict_train_far_total": int(strict_train_far),
        "calibration_train_event_cases": int(calib_ev),
        "calibration_train_far_cases": int(calib_far),
    }
    if claim_path_starvation:
        summary["parity_threshold_calibration"]["reason"] = "claim_path_event_starvation"
        summary["parity_threshold_calibration"]["available"] = bool(
            summary["parity_threshold_calibration"].get("available", False)
        )
    summary["claim_retention_audit"] = claim_retention_audit
    summary["claim_strict_cohorts"] = strict_claim_cohorts

    claim_strata = {
        "train": _summarize_case_metrics_by_strata(claim_train_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "test": _summarize_case_metrics_by_strata(claim_test_metrics, p_lock_threshold=float(args.p_lock_threshold)),
    }
    summary["claim"] = {
        "train": _summarize_case_metrics_by_type(claim_train_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "test": _summarize_case_metrics_by_type(claim_test_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        "strata": claim_strata,
        "control_tiers": {
            "train": _summarize_case_metrics_by_control_tier(claim_train_metrics, p_lock_threshold=float(args.p_lock_threshold)),
            "test": _summarize_case_metrics_by_control_tier(claim_test_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        },
        "center_sources": {
            "train": _summarize_case_metrics_by_center_source(claim_train_metrics, p_lock_threshold=float(args.p_lock_threshold)),
            "test": _summarize_case_metrics_by_center_source(claim_test_metrics, p_lock_threshold=float(args.p_lock_threshold)),
        },
        "knee_detector_comparison": {
            "train": _build_knee_detector_comparison(claim_train_metrics),
            "test": _build_knee_detector_comparison(claim_test_metrics),
        },
        "knee_trace": {
            "train": _build_knee_trace(claim_train_metrics),
            "test": _build_knee_trace(claim_test_metrics),
        },
    }

    parity_confound_gate = _build_parity_confound_gate(
        strata_test=claim_strata.get("test", {}),
        confound_max_ratio=float(args.parity_confound_max_ratio),
        event_minus_far_min=float(args.parity_event_minus_far_min),
        far_nonstorm_max=float(args.parity_far_max),
        use_alignment_fraction=bool(args.parity_use_alignment_fraction),
    )
    summary["parity_confound_gate"] = parity_confound_gate
    geometry_null_strata: dict[str, dict[str, Any]] = {}
    requested_required_checks = [str(v).strip() for v in geometry_required_checks if str(v).strip()]
    optional_checks = [
        "lat_band_shuffle_within_month_hour",
        "center_jitter",
        "mirror_axis_jitter",
        "radial_scramble",
        "radial_shuffle",
        "theta_scramble_within_ring",
    ]
    geometry_null_names = tuple(dict.fromkeys(requested_required_checks + optional_checks))
    geometry_null_names = tuple(name for name in geometry_null_names if name in null_transforms)
    center_swap_distance_stats: dict[str, Any] = {}
    for j, name in enumerate(geometry_null_names):
        transform = null_transforms[name]
        transformed_claim = transform(claim_test_df, rng=np.random.default_rng(int(args.seed) + 9600 + j))
        if str(name) == "center_swap":
            dist_vals = pd.to_numeric(transformed_claim.get("center_swap_center_distance_km"), errors="coerce")
            dist_vals = dist_vals[np.isfinite(dist_vals)]
            if dist_vals.size > 0:
                center_swap_distance_stats = {
                    "n": int(dist_vals.size),
                    "p10_km": float(np.nanpercentile(dist_vals, 10)),
                    "p50_km": float(np.nanpercentile(dist_vals, 50)),
                    "p90_km": float(np.nanpercentile(dist_vals, 90)),
                }
            else:
                center_swap_distance_stats = {"n": 0}
        geom_metrics = _evaluate_cases(
            transformed_claim,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 9700 + j,
            enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
            tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
            parity_odd_inner_outer_min=float(args.parity_odd_inner_outer_min),
            parity_tangential_radial_min=float(args.parity_tangential_radial_min),
            parity_inner_ring_quantile=float(args.parity_inner_ring_quantile),
            alignment_eta_threshold=float(args.alignment_eta_threshold),
            alignment_block_hours=float(args.alignment_block_hours),
        )
        geom_metrics = _apply_axis_robust_gating(geom_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
        geometry_null_strata[name] = _summarize_case_metrics_by_strata(geom_metrics, p_lock_threshold=float(args.p_lock_threshold))
    summary["geometry_null_strata"] = geometry_null_strata
    geometry_null_collapse = _build_geometry_null_collapse_gate(
        observed_strata=claim_strata.get("test", {}),
        null_strata=geometry_null_strata,
        min_rel_drop=float(args.null_collapse_min_drop),
        min_abs_drop=float(args.null_collapse_min_abs_drop),
        rate_key=(
            "alignment_event_weighted_rate"
            if str(parity_confound_gate.get("rate_basis", "")).startswith("alignment")
            else "parity_signal_rate"
        ),
        no_signal_max_rate=float(args.null_no_signal_max_rate),
        required_checks=tuple(requested_required_checks)
        if requested_required_checks
        else ("theta_roll", "center_swap", "storm_track_reassignment", "time_permutation_within_lead"),
        center_swap_distance_stats=center_swap_distance_stats,
    )
    summary["geometry_null_collapse"] = geometry_null_collapse
    summary["geometry_null_by_mode"] = _build_geometry_null_by_mode(
        samples=test_df,
        seed=int(args.seed) + 9900,
    )
    vortex_centered = _is_vortex_centered_context(samples=test_df)
    angular_witness = _build_angular_witness_report(
        observed_samples=claim_test_df,
        theta_roll_samples=(null_transforms.get("theta_roll", _theta_roll_polar))(
            claim_test_df, rng=np.random.default_rng(int(args.seed) + 9801)
        ),
        center_swap_samples=(null_transforms.get("center_swap", _center_swap_polar))(
            claim_test_df, rng=np.random.default_rng(int(args.seed) + 9802)
        ),
        center_jitter_samples=(null_transforms.get("center_jitter", _center_jitter_polar))(
            claim_test_df, rng=np.random.default_rng(int(args.seed) + 9803)
        ),
        margin_min=float(args.angular_witness_margin_min),
        d_min=float(args.angular_witness_d_min),
        p_max=float(args.angular_witness_p_max),
        null_drop_min=float(args.angular_witness_null_drop_min),
        null_abs_drop_min=float(args.angular_witness_null_abs_drop_min),
        null_margin_max=float(args.angular_witness_null_margin_max),
        permutation_n=int(args.angular_witness_permutation_n),
        required=bool(vortex_centered),
    )
    summary["angular_witness"] = angular_witness
    ibtracs_strict_eval_source = _build_ibtracs_strict_eval(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        radius_km=float(args.ibtracs_strict_radius_km),
        far_min_km=float(args.ibtracs_strict_far_min_km),
        time_hours=float(args.ibtracs_strict_time_hours),
        p_lock_threshold=float(args.p_lock_threshold),
        use_flags=bool(args.ibtracs_strict_use_flags),
        time_basis="source",
    )
    ibtracs_strict_eval_valid = _build_ibtracs_strict_eval(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        radius_km=float(args.ibtracs_strict_radius_km),
        far_min_km=float(args.ibtracs_strict_far_min_km),
        time_hours=float(args.ibtracs_strict_time_hours),
        p_lock_threshold=float(args.p_lock_threshold),
        use_flags=bool(args.ibtracs_strict_use_flags),
        time_basis="valid",
    )
    ibtracs_strict_eval = _build_ibtracs_strict_eval(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        radius_km=float(args.ibtracs_strict_radius_km),
        far_min_km=float(args.ibtracs_strict_far_min_km),
        time_hours=float(args.ibtracs_strict_time_hours),
        p_lock_threshold=float(args.p_lock_threshold),
        use_flags=bool(args.ibtracs_strict_use_flags),
        time_basis="selected",
    )
    selected_basis = str(args.ibtracs_selected_time_basis).strip().lower()
    if canonical_time_basis != "auto":
        if selected_basis not in {"auto", str(canonical_time_basis)}:
            raise ValueError(
                "time_basis_contract_violation:"
                f"canonical={canonical_time_basis};ibtracs_selected={selected_basis}"
            )
        selected_basis = str(canonical_time_basis)
    elif selected_basis == "auto":
        basis_counts = (
            claim_test_df.get("ib_time_basis_used", pd.Series(dtype=object))
            .fillna(claim_test_df.get("time_basis", pd.Series(dtype=object)))
            .fillna("source")
            .astype(str)
            .str.lower()
            .value_counts(dropna=False)
            .to_dict()
        )
        if basis_counts:
            selected_basis = str(max(basis_counts.items(), key=lambda kv: kv[1])[0])
        else:
            selected_basis = str(strict_time_basis)
    if selected_basis not in {"source", "valid", "selected"}:
        selected_basis = "selected"
    if selected_basis == "source":
        ibtracs_eval_selected = ibtracs_strict_eval_source
    elif selected_basis == "valid":
        ibtracs_eval_selected = ibtracs_strict_eval_valid
    else:
        ibtracs_eval_selected = ibtracs_strict_eval
    basis_flip_flag = False
    src_margin = _to_float((ibtracs_strict_eval_source or {}).get("event_minus_far_parity_rate"))
    val_margin = _to_float((ibtracs_strict_eval_valid or {}).get("event_minus_far_parity_rate"))
    if (src_margin is not None) and (val_margin is not None):
        basis_flip_flag = bool((src_margin * val_margin) < 0.0 or abs(src_margin - val_margin) >= 0.15)
    ibtracs_alignment_gate_source = _build_ibtracs_alignment_gate(
        strict_eval=ibtracs_strict_eval_source,
        min_event_cases=int(args.ibtracs_alignment_min_event_cases),
        min_far_cases=int(args.ibtracs_alignment_min_far_cases),
        margin_min=float(args.ibtracs_alignment_margin_min),
        confound_max=float(args.ibtracs_alignment_confound_max),
    )
    ibtracs_alignment_gate_valid = _build_ibtracs_alignment_gate(
        strict_eval=ibtracs_strict_eval_valid,
        min_event_cases=int(args.ibtracs_alignment_min_event_cases),
        min_far_cases=int(args.ibtracs_alignment_min_far_cases),
        margin_min=float(args.ibtracs_alignment_margin_min),
        confound_max=float(args.ibtracs_alignment_confound_max),
    )
    summary["ibtracs_strict_eval"] = ibtracs_strict_eval
    summary["ibtracs_strict_eval_source"] = ibtracs_strict_eval_source
    summary["ibtracs_strict_eval_valid"] = ibtracs_strict_eval_valid
    summary["ibtracs_selected_time_basis"] = selected_basis
    ibtracs_alignment_gate = _build_ibtracs_alignment_gate(
        strict_eval=ibtracs_eval_selected,
        min_event_cases=int(args.ibtracs_alignment_min_event_cases),
        min_far_cases=int(args.ibtracs_alignment_min_far_cases),
        margin_min=float(args.ibtracs_alignment_margin_min),
        confound_max=float(args.ibtracs_alignment_confound_max),
    )
    hard_reasons = [str(v) for v in (ibtracs_alignment_gate.get("reason_codes") or [])]
    warning_reasons: list[str] = []
    if basis_flip_flag:
        valid_gate_pass = bool((ibtracs_alignment_gate_valid or {}).get("passed", False))
        pinned_valid = bool(
            str(selected_basis).lower() == "valid"
            and str(canonical_time_basis).lower() == "valid"
        )
        if pinned_valid and valid_gate_pass:
            warning_reasons.append("basis_flip_changes_label")
        else:
            hard_reasons.append("basis_flip_changes_label")
    # Recompute pass/reason fields after hard/warning split.
    if hard_reasons:
        hard_counts = pd.Series(hard_reasons, dtype=object).value_counts(dropna=False).to_dict()
        ibtracs_alignment_gate["reason_codes"] = hard_reasons
        ibtracs_alignment_gate["reason_code_counts"] = hard_counts
        ibtracs_alignment_gate["passed"] = False
        actions = [str(v) for v in (ibtracs_alignment_gate.get("actions") or [])]
        if any("basis_flip_changes_label" in r for r in hard_reasons):
            if "compare_source_vs_valid_time_basis_and_lock_one_contract" not in actions:
                actions.append("compare_source_vs_valid_time_basis_and_lock_one_contract")
        ibtracs_alignment_gate["actions"] = actions
        ibtracs_alignment_gate["remediation_hints"] = {
            str(code): [
                "compare_source_vs_valid_time_basis_and_lock_one_contract"
                if "basis_flip_changes_label" in str(code)
                else "increase_strict_overlay_coverage_or_adjust_split"
                if "too_few_ibtracs_" in str(code)
                else "use_valid_time_for_ibtracs_matching"
                if "time_basis_mismatch" in str(code)
                else "normalize_longitudes_to_minus180_180_before_matching"
                if "lon_wrap" in str(code)
                else "audit_distance_units_and_haversine_inputs"
                if "distance_units" in str(code)
                else "calibrate_vortex_center_definition_against_ibtracs"
                if "center_definition" in str(code)
                else "verify_event_vs_far_cohort_definition_and_distance_thresholds"
            ]
            for code in hard_counts.keys()
        }
    ibtracs_alignment_gate["hard_reason_codes"] = [str(v) for v in (ibtracs_alignment_gate.get("reason_codes") or [])]
    ibtracs_alignment_gate["warning_reason_codes"] = warning_reasons
    ibtracs_alignment_gate["selected_time_basis"] = selected_basis
    ibtracs_alignment_gate["source_basis_gate"] = ibtracs_alignment_gate_source
    ibtracs_alignment_gate["valid_basis_gate"] = ibtracs_alignment_gate_valid
    ibtracs_alignment_gate["basis_flip_changes_label"] = bool(basis_flip_flag)
    summary["ibtracs_alignment_gate"] = ibtracs_alignment_gate
    claim_reason_codes: list[str] = []
    if not anomaly_mode_gate.get("passed", False):
        claim_reason_codes.append("anomaly_canonical_fail")
    if not parity_confound_gate.get("passed", False):
        claim_reason_codes.append("mode_weak_margin")
    if not geometry_null_collapse.get("passed", False):
        claim_reason_codes.append("mode_null_fail")
    if not angular_witness.get("gate", {}).get("passed", False):
        claim_reason_codes.append("non_angular_signal")
    if not ibtracs_alignment_gate.get("passed", False):
        claim_reason_codes.append("ibtracs_alignment_fail")
    if bool((summary.get("claim_retention_audit", {}).get("starvation", {}) or {}).get("claim_path_event_starvation", False)):
        claim_reason_codes.append("claim_path_event_starvation")
    if claim_reason_codes:
        claim_mode = "invalid"
    else:
        dis = _to_float(summary.get("anomaly_mode_selection_casewise", {}).get("selection_disagreement_rate"))
        claim_mode = "ensemble" if (dis is not None and dis > 0.10) else "canonical"
    summary["claim_mode"] = claim_mode
    summary["claim_reason_codes"] = claim_reason_codes
    summary["interpretability"] = {
        "diagnostic_only": bool(
            (not anomaly_mode_gate.get("passed", False))
            or (not parity_confound_gate.get("passed", False))
            or (not geometry_null_collapse.get("passed", False))
            or (not angular_witness.get("gate", {}).get("passed", False))
            or (not ibtracs_alignment_gate.get("passed", False))
        ),
        "reasons": [
            *([] if anomaly_mode_gate.get("passed", False) else ["anomaly_canonical_fail"]),
            *([] if parity_confound_gate.get("passed", False) else ["parity_confound_high"]),
            *([] if geometry_null_collapse.get("passed", False) else ["geometry_nulls_do_not_collapse_signal"]),
            *([] if angular_witness.get("gate", {}).get("passed", False) else ["angular_witness_fail"]),
            *([] if ibtracs_alignment_gate.get("passed", False) else ["ibtracs_alignment_fail"]),
            *(
                []
                if not bool((summary.get("claim_retention_audit", {}).get("starvation", {}) or {}).get("claim_path_event_starvation", False))
                else ["claim_path_event_starvation"]
            ),
        ],
    }

    knee_candidate_table = _build_knee_candidate_table(
        samples=test_df,
        modes=[
            "current",
            "raw",
            "anomaly_lat_hour",
            "anomaly_lat_day",
            "scalars_only",
            "local_frame",
            "vorticity",
            "polar_spiral",
            "polar_chiral",
            "polar_odd_ratio",
            "polar_eta",
        ],
    )
    summary["knee_candidate_table_summary"] = {
        "n_rows": int(knee_candidate_table.shape[0]),
        "n_cases": int(knee_candidate_table["case_id"].nunique()) if not knee_candidate_table.empty else 0,
        "modes": sorted(knee_candidate_table["mode"].dropna().astype(str).unique().tolist()) if not knee_candidate_table.empty else [],
    }
    null_knee_detected = bool(any(bool(v.get("knee_detected", False)) for v in (null_controls or {}).values()))
    null_knee_detected_geometry = bool(
        bool((null_controls or {}).get("theta_roll", {}).get("knee_detected", False))
        or bool((null_controls or {}).get("center_swap", {}).get("knee_detected", False))
        or bool((null_controls or {}).get("mirror_axis_jitter", {}).get("knee_detected", False))
        or bool((null_controls or {}).get("radial_scramble", {}).get("knee_detected", False))
        or bool((null_controls or {}).get("center_jitter", {}).get("knee_detected", False))
    )
    knee_weather = _build_weather_knee_acceptance(
        strict_case_metrics=test_case_metrics,
        candidate_table=knee_candidate_table,
        null_knee_detected=null_knee_detected_geometry,
        delta_bic_min=float(args.knee_weather_delta_bic_min),
        resid_improvement_min=float(args.knee_weather_resid_improvement_min),
        slope_delta_min=float(args.knee_weather_slope_delta_min),
        min_consistent_modes=int(args.knee_weather_min_consistent_modes),
        knee_l_rel_tol=float(args.knee_weather_knee_l_rel_tol),
    )
    summary["knee_modes"] = {
        "strict": _build_knee_trace(test_case_metrics),
        "weather": knee_weather["summary"],
        "null_knee_detected_any": bool(null_knee_detected),
        "null_knee_detected_geometry": bool(null_knee_detected_geometry),
    }
    summary["knee_strategy"] = _build_knee_strategy_summary(claim_test_metrics)

    confound_df, confound_summary = _build_parity_confound_dashboard(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        case_meta=case_meta,
        axis_meta=axis_meta,
    )
    summary["parity_confound_dashboard_summary"] = confound_summary
    summary["parity_breakdown"] = _build_parity_breakdown(confound_df, axis_meta=axis_meta)
    far_clean_df, far_clean_summary = _build_far_cleanliness_dashboard(claim_test_df)
    summary["far_cleanliness_summary"] = far_clean_summary
    time_basis_audit = _build_time_basis_audit_from_samples(claim_test_df)
    center_definition_audit = _build_center_definition_audit_from_samples(claim_test_df)
    summary["time_basis_audit"] = time_basis_audit
    summary["center_definition_audit"] = center_definition_audit

    ablation_path = out_path.with_name("parity_ablation.json")
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(json.dumps(parity_ablation, indent=2, sort_keys=True), encoding="utf-8")
    summary["parity_ablation_path"] = str(ablation_path.resolve())
    anomaly_path = out_path.with_name("anomaly_mode_ablation.json")
    anomaly_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_path.write_text(json.dumps(anomaly_ablation, indent=2, sort_keys=True), encoding="utf-8")
    summary["anomaly_mode_ablation_path"] = str(anomaly_path.resolve())
    anomaly_sel_json_path = out_path.with_name("anomaly_mode_selection.json")
    anomaly_sel_json_path.write_text(json.dumps(anomaly_selection_train, indent=2, sort_keys=True), encoding="utf-8")
    summary["anomaly_mode_selection_path"] = str(anomaly_sel_json_path.resolve())
    anomaly_sel_case_path = out_path.with_name("anomaly_mode_selection_casewise.parquet")
    anomaly_selection_casewise["per_case"].to_parquet(anomaly_sel_case_path, index=False)
    summary["anomaly_mode_selection_casewise_path"] = str(anomaly_sel_case_path.resolve())
    anomaly_gate_path = out_path.with_name("anomaly_mode_gate.json")
    anomaly_gate_path.write_text(json.dumps(anomaly_mode_gate, indent=2, sort_keys=True), encoding="utf-8")
    summary["anomaly_mode_gate_path"] = str(anomaly_gate_path.resolve())
    parity_gate_path = out_path.with_name("parity_confound_gate.json")
    parity_gate_path.write_text(json.dumps(parity_confound_gate, indent=2, sort_keys=True), encoding="utf-8")
    summary["parity_confound_gate_path"] = str(parity_gate_path.resolve())
    strict_cov_path = out_path.with_name("strict_test_coverage.json")
    strict_cov_path.write_text(json.dumps(summary.get("strict_test_coverage", {}), indent=2, sort_keys=True), encoding="utf-8")
    summary["strict_test_coverage_path"] = str(strict_cov_path.resolve())
    strict_cov_train_path = out_path.with_name("strict_train_coverage.json")
    strict_cov_train_path.write_text(json.dumps(summary.get("strict_train_coverage", {}), indent=2, sort_keys=True), encoding="utf-8")
    summary["strict_train_coverage_path"] = str(strict_cov_train_path.resolve())
    shard_manifest_path = out_path.with_name("evaluation_shard_manifest.json")
    shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary["evaluation_shard_manifest_path"] = str(shard_manifest_path.resolve())
    geom_audit_path = out_path.with_name("geometry_null_audit.json")
    geom_audit_path.write_text(json.dumps(geometry_null_audit, indent=2, sort_keys=True), encoding="utf-8")
    summary["geometry_null_audit_path"] = str(geom_audit_path.resolve())
    confound_path = out_path.with_name("parity_confound_dashboard.parquet")
    confound_path.parent.mkdir(parents=True, exist_ok=True)
    confound_df.to_parquet(confound_path, index=False)
    summary["parity_confound_dashboard_path"] = str(confound_path.resolve())
    far_clean_path = out_path.with_name("far_cleanliness_dashboard.parquet")
    far_clean_path.parent.mkdir(parents=True, exist_ok=True)
    far_clean_df.to_parquet(far_clean_path, index=False)
    summary["far_cleanliness_dashboard_path"] = str(far_clean_path.resolve())
    far_clean_summary_path = out_path.with_name("far_cleanliness_summary.json")
    far_clean_summary_path.write_text(json.dumps(far_clean_summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["far_cleanliness_summary_path"] = str(far_clean_summary_path.resolve())
    parity_thresholds_path = out_path.with_name("parity_thresholds.json")
    parity_thresholds_path.write_text(
        json.dumps(
            {
                "enabled": bool(args.calibrate_parity_thresholds),
                "applied": summary.get("parity_thresholds_applied", {}),
                "calibration": summary.get("parity_threshold_calibration", {}),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary["parity_thresholds_path"] = str(parity_thresholds_path.resolve())
    parity_feature_path = out_path.with_name("parity_feature_distributions.parquet")
    parity_feature_distributions.to_parquet(parity_feature_path, index=False)
    summary["parity_feature_distributions_path"] = str(parity_feature_path.resolve())
    knee_table_path = out_path.with_name("knee_candidate_table.parquet")
    knee_candidate_table.to_parquet(knee_table_path, index=False)
    summary["knee_candidate_table_path"] = str(knee_table_path.resolve())
    knee_weather_path = out_path.with_name("knee_weather_acceptance.parquet")
    knee_weather["per_case"].to_parquet(knee_weather_path, index=False)
    summary["knee_weather_acceptance_path"] = str(knee_weather_path.resolve())
    ibtracs_strict_path = out_path.with_name("ibtracs_strict_eval.json")
    ibtracs_strict_path.write_text(json.dumps(ibtracs_strict_eval, indent=2, sort_keys=True), encoding="utf-8")
    summary["ibtracs_strict_eval_path"] = str(ibtracs_strict_path.resolve())
    ibtracs_strict_source_path = out_path.with_name("ibtracs_strict_eval_source.json")
    ibtracs_strict_source_path.write_text(json.dumps(ibtracs_strict_eval_source, indent=2, sort_keys=True), encoding="utf-8")
    summary["ibtracs_strict_eval_source_path"] = str(ibtracs_strict_source_path.resolve())
    ibtracs_strict_valid_path = out_path.with_name("ibtracs_strict_eval_valid.json")
    ibtracs_strict_valid_path.write_text(json.dumps(ibtracs_strict_eval_valid, indent=2, sort_keys=True), encoding="utf-8")
    summary["ibtracs_strict_eval_valid_path"] = str(ibtracs_strict_valid_path.resolve())
    ibtracs_alignment_path = out_path.with_name("ibtracs_alignment_gate.json")
    ibtracs_alignment_path.write_text(json.dumps(ibtracs_alignment_gate, indent=2, sort_keys=True), encoding="utf-8")
    summary["ibtracs_alignment_gate_path"] = str(ibtracs_alignment_path.resolve())
    time_basis_audit_path = out_path.with_name("time_basis_audit.json")
    time_basis_audit_path.write_text(json.dumps(time_basis_audit, indent=2, sort_keys=True), encoding="utf-8")
    summary["time_basis_audit_path"] = str(time_basis_audit_path.resolve())
    center_definition_audit_path = out_path.with_name("center_definition_audit.json")
    center_definition_audit_path.write_text(json.dumps(center_definition_audit, indent=2, sort_keys=True), encoding="utf-8")
    summary["center_definition_audit_path"] = str(center_definition_audit_path.resolve())
    claim_retention_audit_path = out_path.with_name("claim_retention_audit.json")
    claim_retention_audit_path.write_text(
        json.dumps(summary.get("claim_retention_audit", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["claim_retention_audit_path"] = str(claim_retention_audit_path.resolve())

    claim_contract = _build_claim_contract(
        summary=summary,
        args=args,
        dataset_dir=dataset_dir,
        config_path=config_path,
        out_path=out_path,
    )
    contract_errors = _validate_claim_contract(claim_contract)
    if contract_errors:
        raise ValueError(f"invalid_claim_contract_schema:{';'.join(contract_errors)}")
    claim_contract_path = out_path.with_name("claim_contract.json")
    claim_contract_path.write_text(json.dumps(claim_contract, indent=2, sort_keys=True), encoding="utf-8")
    if not claim_contract_path.exists():
        raise ValueError("missing_claim_contract")
    summary["claim_contract_path"] = str(claim_contract_path.resolve())
    summary["claim_contract"] = {
        "schema_version": int(claim_contract.get("schema_version", 0)),
        "claim_mode": claim_contract.get("claim_mode"),
        "canonical_anomaly_mode_by_lead": claim_contract.get("canonical_anomaly_mode_by_lead", {}),
    }

    reproduce_path = out_path.with_name("reproduce.sh")
    _write_reproduce_script(
        path=reproduce_path,
        args=args,
        dataset_dir=dataset_dir,
        config_path=config_path,
        out_path=out_path,
        claim_contract=claim_contract,
    )
    summary["reproduce_script_path"] = str(reproduce_path.resolve())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote minipilot evaluation to {out_path}")
    return 0


def _resolve_storm_id_col(samples: pd.DataFrame, requested: str) -> str | None:
    if requested and requested.lower() != "auto":
        return requested if requested in samples.columns else None
    for cand in ("storm_id", "cyclone_id", "sid", "tc_id", "event_id", "track_id", "vortex_track_id"):
        if cand in samples.columns:
            return cand
    return None


def _with_observable_mode(samples: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    out = samples.copy()
    key = str(mode or "current")
    col = _mode_to_column(key)
    if col is None:
        return out
    if col not in out.columns:
        return out
    vals = pd.to_numeric(out[col], errors="coerce")
    fallback = pd.to_numeric(out.get("O"), errors="coerce")
    out["O"] = _sanitize_observable_series(vals, fallback=fallback)
    out["anomaly_mode_claim"] = str(mode)
    return out


def _with_observable_mode_by_lead(
    samples: pd.DataFrame,
    *,
    mode_by_lead: dict[str, str],
    default_mode: str,
) -> pd.DataFrame:
    out = samples.copy()
    if out.empty:
        return out
    if "lead_bucket" not in out.columns:
        return _with_observable_mode(out, mode=default_mode)
    out["lead_bucket"] = out["lead_bucket"].astype(str)
    out["anomaly_mode_claim"] = out["lead_bucket"].map(lambda x: str(mode_by_lead.get(str(x), default_mode)))
    fallback_o = pd.to_numeric(out.get("O"), errors="coerce")
    o_vals = fallback_o.copy()
    for lead, idx in out.groupby("lead_bucket", dropna=False).groups.items():
        lead_key = str(lead)
        mode = str(mode_by_lead.get(lead_key, default_mode))
        col = _mode_to_column(mode)
        if col is None or col not in out.columns:
            col = _mode_to_column(default_mode)
        if col is None or col not in out.columns:
            continue
        vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
        o_vals.loc[idx] = vals
        out.loc[idx, "anomaly_mode_claim"] = mode
    out["O"] = _sanitize_observable_series(o_vals, fallback=fallback_o)
    return out


def _sanitize_observable_series(
    values: pd.Series,
    *,
    fallback: pd.Series | None = None,
) -> pd.Series:
    """Assign claim observable without row-starvation.

    The claim path must be case-first. This helper keeps rows and guarantees a positive
    observable for downstream pipeline calls:
    1) use finite positive selected-mode values,
    2) fallback to prior O if available and positive,
    3) use absolute selected-mode magnitude,
    4) final epsilon floor.
    """
    v = pd.to_numeric(values, errors="coerce").copy()
    if fallback is None:
        fb = pd.Series(np.nan, index=v.index, dtype=float)
    else:
        fb = pd.to_numeric(fallback, errors="coerce").reindex(v.index)

    out = v.copy()
    bad = (~np.isfinite(out)) | (out <= 0.0)
    if bool(bad.any()):
        out.loc[bad] = fb.loc[bad]
    bad = (~np.isfinite(out)) | (out <= 0.0)
    if bool(bad.any()):
        abs_v = np.abs(v)
        out.loc[bad] = abs_v.loc[bad]
    bad = (~np.isfinite(out)) | (out <= 0.0)
    if bool(bad.any()):
        out.loc[bad] = 1e-8
    return pd.to_numeric(out, errors="coerce").fillna(1e-8).astype(float)


def _score_mode_from_samples(
    *,
    samples: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    rank_df = _compute_case_eta_scores(samples)
    if rank_df.empty:
        return {
            "n_cases": 0,
            "n_event_cases": 0,
            "n_far_cases": 0,
            "event_rate": 0.0,
            "far_rate": 0.0,
            "margin": 0.0,
            "null_survival": 1.0,
            "eta_threshold": 0.0,
            "theta_event_rate": 0.0,
            "center_event_rate": 0.0,
        }
    event_mask = _event_mask_from_frame(rank_df)
    far_mask = ~event_mask
    event = rank_df.loc[event_mask].copy()
    far = rank_df.loc[far_mask].copy()
    far_vals = pd.to_numeric(far.get("eta_score"), errors="coerce").to_numpy(dtype=float)
    far_vals = far_vals[np.isfinite(far_vals)]
    if far_vals.size > 0:
        med = float(np.median(far_vals))
        mad = float(np.median(np.abs(far_vals - med)))
        thr = med + max(mad, 1e-8)
    else:
        all_vals = pd.to_numeric(rank_df.get("eta_score"), errors="coerce").to_numpy(dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        thr = float(np.quantile(all_vals, 0.75)) if all_vals.size > 0 else 0.0

    event_scores = pd.to_numeric(event.get("eta_score"), errors="coerce")
    far_scores = pd.to_numeric(far.get("eta_score"), errors="coerce")
    event_rate = float((event_scores >= thr).mean()) if not event_scores.empty else 0.0
    far_rate = float((far_scores >= thr).mean()) if not far_scores.empty else 0.0

    theta_rank = _compute_case_eta_scores(_theta_roll_polar(samples, rng=np.random.default_rng(int(seed) + 1)))
    center_rank = _compute_case_eta_scores(_center_jitter_polar(samples, rng=np.random.default_rng(int(seed) + 2)))

    def _event_rate_for(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        ev = _event_mask_from_frame(df)
        if not bool(ev.any()):
            return 0.0
        score = pd.to_numeric(df.get("eta_score"), errors="coerce")
        return float((score.loc[ev] >= thr).mean())

    theta_event = _event_rate_for(theta_rank)
    center_event = _event_rate_for(center_rank)
    if event_rate > 0.0:
        null_survival = float(0.5 * (theta_event / max(event_rate, 1e-9)) + 0.5 * (center_event / max(event_rate, 1e-9)))
    else:
        null_survival = 1.0
    return {
        "n_cases": int(rank_df.shape[0]),
        "n_event_cases": int(event.shape[0]),
        "n_far_cases": int(far.shape[0]),
        "event_rate": float(event_rate),
        "far_rate": float(far_rate),
        "margin": float(event_rate - far_rate),
        "null_survival": float(null_survival),
        "eta_threshold": float(thr),
        "theta_event_rate": float(theta_event),
        "center_event_rate": float(center_event),
    }


def _select_canonical_anomaly_mode_by_lead(
    *,
    train_df: pd.DataFrame,
    seed: int,
    far_tolerance: float,
) -> dict[str, Any]:
    modes = ("none", "lat_day", "lat_hour")
    weights = {"margin": 1.0, "far": 0.5, "null": 0.5, "complexity": 0.05}
    complexity = {"none": 0.0, "lat_day": 1.0, "lat_hour": 2.0}
    leads = sorted(train_df.get("lead_bucket", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if not leads:
        leads = ["all"]
    per_lead: dict[str, Any] = {}
    canonical: dict[str, str] = {}
    selected_all = True
    robust_all = True
    for i, lead in enumerate(leads):
        base = train_df.copy()
        if lead != "all":
            base = base.loc[base["lead_bucket"].astype(str) == str(lead)].copy()
        mode_rows: dict[str, Any] = {}
        for j, mode in enumerate(modes):
            col = _mode_to_column(mode)
            if col is None or col not in base.columns:
                continue
            dfv = base.copy()
            dfv["O"] = pd.to_numeric(dfv[col], errors="coerce")
            dfv = dfv.loc[np.isfinite(dfv["O"]) & (dfv["O"] > 0)].copy()
            stats = _score_mode_from_samples(samples=dfv, seed=int(seed) + 100 * i + 10 * j)
            score = (
                weights["margin"] * float(stats["margin"])
                - weights["far"] * float(stats["far_rate"])
                - weights["null"] * float(stats["null_survival"])
                - weights["complexity"] * float(complexity.get(mode, 0.0))
            )
            mode_rows[mode] = {
                **stats,
                "mode_complexity": float(complexity.get(mode, 0.0)),
                "objective_score": float(score),
            }
        if not mode_rows:
            per_lead[str(lead)] = {
                "selected_mode": None,
                "selected": False,
                "robust_far": False,
                "modes": {},
            }
            selected_all = False
            robust_all = False
            continue
        ordered = sorted(mode_rows.items(), key=lambda kv: (-float(kv[1]["objective_score"]), -float(kv[1]["margin"]), float(kv[1]["far_rate"]), kv[0]))
        selected_mode = str(ordered[0][0])
        canonical[str(lead)] = selected_mode
        far_values = [float(v["far_rate"]) for v in mode_rows.values()]
        min_far = float(min(far_values)) if far_values else 0.0
        sel_far = float(mode_rows[selected_mode]["far_rate"])
        robust = bool(sel_far <= (min_far + float(far_tolerance)))
        robust_all = robust_all and robust
        per_lead[str(lead)] = {
            "selected_mode": selected_mode,
            "selected": True,
            "robust_far": bool(robust),
            "selected_far_rate": float(sel_far),
            "best_far_rate": float(min_far),
            "far_tolerance": float(far_tolerance),
            "modes": mode_rows,
        }

    # Default to the most frequently selected mode.
    if canonical:
        counts = pd.Series(list(canonical.values())).value_counts()
        default_mode = str(counts.index[0])
    else:
        default_mode = "none"
    summary = {
        "selection_scope": "train_only",
        "weights": {k: float(v) for k, v in weights.items()},
        "far_tolerance": float(far_tolerance),
        "canonical_mode_by_lead": canonical,
        "default_mode": str(default_mode),
        "per_lead": per_lead,
        "selected_all_leads": bool(selected_all and bool(canonical)),
        "robust_far_all_leads": bool(robust_all and bool(canonical)),
        "passed": bool((selected_all and bool(canonical)) and (robust_all and bool(canonical))),
    }
    return summary


def _build_anomaly_mode_gate(
    *,
    anomaly_ablation: dict[str, Any],
    anomaly_selection_summary: dict[str, Any] | None,
    agreement_mean_min: float,
    agreement_min_min: float,
) -> dict[str, Any]:
    dec = (anomaly_ablation or {}).get("decision_stability", {}) or {}
    mean_agree = _to_float(dec.get("agreement_vs_none_mean"))
    min_agree = _to_float(dec.get("agreement_vs_none_min"))
    mean_rank = _to_float(dec.get("rank_agreement_vs_none_mean"))
    min_rank = _to_float(dec.get("rank_agreement_vs_none_min"))
    selection_summary = anomaly_selection_summary or {}
    selected_all = bool(selection_summary.get("selected_all_leads", False))
    robust_far = bool(selection_summary.get("robust_far_all_leads", False))
    passed = bool(selected_all and robust_far)
    return {
        "passed": passed,
        "canonical_mode_selected": bool(selected_all),
        "canonical_mode_robust_far": bool(robust_far),
        "agreement_vs_none_mean": mean_agree,
        "agreement_vs_none_min": min_agree,
        "rank_agreement_vs_none_mean": mean_rank,
        "rank_agreement_vs_none_min": min_rank,
        "pass_decision_agreement_diagnostic": bool(
            (mean_agree is not None and mean_agree >= float(agreement_mean_min))
            and (min_agree is not None and min_agree >= float(agreement_min_min))
        ),
        "thresholds": {
            "agreement_mean_min": float(agreement_mean_min),
            "agreement_min_min": float(agreement_min_min),
        },
        "fallback_mode_if_failed": "scalars_only",
    }


def _select_claim_mode(
    *,
    requested_mode: str,
    parity_ablation: dict[str, Any],
    anomaly_mode_gate: dict[str, Any],
) -> str:
    req = str(requested_mode or "auto")
    if req != "auto":
        return req
    if not bool(anomaly_mode_gate.get("passed", False)):
        return str(anomaly_mode_gate.get("fallback_mode_if_failed", "scalars_only"))
    rec = (parity_ablation or {}).get("recommended_mode_by_contrast", {}) or {}
    mode = rec.get("mode")
    if isinstance(mode, str) and _mode_to_column(mode) is not None:
        variants = (parity_ablation or {}).get("variants", {}) or {}
        v = variants.get(str(mode), {}) if isinstance(variants, dict) else {}
        event_rate = _to_float(v.get("event_parity_signal_rate"))
        margin = _to_float(v.get("event_minus_far_parity_rate"))
        # Avoid degenerate picks when all rates are effectively zero.
        if event_rate is not None and event_rate > 0.0 and margin is not None and margin > 0.0:
            return mode
    return "current"


def _build_parity_confound_gate(
    *,
    strata_test: dict[str, Any],
    confound_max_ratio: float,
    event_minus_far_min: float,
    far_nonstorm_max: float,
    use_alignment_fraction: bool = True,
) -> dict[str, Any]:
    events = (strata_test or {}).get("events", {}) or {}
    far = (strata_test or {}).get("far_nonstorm", {}) or {}
    if bool(use_alignment_fraction):
        align_event = _to_float(events.get("alignment_event_weighted_rate"))
        align_far = _to_float(far.get("alignment_far_weighted_rate"))
        if align_event is None:
            align_event = _to_float(events.get("alignment_fraction_rate"))
        if align_far is None:
            align_far = _to_float(far.get("alignment_fraction_rate"))
        parity_event = _to_float(events.get("parity_signal_rate"))
        parity_far = _to_float(far.get("parity_signal_rate"))
        if (align_event is not None) and (align_far is not None) and (parity_event is not None) and (parity_far is not None):
            # Storm-local alignment should require both intermittent alignment and parity activation.
            event_rate = float((2.0 * float(align_event) * float(parity_event)) / max(float(align_event) + float(parity_event), 1e-9))
            far_rate = float((2.0 * float(align_far) * float(parity_far)) / max(float(align_far) + float(parity_far), 1e-9))
            rate_basis = "alignment_parity_hmean"
        else:
            event_rate = align_event
            far_rate = align_far
            rate_basis = "alignment_fraction"
    else:
        event_rate = None
        far_rate = None
        rate_basis = "parity_signal_rate"
    if event_rate is None:
        event_rate = float(events.get("parity_signal_rate", 0.0))
        rate_basis = "parity_signal_rate_fallback"
    if far_rate is None:
        far_rate = float(far.get("parity_signal_rate", 0.0))
        rate_basis = "parity_signal_rate_fallback"
    event_rate = float(event_rate)
    far_rate = float(far_rate)
    eps = 1e-9
    confound_ratio = float(far_rate / max(event_rate, eps))
    margin = float(event_rate - far_rate)
    passed = bool(
        (event_rate > 0)
        and (confound_ratio < float(confound_max_ratio))
        and (margin >= float(event_minus_far_min))
        and (far_rate <= float(far_nonstorm_max))
    )
    return {
        "passed": passed,
        "event_parity_rate": event_rate,
        "far_nonstorm_parity_rate": far_rate,
        "event_minus_far_rate": margin,
        "confound_rate": confound_ratio,
        "confound_max_ratio": float(confound_max_ratio),
        "event_minus_far_min": float(event_minus_far_min),
        "far_nonstorm_max": float(far_nonstorm_max),
        "rate_basis": str(rate_basis),
    }


def _build_geometry_null_collapse_gate(
    *,
    observed_strata: dict[str, Any],
    null_strata: dict[str, dict[str, Any]],
    min_rel_drop: float,
    min_abs_drop: float,
    rate_key: str = "parity_signal_rate",
    no_signal_max_rate: float = 0.05,
    required_checks: tuple[str, ...] = ("theta_roll", "center_swap", "storm_track_reassignment", "time_permutation_within_lead"),
    center_swap_distance_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = (observed_strata or {}).get("events", {}) or {}
    observed = _to_float(events.get(str(rate_key)))
    if observed is None:
        observed = float(events.get("parity_signal_rate", 0.0))
    observed = float(observed)
    no_signal = bool(observed <= float(no_signal_max_rate))
    rows: dict[str, Any] = {}
    passed = True
    check_names = [str(k) for k in (null_strata or {}).keys()]
    for name in check_names:
        strata = (null_strata or {}).get(name, {}) or {}
        null_events = (strata or {}).get("events", {}) or {}
        null_rate_v = _to_float(null_events.get(str(rate_key)))
        if null_rate_v is None:
            null_rate_v = float(null_events.get("parity_signal_rate", 0.0))
        null_rate = float(null_rate_v)
        abs_drop = float(observed - null_rate)
        rel_drop = float(abs_drop / max(observed, 1e-9))
        if no_signal:
            ok = bool(null_rate <= float(no_signal_max_rate))
            mode = "not_applicable_no_signal"
        else:
            ok = bool((abs_drop >= float(min_abs_drop)) and (rel_drop >= float(min_rel_drop)))
            mode = "signal_collapse"
        required = str(name) in set(str(v) for v in required_checks)
        center_distance_ok = True
        if str(name) == "center_swap":
            dist = center_swap_distance_stats or {}
            p50 = _to_float(dist.get("p50_km"))
            p10 = _to_float(dist.get("p10_km"))
            center_distance_ok = bool((p50 is not None and p50 >= 800.0) and (p10 is not None and p10 >= 500.0))
            if not bool(no_signal):
                ok = bool(ok and center_distance_ok)
        rows[name] = {
            "event_parity_rate_observed": observed,
            "event_parity_rate_null": null_rate,
            "absolute_drop": abs_drop,
            "relative_drop": rel_drop,
            "mode": mode,
            "required": bool(required),
            "passed": ok,
        }
        if str(name) == "center_swap":
            rows[name]["center_swap_distance_stats"] = dict(center_swap_distance_stats or {})
            rows[name]["center_swap_distance_gate"] = {
                "p50_km_min": 800.0,
                "p10_km_min": 500.0,
                "passed": bool(center_distance_ok),
                "applied": bool(not no_signal),
            }
        if required:
            passed = passed and ok
    return {
        "passed": bool(passed),
        "min_relative_drop": float(min_rel_drop),
        "min_absolute_drop": float(min_abs_drop),
        "rate_key": str(rate_key),
        "no_signal_mode": bool(no_signal),
        "no_signal_max_rate": float(no_signal_max_rate),
        "required_checks": [str(v) for v in required_checks],
        "center_swap_distance_stats": dict(center_swap_distance_stats or {}),
        "checks": rows,
    }


def _build_knee_candidate_table(*, samples: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "lead_bucket",
                "mode",
                "rank",
                "n_scales",
                "knee_L",
                "delta_bic",
                "resid_improvement",
                "slope_left",
                "slope_right",
                "slope_delta",
                "sanity_pass_relaxed",
                "rejection_reason",
            ]
        )
    rows: list[dict[str, Any]] = []
    for mode in [str(m) for m in modes]:
        mode_df = _with_observable_mode(samples, mode=mode)
        if mode_df.empty:
            continue
        for case_id, grp in mode_df.groupby("case_id", dropna=False):
            curve = _case_eta_by_scale(grp)
            if curve is None:
                continue
            L = curve["L"]
            eta = curve["eta"]
            x = np.log(np.maximum(L, 1e-12))
            y = np.log(np.maximum(eta, 1e-12))
            n = int(x.size)
            if n < 5:
                continue
            rss0 = _linear_rss(x, y)
            bic0 = _bic(rss0, n=n, k=2)
            cand_rows: list[dict[str, Any]] = []
            for idx in range(2, n - 2):
                rss1, slope_left, slope_right = _piecewise_rss_and_slopes(x, y, idx)
                bic1 = _bic(rss1, n=n, k=3)
                delta_bic = float(bic0 - bic1)
                resid_impr = float((rss0 - rss1) / max(rss0, 1e-12))
                slope_delta = float(abs(slope_right - slope_left))
                slope_ok = bool(abs(slope_left) <= 8.0 and abs(slope_right) <= 8.0)
                rec = {
                    "case_id": str(case_id),
                    "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown",
                    "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else None,
                    "mode": mode,
                    "rank": 0,
                    "n_scales": int(n),
                    "knee_L": float(L[idx]),
                    "delta_bic": delta_bic,
                    "resid_improvement": resid_impr,
                    "slope_left": float(slope_left),
                    "slope_right": float(slope_right),
                    "slope_delta": slope_delta,
                    "sanity_pass_relaxed": bool(slope_ok and resid_impr >= 0.0),
                    "rejection_reason": ";".join(
                        [
                            *([] if slope_ok else ["slope_unphysical"]),
                            *([] if resid_impr >= 0.0 else ["resid_not_improved"]),
                        ]
                    ),
                }
                cand_rows.append(rec)
            if not cand_rows:
                continue
            cand_rows = sorted(cand_rows, key=lambda r: float(r.get("delta_bic", -np.inf)), reverse=True)
            for rank, rec in enumerate(cand_rows, start=1):
                r = dict(rec)
                r["rank"] = int(rank)
                rows.append(r)
    return pd.DataFrame(rows)


def _build_weather_knee_acceptance(
    *,
    strict_case_metrics: pd.DataFrame,
    candidate_table: pd.DataFrame,
    null_knee_detected: bool,
    delta_bic_min: float,
    resid_improvement_min: float,
    slope_delta_min: float,
    min_consistent_modes: int,
    knee_l_rel_tol: float,
) -> dict[str, Any]:
    if strict_case_metrics.empty:
        empty = pd.DataFrame(
            columns=[
                "case_id",
                "strict_detected",
                "weather_candidate_pass",
                "weather_mode_consistent",
                "consistent_mode_count",
                "weather_knee_L",
                "weather_delta_bic",
                "weather_resid_improvement",
                "weather_slope_delta",
                "null_knee_detected",
                "knee_acceptance",
            ]
        )
        return {"summary": {"n_cases": 0, "acceptance_rate": 0.0}, "per_case": empty}

    top = candidate_table.copy()
    if not top.empty:
        top = top.sort_values(["case_id", "mode", "delta_bic"], ascending=[True, True, False]).groupby(
            ["case_id", "mode"], as_index=False
        ).head(1)

    rows: list[dict[str, Any]] = []
    for _, rec in strict_case_metrics.iterrows():
        cid = str(rec.get("case_id"))
        strict_detected = bool(rec.get("knee_detected", False))
        sub = top.loc[top["case_id"].astype(str) == cid].copy() if not top.empty else pd.DataFrame()
        if sub.empty:
            weather_candidate_pass = False
            consistent_mode_count = 0
            mode_consistent = False
            knee_l = None
            delta_bic = None
            resid_impr = None
            slope_delta = None
        else:
            sub["delta_bic"] = pd.to_numeric(sub["delta_bic"], errors="coerce")
            sub["resid_improvement"] = pd.to_numeric(sub["resid_improvement"], errors="coerce")
            sub["slope_delta"] = pd.to_numeric(sub["slope_delta"], errors="coerce")
            sub["knee_L"] = pd.to_numeric(sub["knee_L"], errors="coerce")
            good = sub.loc[
                (sub["delta_bic"] >= float(delta_bic_min))
                & (sub["resid_improvement"] >= float(resid_improvement_min))
                & (sub["slope_delta"] >= float(slope_delta_min))
            ].copy()
            if good.empty:
                weather_candidate_pass = False
                consistent_mode_count = 0
                mode_consistent = False
                knee_l = _to_float(sub["knee_L"].median())
                delta_bic = _to_float(sub["delta_bic"].max())
                resid_impr = _to_float(sub["resid_improvement"].max())
                slope_delta = _to_float(sub["slope_delta"].max())
            else:
                knee_vals = pd.to_numeric(good["knee_L"], errors="coerce").to_numpy(dtype=float)
                knee_vals = knee_vals[np.isfinite(knee_vals) & (knee_vals > 0)]
                if knee_vals.size == 0:
                    consistent_mode_count = 0
                    mode_consistent = False
                    knee_l = None
                else:
                    knee_l = float(np.median(knee_vals))
                    rel = np.abs(knee_vals - knee_l) / max(abs(knee_l), 1e-12)
                    consistent_mode_count = int(np.sum(rel <= float(knee_l_rel_tol)))
                    mode_consistent = bool(consistent_mode_count >= int(min_consistent_modes))
                weather_candidate_pass = bool(mode_consistent)
                delta_bic = _to_float(good["delta_bic"].max())
                resid_impr = _to_float(good["resid_improvement"].max())
                slope_delta = _to_float(good["slope_delta"].max())
        knee_acceptance = bool(strict_detected or (weather_candidate_pass and (not bool(null_knee_detected))))
        rows.append(
            {
                "case_id": cid,
                "strict_detected": bool(strict_detected),
                "weather_candidate_pass": bool(weather_candidate_pass),
                "weather_mode_consistent": bool(mode_consistent),
                "consistent_mode_count": int(consistent_mode_count),
                "weather_knee_L": knee_l,
                "weather_delta_bic": delta_bic,
                "weather_resid_improvement": resid_impr,
                "weather_slope_delta": slope_delta,
                "null_knee_detected": bool(null_knee_detected),
                "knee_acceptance": bool(knee_acceptance),
            }
        )

    per_case = pd.DataFrame(rows)
    return {
        "summary": {
            "n_cases": int(per_case.shape[0]),
            "strict_detected_rate": float(per_case["strict_detected"].mean()) if not per_case.empty else 0.0,
            "weather_candidate_pass_rate": float(per_case["weather_candidate_pass"].mean()) if not per_case.empty else 0.0,
            "knee_acceptance_rate": float(per_case["knee_acceptance"].mean()) if not per_case.empty else 0.0,
            "null_knee_detected": bool(null_knee_detected),
            "thresholds": {
                "delta_bic_min": float(delta_bic_min),
                "resid_improvement_min": float(resid_improvement_min),
                "slope_delta_min": float(slope_delta_min),
                "min_consistent_modes": int(min_consistent_modes),
                "knee_l_rel_tol": float(knee_l_rel_tol),
            },
        },
        "per_case": per_case,
    }


def _build_case_meta(samples: pd.DataFrame, storm_id_col: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, grp in samples.groupby("case_id", dropna=False):
        if grp.empty:
            continue
        if "onset_time" in grp.columns and grp["onset_time"].notna().any():
            anchor = pd.to_datetime(grp["onset_time"], errors="coerce").median()
        else:
            anchor = pd.to_datetime(grp["t"], errors="coerce").median()
        lead_nunique = int(grp["lead_bucket"].nunique(dropna=False)) if "lead_bucket" in grp.columns else 0
        lead_first = str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else None
        case_type = str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown"
        center_source = str(grp["center_source"].iloc[0]) if "center_source" in grp.columns and grp["center_source"].notna().any() else None
        control_tier = str(grp["control_tier"].iloc[0]) if "control_tier" in grp.columns and grp["control_tier"].notna().any() else None
        match_quality = str(grp["match_quality"].iloc[0]) if "match_quality" in grp.columns and grp["match_quality"].notna().any() else None
        storm_distance_cohort = None
        if "storm_distance_cohort" in grp.columns and grp["storm_distance_cohort"].notna().any():
            cohort_series = grp["storm_distance_cohort"].dropna().astype(str)
            if not cohort_series.empty:
                storm_distance_cohort = str(cohort_series.value_counts().idxmax())
        storm_id = None
        if storm_id_col and storm_id_col in grp.columns:
            vals = grp[storm_id_col].dropna()
            if not vals.empty:
                storm_id = str(vals.iloc[0])
        rows.append(
            {
                "case_id": str(case_id),
                "anchor_time": pd.to_datetime(anchor, errors="coerce"),
                "lead_bucket_nunique": lead_nunique,
                "lead_bucket": lead_first,
                "case_type": case_type,
                "center_source": center_source,
                "control_tier": control_tier,
                "match_quality": match_quality,
                "storm_distance_cohort": storm_distance_cohort,
                "storm_max": int(pd.to_numeric(grp.get("storm", 0), errors="coerce").fillna(0).max()),
                "near_storm_max": int(pd.to_numeric(grp.get("near_storm", 0), errors="coerce").fillna(0).max()),
                "pregen_max": int(pd.to_numeric(grp.get("pregen", 0), errors="coerce").fillna(0).max()),
                "ib_event_max": int((pd.to_numeric(grp.get("ib_event", 0), errors="coerce").fillna(0) > 0).any()),
                "ib_far_max": int((pd.to_numeric(grp.get("ib_far", 0), errors="coerce").fillna(0) > 0).any()),
                "ib_event_strict_max": int((pd.to_numeric(grp.get("ib_event_strict", grp.get("ib_event", 0)), errors="coerce").fillna(0) > 0).any()),
                "ib_far_strict_max": int((pd.to_numeric(grp.get("ib_far_strict", grp.get("ib_far", 0)), errors="coerce").fillna(0) > 0).any()),
                "ib_far_quality_tag": (
                    str(grp["ib_far_quality_tag"].dropna().astype(str).value_counts().idxmax())
                    if "ib_far_quality_tag" in grp.columns and grp["ib_far_quality_tag"].notna().any()
                    else None
                ),
                "storm_id": storm_id,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["anchor_time"] = pd.to_datetime(out["anchor_time"], errors="coerce")
    out = out.dropna(subset=["anchor_time"]).reset_index(drop=True)
    out["split_stratum"] = _derive_strata_labels(out)
    return out


def _blocked_time_split(case_meta: pd.DataFrame, split_date: pd.Timestamp, buffer_hours: float) -> dict[str, set[str]]:
    if case_meta.empty:
        return {"train_cases": set(), "test_cases": set(), "buffer_cases": set()}
    buffer = pd.Timedelta(hours=float(buffer_hours))
    train_cut = split_date - buffer
    test_cut = split_date + buffer
    frame = case_meta.copy()
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    frame = frame.dropna(subset=["case_id", "anchor_time"]).copy()
    if frame.empty:
        return {"train_cases": set(), "test_cases": set(), "buffer_cases": set()}

    train_cases: set[str] = set()
    test_cases: set[str] = set()
    buffer_cases: set[str] = set()
    assigned: set[str] = set()

    # Keep storm IDs isolated to a single side of the split; cross-boundary storms go to buffer.
    storm_mask = frame["storm_id"].notna() & (frame["storm_id"].astype(str).str.strip() != "")
    with_storm = frame.loc[storm_mask].copy()
    for _, grp in with_storm.groupby("storm_id", dropna=False):
        case_ids = set(grp["case_id"].astype(str))
        t_min = pd.to_datetime(grp["anchor_time"], errors="coerce").min()
        t_max = pd.to_datetime(grp["anchor_time"], errors="coerce").max()
        assigned.update(case_ids)
        if pd.isna(t_min) or pd.isna(t_max):
            buffer_cases.update(case_ids)
        elif t_max < train_cut:
            train_cases.update(case_ids)
        elif t_min >= test_cut:
            test_cases.update(case_ids)
        else:
            buffer_cases.update(case_ids)

    remaining = frame.loc[~frame["case_id"].astype(str).isin(assigned)].copy()
    if not remaining.empty:
        train_mask = remaining["anchor_time"] < train_cut
        test_mask = remaining["anchor_time"] >= test_cut
        buffer_mask = (~train_mask) & (~test_mask)
        train_cases.update(remaining.loc[train_mask, "case_id"].astype(str))
        test_cases.update(remaining.loc[test_mask, "case_id"].astype(str))
        buffer_cases.update(remaining.loc[buffer_mask, "case_id"].astype(str))
    return {
        "train_cases": train_cases,
        "test_cases": test_cases,
        "buffer_cases": buffer_cases,
    }


def _build_split_audit(
    *,
    case_meta: pd.DataFrame,
    train_cases: set[str],
    test_cases: set[str],
    buffer_cases: set[str],
    split_date: pd.Timestamp,
    buffer_hours: float,
    storm_id_col: str | None,
) -> dict[str, Any]:
    overlap = sorted(train_cases.intersection(test_cases))
    lead_mixed = int((case_meta["lead_bucket_nunique"] > 1).sum()) if "lead_bucket_nunique" in case_meta.columns else 0
    time_buffer_violations, min_gap_hours = _time_buffer_violation_stats(
        case_meta=case_meta,
        train_cases=train_cases,
        test_cases=test_cases,
        buffer_hours=float(buffer_hours),
    )

    storm_id_available = bool(storm_id_col and "storm_id" in case_meta.columns and case_meta["storm_id"].notna().any())
    if storm_id_available:
        train_ids = set(case_meta.loc[case_meta["case_id"].isin(train_cases), "storm_id"].dropna().astype(str))
        test_ids = set(case_meta.loc[case_meta["case_id"].isin(test_cases), "storm_id"].dropna().astype(str))
        storm_overlap = sorted(train_ids.intersection(test_ids))
    else:
        train_ids = set()
        test_ids = set()
        storm_overlap = []

    train_strata = _split_stratum_counts(case_meta=case_meta, case_ids=train_cases)
    test_strata = _split_stratum_counts(case_meta=case_meta, case_ids=test_cases)
    buffer_strata = _split_stratum_counts(case_meta=case_meta, case_ids=buffer_cases)

    return {
        "split_date": str(split_date.date()),
        "time_buffer_hours": float(buffer_hours),
        "n_train_cases": int(len(train_cases)),
        "n_test_cases": int(len(test_cases)),
        "n_buffer_excluded_cases": int(len(buffer_cases)),
        "case_id_overlap_count": int(len(overlap)),
        "case_id_overlap_examples": overlap[:10],
        "lead_mixed_case_count": lead_mixed,
        "time_buffer_violation_count": int(time_buffer_violations),
        "min_train_test_gap_hours": min_gap_hours,
        "storm_id_available": storm_id_available,
        "storm_id_train_count": int(len(train_ids)),
        "storm_id_test_count": int(len(test_ids)),
        "storm_id_overlap_count": int(len(storm_overlap)),
        "storm_id_overlap_examples": storm_overlap[:10],
        "stratum_counts": {
            "train": train_strata.get("counts", {}),
            "test": test_strata.get("counts", {}),
            "buffer": buffer_strata.get("counts", {}),
        },
        "stratum_counts_by_lead": {
            "train": train_strata.get("by_lead", {}),
            "test": test_strata.get("by_lead", {}),
            "buffer": buffer_strata.get("by_lead", {}),
        },
    }


def _split_stratum_counts(*, case_meta: pd.DataFrame, case_ids: set[str]) -> dict[str, Any]:
    if case_meta.empty or not case_ids:
        return {"counts": {"events": 0, "near_storm": 0, "far_nonstorm": 0, "other": 0}, "by_lead": {}}
    sub = case_meta.loc[case_meta["case_id"].astype(str).isin(case_ids)].copy()
    if sub.empty:
        return {"counts": {"events": 0, "near_storm": 0, "far_nonstorm": 0, "other": 0}, "by_lead": {}}
    sub["split_stratum"] = _derive_strata_labels(sub)
    sub["lead_bucket"] = sub["lead_bucket"].astype(str)
    counts = sub["split_stratum"].value_counts(dropna=False).to_dict()
    by_lead = {
        str(lead): {str(k): int(v) for k, v in grp["split_stratum"].value_counts(dropna=False).to_dict().items()}
        for lead, grp in sub.groupby("lead_bucket", dropna=False)
    }
    return {
        "counts": {
            "events": int(counts.get("events", 0)),
            "near_storm": int(counts.get("near_storm", 0)),
            "far_nonstorm": int(counts.get("far_nonstorm", 0)),
            "other": int(counts.get("other", 0)),
        },
        "by_lead": by_lead,
    }


def _time_buffer_violation_stats(
    *,
    case_meta: pd.DataFrame,
    train_cases: set[str],
    test_cases: set[str],
    buffer_hours: float,
) -> tuple[int, float | None]:
    train_times = pd.to_datetime(
        case_meta.loc[case_meta["case_id"].isin(train_cases), "anchor_time"],
        errors="coerce",
    ).dropna()
    test_times = pd.to_datetime(
        case_meta.loc[case_meta["case_id"].isin(test_cases), "anchor_time"],
        errors="coerce",
    ).dropna()
    if train_times.empty or test_times.empty:
        return 0, None

    train_ns = train_times.astype("int64").to_numpy()
    test_ns = np.sort(test_times.astype("int64").to_numpy())
    idx = np.searchsorted(test_ns, train_ns)
    nearest = np.full(train_ns.shape, np.iinfo(np.int64).max, dtype=np.int64)

    right_mask = idx < test_ns.size
    nearest[right_mask] = np.minimum(nearest[right_mask], np.abs(test_ns[idx[right_mask]] - train_ns[right_mask]))
    left_idx = np.clip(idx - 1, 0, test_ns.size - 1)
    nearest = np.minimum(nearest, np.abs(test_ns[left_idx] - train_ns))

    buffer_ns = int(float(buffer_hours) * 3600.0 * 1e9)
    violations = int((nearest < buffer_ns).sum())
    min_gap_hours = float(nearest.min() / 1e9 / 3600.0) if nearest.size > 0 else None
    return violations, min_gap_hours


def _compute_parity_localization_metrics(
    case_df: pd.DataFrame,
    *,
    inner_quantile: float,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "odd_inner": None,
        "odd_outer": None,
        "odd_inner_outer_ratio": None,
        "tangential_radial_ratio": None,
    }
    if case_df is None or case_df.empty:
        return out

    odd_col = None
    for col in ("O_polar_eta", "O_polar_odd_ratio", "O_polar_chiral"):
        if col in case_df.columns:
            odd_col = col
            break
    if odd_col is not None and "L" in case_df.columns:
        lvals = pd.to_numeric(case_df.get("L"), errors="coerce").to_numpy(dtype=float)
        odd_vals = np.abs(pd.to_numeric(case_df.get(odd_col), errors="coerce").to_numpy(dtype=float))
        valid = np.isfinite(lvals) & np.isfinite(odd_vals) & (lvals > 0)
        if np.any(valid):
            lv = lvals[valid]
            ov = odd_vals[valid]
            q = float(np.nanquantile(lv, min(max(float(inner_quantile), 0.05), 0.95)))
            inner = ov[lv <= q]
            outer = ov[lv > q]
            if inner.size > 0 and outer.size > 0:
                odd_inner = float(np.nanmedian(inner))
                odd_outer = float(np.nanmedian(outer))
                out["odd_inner"] = odd_inner
                out["odd_outer"] = odd_outer
                out["odd_inner_outer_ratio"] = float(odd_inner / max(odd_outer, 1e-9))

    if "O_polar_tangential" in case_df.columns and "O_polar_radial" in case_df.columns:
        vt = np.abs(pd.to_numeric(case_df.get("O_polar_tangential"), errors="coerce").to_numpy(dtype=float))
        vr = np.abs(pd.to_numeric(case_df.get("O_polar_radial"), errors="coerce").to_numpy(dtype=float))
        valid = np.isfinite(vt) & np.isfinite(vr)
        if np.any(valid):
            ratio = vt[valid] / (vr[valid] + 1e-9)
            if ratio.size > 0:
                out["tangential_radial_ratio"] = float(np.nanmedian(ratio))
    return out


def _compute_alignment_metrics(
    case_df: pd.DataFrame,
    *,
    eta_threshold: float,
    block_hours: float,
) -> dict[str, float]:
    out = {
        "eventness": 0.0,
        "farness": 0.0,
        "A_align": 0.0,
        "S_align": 0.0,
        "A_align_event": 0.0,
        "A_align_far": 0.0,
        "S_align_event": 0.0,
        "S_align_far": 0.0,
        "n_align_times": 0.0,
        "n_align_blocks": 0.0,
    }
    if case_df is None or case_df.empty:
        return out
    if "hand" not in case_df.columns or "O" not in case_df.columns:
        return out
    time_col = "t" if "t" in case_df.columns else ("t_valid" if "t_valid" in case_df.columns else None)
    if time_col is None:
        return out

    df = case_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["L_scale"] = pd.to_numeric(df.get("L"), errors="coerce")
    df["O"] = pd.to_numeric(df.get("O"), errors="coerce")
    df = df.dropna(subset=[time_col, "L_scale", "O"]).copy()
    if df.empty:
        return out

    pair = df.pivot_table(index=[time_col, "L_scale"], columns="hand", values="O", aggfunc="first").reset_index()
    if ("L" not in pair.columns) or ("R" not in pair.columns):
        # hand columns did not materialize
        return out
    o_l = pd.to_numeric(pair["L"], errors="coerce").to_numpy(dtype=float)
    o_r = pd.to_numeric(pair["R"], errors="coerce").to_numpy(dtype=float)
    denom = 0.5 * (o_l + o_r)
    valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
    eta = np.full(o_l.shape, np.nan, dtype=float)
    eta[valid] = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
    pair["eta_tl"] = eta
    pair = pair.dropna(subset=["eta_tl"]).copy()
    if pair.empty:
        return out

    t_key = time_col
    eta_t = pair.groupby(t_key, as_index=False).agg(eta_t=("eta_tl", "median"), n_scales=("eta_tl", "count"))
    meta = df.groupby(t_key, as_index=False).agg(
        w_event_t=("w_event", "median") if "w_event" in df.columns else ("O", "size"),
        w_far_t=("w_far", "median") if "w_far" in df.columns else ("O", "size"),
    )
    eta_t = eta_t.merge(meta, on=t_key, how="left")
    if "w_event_t" not in eta_t.columns:
        eta_t["w_event_t"] = 0.0
    if "w_far_t" not in eta_t.columns:
        eta_t["w_far_t"] = 0.0
    eta_t["w_event_t"] = pd.to_numeric(eta_t["w_event_t"], errors="coerce").fillna(0.0)
    eta_t["w_far_t"] = pd.to_numeric(eta_t["w_far_t"], errors="coerce").fillna(0.0)

    thr = max(float(eta_threshold), 0.0)
    eta_vals = pd.to_numeric(eta_t["eta_t"], errors="coerce").to_numpy(dtype=float)
    pass_t = (eta_vals >= thr).astype(float)
    event_w = pd.to_numeric(eta_t["w_event_t"], errors="coerce").to_numpy(dtype=float)
    far_w = pd.to_numeric(eta_t["w_far_t"], errors="coerce").to_numpy(dtype=float)

    bh = max(float(block_hours), 0.0)
    n_blocks = 0
    if bh > 0:
        freq = f"{max(int(round(bh)), 1)}h"
        eta_t["time_block"] = pd.to_datetime(eta_t[t_key], errors="coerce").dt.floor(freq)
        n_blocks = int(eta_t["time_block"].nunique(dropna=True))
    out.update(
        {
            "eventness": float(np.nanmean(event_w)) if event_w.size else 0.0,
            "farness": float(np.nanmean(far_w)) if far_w.size else 0.0,
            "A_align": float(np.nanmean(pass_t)) if pass_t.size else 0.0,
            "S_align": float(np.nanmean(eta_vals)) if eta_vals.size else 0.0,
            "A_align_event": _weighted_mean_np(pass_t, event_w),
            "A_align_far": _weighted_mean_np(pass_t, far_w),
            "S_align_event": _weighted_mean_np(eta_vals, event_w),
            "S_align_far": _weighted_mean_np(eta_vals, far_w),
            "n_align_times": float(len(eta_t)),
            "n_align_blocks": float(n_blocks),
        }
    )
    return out


def _weighted_mean_np(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(v[valid] * w[valid]) / max(np.sum(w[valid]), 1e-12))


def _evaluate_cases(
    samples: pd.DataFrame,
    *,
    dataset_dir: Path,
    config_path: Path,
    seed: int,
    enable_time_frequency_knee: bool,
    tf_knee_bic_delta_min: float,
    parity_odd_inner_outer_min: float = 0.0,
    parity_tangential_radial_min: float = 0.0,
    parity_inner_ring_quantile: float = 0.5,
    alignment_eta_threshold: float = 0.10,
    alignment_block_hours: float = 3.0,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "center_source",
                "control_tier",
                "lead_bucket",
                "storm_distance_cohort",
                "nearest_storm_distance_km",
                "storm_max",
                "near_storm_max",
                "pregen_max",
                "ib_event_max",
                "ib_far_max",
                "ib_event_strict_max",
                "ib_far_strict_max",
                "ib_far_quality_tag",
                "split_stratum",
                "knee_detected",
                "knee_size_detected",
                "knee_confidence",
                "P_lock",
                "parity_signal_pass",
                "band_class_hat",
                "R_Omega",
                "ridge_strength",
                "omega_k_hat",
                "knee_rejected_because",
                "knee_delta_bic",
                "knee_strength",
                "knee_p",
                "knee_curvature_peak_ratio",
                "knee_curvature_alignment",
                "knee_candidate_count_proposed",
                "knee_candidate_count_evaluated",
                "knee_candidate_count_sanity_pass",
                "tf_knee_detected",
                "knee_tf_detected",
                "knee_tf_accepted",
                "knee_coincidence",
                "tf_knee_freq_per_hour",
                "tf_knee_delta_bic",
                "tf_knee_reason",
                "tf_knee_consistency_cv",
                "tf_knee_null_gate_pass",
                "knee_detected_effective",
                "parity_localization_pass",
                "parity_tangential_pass",
                "odd_energy_inner",
                "odd_energy_outer",
                "odd_energy_inner_outer_ratio",
                "tangential_radial_ratio",
                "eventness",
                "farness",
                "A_align",
                "S_align",
                "A_align_event",
                "A_align_far",
                "S_align_event",
                "S_align_far",
                "n_align_times",
                "n_align_blocks",
                "run_error",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for i, (case_id, case_df) in enumerate(samples.groupby("case_id", dropna=False)):
        result = _run_partition_pipeline(
            case_df,
            template_dataset=dataset_dir,
            config_path=config_path,
            seed=int(seed + i),
        )
        tf_metrics = _detect_time_frequency_knee(
            case_df,
            bic_delta_min=float(tf_knee_bic_delta_min),
            enabled=bool(enable_time_frequency_knee),
        )
        rejection = str(result.get("knee_rejected_because", ""))
        base_knee = bool(result.get("knee_detected", False))
        tf_null_gate = False
        tf_fallback_accept = False
        if bool(enable_time_frequency_knee) and (not base_knee) and ("no_candidate" in rejection) and bool(tf_metrics.get("detected", False)):
            tf_null_gate = _tf_knee_null_gate(
                case_df=case_df,
                bic_delta_min=float(tf_knee_bic_delta_min),
                seed=int(seed + 10000 + i),
            )
            tf_fallback_accept = bool(tf_null_gate and bool(tf_metrics.get("consistent", False)))
        knee_detected_effective = bool(base_knee or tf_fallback_accept)
        parity_local = _compute_parity_localization_metrics(
            case_df,
            inner_quantile=float(parity_inner_ring_quantile),
        )
        odd_ratio = _to_float(parity_local.get("odd_inner_outer_ratio"))
        tan_ratio = _to_float(parity_local.get("tangential_radial_ratio"))
        parity_localization_pass = bool(
            odd_ratio is not None and odd_ratio >= float(parity_odd_inner_outer_min)
        )
        parity_tangential_pass = bool(
            tan_ratio is not None and tan_ratio >= float(parity_tangential_radial_min)
        )
        parity_signal_pass = bool(result.get("parity_signal_pass", False))
        parity_signal_pass = bool(parity_signal_pass and parity_localization_pass and parity_tangential_pass)
        align = _compute_alignment_metrics(
            case_df,
            eta_threshold=float(alignment_eta_threshold),
            block_hours=float(alignment_block_hours),
        )
        case_row = {
                "case_id": str(case_id),
                "case_type": str(case_df["case_type"].iloc[0]) if "case_type" in case_df.columns else "unknown",
                "center_source": (
                    str(case_df["center_source"].iloc[0])
                    if "center_source" in case_df.columns and case_df["center_source"].notna().any()
                    else None
                ),
                "control_tier": (
                    str(case_df["control_tier"].iloc[0])
                    if "control_tier" in case_df.columns and case_df["control_tier"].notna().any()
                    else None
                ),
                "lead_bucket": str(case_df["lead_bucket"].iloc[0]) if "lead_bucket" in case_df.columns else None,
                "storm_distance_cohort": (
                    str(case_df["storm_distance_cohort"].dropna().astype(str).value_counts().idxmax())
                    if "storm_distance_cohort" in case_df.columns and case_df["storm_distance_cohort"].notna().any()
                    else None
                ),
                "nearest_storm_distance_km": _to_float(pd.to_numeric(case_df.get("nearest_storm_distance_km"), errors="coerce").mean()),
                "storm_max": int(pd.to_numeric(case_df.get("storm", 0), errors="coerce").fillna(0).max()),
                "near_storm_max": int(pd.to_numeric(case_df.get("near_storm", 0), errors="coerce").fillna(0).max()),
                "pregen_max": int(pd.to_numeric(case_df.get("pregen", 0), errors="coerce").fillna(0).max()),
                "ib_event_max": int((pd.to_numeric(case_df.get("ib_event", 0), errors="coerce").fillna(0) > 0).any()),
                "ib_far_max": int((pd.to_numeric(case_df.get("ib_far", 0), errors="coerce").fillna(0) > 0).any()),
                "ib_event_strict_max": int((pd.to_numeric(case_df.get("ib_event_strict", case_df.get("ib_event", 0)), errors="coerce").fillna(0) > 0).any()),
                "ib_far_strict_max": int((pd.to_numeric(case_df.get("ib_far_strict", case_df.get("ib_far", 0)), errors="coerce").fillna(0) > 0).any()),
                "ib_far_quality_tag": (
                    str(case_df["ib_far_quality_tag"].dropna().astype(str).value_counts().idxmax())
                    if "ib_far_quality_tag" in case_df.columns and case_df["ib_far_quality_tag"].notna().any()
                    else None
                ),
                "knee_detected": bool(result.get("knee_detected", False)),
                "knee_size_detected": bool(result.get("knee_detected", False)),
                "knee_confidence": _to_float(result.get("knee_confidence")),
                "P_lock": _to_float(result.get("P_lock")),
                "parity_signal_pass": bool(parity_signal_pass),
                "parity_localization_pass": bool(parity_localization_pass),
                "parity_tangential_pass": bool(parity_tangential_pass),
                "odd_energy_inner": _to_float(parity_local.get("odd_inner")),
                "odd_energy_outer": _to_float(parity_local.get("odd_outer")),
                "odd_energy_inner_outer_ratio": odd_ratio,
                "tangential_radial_ratio": tan_ratio,
                "eventness": _to_float(align.get("eventness")),
                "farness": _to_float(align.get("farness")),
                "A_align": _to_float(align.get("A_align")),
                "S_align": _to_float(align.get("S_align")),
                "A_align_event": _to_float(align.get("A_align_event")),
                "A_align_far": _to_float(align.get("A_align_far")),
                "S_align_event": _to_float(align.get("S_align_event")),
                "S_align_far": _to_float(align.get("S_align_far")),
                "n_align_times": _to_float(align.get("n_align_times")),
                "n_align_blocks": _to_float(align.get("n_align_blocks")),
                "band_class_hat": result.get("band_class_hat"),
                "R_Omega": _to_float(result.get("R_Omega")),
                "ridge_strength": _to_float(result.get("ridge_strength")),
                "omega_k_hat": _to_float(result.get("omega_k_hat")),
                "knee_rejected_because": str(result.get("knee_rejected_because", "")),
                "knee_delta_bic": _to_float(result.get("knee_delta_bic")),
                "knee_strength": _to_float(result.get("knee_strength")),
                "knee_p": _to_float(result.get("knee_p")),
                "knee_curvature_peak_ratio": _to_float(result.get("knee_curvature_peak_ratio")),
                "knee_curvature_alignment": _to_float(result.get("knee_curvature_alignment")),
                "knee_candidate_count_proposed": _to_float(result.get("knee_candidate_count_proposed")),
                "knee_candidate_count_evaluated": _to_float(result.get("knee_candidate_count_evaluated")),
                "knee_candidate_count_sanity_pass": _to_float(result.get("knee_candidate_count_sanity_pass")),
                "tf_knee_detected": bool(tf_metrics.get("detected", False)),
                "knee_tf_detected": bool(tf_metrics.get("detected", False)),
                "knee_tf_accepted": bool(tf_fallback_accept),
                "knee_coincidence": bool(bool(result.get("knee_detected", False)) and bool(tf_metrics.get("detected", False))),
                "tf_knee_freq_per_hour": _to_float(tf_metrics.get("knee_freq_per_hour")),
                "tf_knee_delta_bic": _to_float(tf_metrics.get("delta_bic")),
                "tf_knee_reason": tf_metrics.get("reason"),
                "tf_knee_consistency_cv": _to_float(tf_metrics.get("consistency_cv")),
                "tf_knee_null_gate_pass": bool(tf_null_gate),
                "knee_detected_effective": bool(knee_detected_effective),
                "run_error": result.get("run_error"),
            }
        case_row["split_stratum"] = str(_derive_strata_labels(pd.DataFrame([case_row])).iloc[0])
        out_rows.append(case_row)
    return pd.DataFrame(out_rows)


def _run_partition_pipeline(
    samples: pd.DataFrame,
    *,
    template_dataset: Path,
    config_path: Path,
    seed: int,
) -> dict[str, Any]:
    default_out = {
        "knee_detected": False,
        "knee_confidence": None,
        "P_lock": None,
        "parity_signal_pass": False,
        "band_class_hat": None,
        "R_Omega": None,
        "ridge_strength": None,
        "omega_k_hat": None,
        "knee_rejected_because": "",
        "knee_delta_bic": None,
        "knee_strength": None,
        "knee_p": None,
        "knee_curvature_peak_ratio": None,
        "knee_curvature_alignment": None,
        "knee_candidate_count_proposed": None,
        "knee_candidate_count_evaluated": None,
        "knee_candidate_count_sanity_pass": None,
        "run_error": "empty_partition",
    }
    if samples.empty:
        return default_out
    with tempfile.TemporaryDirectory(prefix="gka_weather_eval_") as tmp:
        tmp_dir = Path(tmp)
        ds_dir = tmp_dir / "dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        spec = yaml.safe_load((template_dataset / "dataset.yaml").read_text(encoding="utf-8"))
        spec["id"] = f"{spec.get('id', 'weather')}_eval"
        (ds_dir / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        samples.to_parquet(ds_dir / "samples.parquet", index=False)

        run_out = tmp_dir / "out"
        try:
            run = run_pipeline(
                dataset_path=str(ds_dir),
                domain=spec.get("domain", "weather"),
                out_dir=str(run_out),
                config_path=str(config_path),
                null_n=0,
                allow_missing=False,
                seed=int(seed),
                dump_intermediates=False,
                argv=["gka", "run", str(ds_dir)],
            )
            row = run.summary.iloc[0].to_dict()
            row["run_error"] = None
            return row
        except ValueError as exc:
            msg = str(exc)
            if "Knee detection requires at least" in msg:
                out = default_out.copy()
                out["run_error"] = "insufficient_scales"
                return out
            out = default_out.copy()
            if "Not enough points for scaling fit after exclusions" in msg:
                out["run_error"] = "insufficient_scaling_points"
            else:
                out["run_error"] = f"value_error:{type(exc).__name__}"
            return out
        except Exception as exc:
            out = default_out.copy()
            out["run_error"] = f"pipeline_error:{type(exc).__name__}"
            return out


def _direction_randomization(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    key_cols = ["case_id", "t", "L"]
    out_frames: list[pd.DataFrame] = []
    for _, grp in df.groupby(key_cols, dropna=False):
        g = grp.copy()
        if g["hand"].nunique() == 2 and rng.random() < 0.5:
            l_mask = g["hand"] == "L"
            r_mask = g["hand"] == "R"
            if l_mask.any() and r_mask.any():
                o_l = float(g.loc[l_mask, "O"].iloc[0])
                o_r = float(g.loc[r_mask, "O"].iloc[0])
                g.loc[l_mask, "O"] = o_r
                g.loc[r_mask, "O"] = o_l
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def _spatial_shuffle_control(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    out = []
    for _, grp in df.groupby(["t", "L", "hand"], dropna=False):
        g = grp.copy()
        vals = g["O"].to_numpy(dtype=float)
        rng.shuffle(vals)
        g["O"] = vals
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _phase_bucket_for_null(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    if "storm_phase" in df.columns:
        s = df["storm_phase"].fillna("").astype(str).str.strip().str.lower()
        if bool((s != "").any()):
            return s.replace("", "unknown")
    if "t_rel_h" in df.columns and "case_id" in df.columns:
        t_rel = pd.to_numeric(df["t_rel_h"], errors="coerce")
        out = pd.Series("unknown", index=df.index, dtype=object)
        for _, grp in df.groupby("case_id", dropna=False):
            idx = grp.index
            vals = pd.to_numeric(grp.get("t_rel_h"), errors="coerce")
            finite = vals[np.isfinite(vals)]
            if finite.shape[0] < 4:
                out.loc[idx] = "unknown"
                continue
            q1 = float(np.nanquantile(finite, 0.25))
            q2 = float(np.nanquantile(finite, 0.50))
            q3 = float(np.nanquantile(finite, 0.75))
            v = vals.to_numpy(dtype=float)
            labels = np.full(v.shape[0], "unknown", dtype=object)
            labels[np.isfinite(v) & (v <= q1)] = "phase_q1"
            labels[np.isfinite(v) & (v > q1) & (v <= q2)] = "phase_q2"
            labels[np.isfinite(v) & (v > q2) & (v <= q3)] = "phase_q3"
            labels[np.isfinite(v) & (v > q3)] = "phase_q4"
            out.loc[idx] = labels
        return out
    return pd.Series("unknown", index=df.index, dtype=object)


def _time_permutation_within_lead(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    if "lead_bucket" not in df.columns:
        df["lead_bucket"] = "none"
    df["_phase_bucket"] = _phase_bucket_for_null(df)
    # Phase-aware permutation: keep lead and phase distributions, break within-track temporal ordering.
    key_cols = [c for c in ("lead_bucket", "_phase_bucket", "L", "hand") if c in df.columns]
    if "case_id" in df.columns and "storm_id" in df.columns:
        # Keep case-local structure from surviving by forcing cross-case shuffles inside the same phase bucket.
        key_cols = [c for c in key_cols if c != "case_id"]
    if not key_cols:
        key_cols = [c for c in ("lead_bucket", "L", "hand") if c in df.columns]
    out = []
    shuffle_cols = [c for c in ("O", "O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio", "O_polar_left", "O_polar_right") if c in df.columns]
    if not shuffle_cols:
        shuffle_cols = ["O"] if "O" in df.columns else []
    for _, grp in df.groupby(key_cols, dropna=False):
        g = grp.copy()
        if g.shape[0] <= 1:
            out.append(g)
            continue
        perm = rng.permutation(g.index.to_numpy())
        for col in shuffle_cols:
            vals = pd.to_numeric(df.loc[perm, col], errors="coerce").to_numpy(dtype=float)
            if vals.size == g.shape[0]:
                if col == "O":
                    vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-8), vals)
                g[col] = vals
        out.append(g)
    out_df = pd.concat(out, ignore_index=True)
    if "_phase_bucket" in out_df.columns:
        out_df = out_df.drop(columns=["_phase_bucket"])
    return out_df


def _lat_band_shuffle_within_month_hour(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    if "t" not in df.columns:
        return _time_permutation_within_lead(samples, rng)
    ts = pd.to_datetime(df["t"], errors="coerce")
    df["_month"] = ts.dt.month.astype("Int64")
    df["_hour"] = ts.dt.hour.astype("Int64")
    lat_src = pd.to_numeric(df.get("lat0", df.get("lat")), errors="coerce")
    if lat_src is None:
        return _time_permutation_within_lead(samples, rng)
    df["_lat_bin"] = (np.floor(lat_src / 2.0) * 2.0).astype(float)
    key_cols = [c for c in ("lead_bucket", "_month", "_hour", "_lat_bin", "L", "hand") if c in df.columns]
    if len(key_cols) < 3:
        return _time_permutation_within_lead(samples, rng)
    shuffle_cols = [c for c in ("O", "O_scalar", "O_raw", "O_vector", "O_local_frame", "O_meanflow", "O_vorticity", "O_lat_hour", "O_lat_day") if c in df.columns]
    shuffle_cols.extend([c for c in df.columns if str(c).startswith("O_polar_") and c not in shuffle_cols])
    out_frames: list[pd.DataFrame] = []
    for _, grp in df.groupby(key_cols, dropna=False):
        g = grp.copy()
        if g.shape[0] <= 1:
            out_frames.append(g)
            continue
        perm = rng.permutation(g.index.to_numpy())
        for col in shuffle_cols:
            src = pd.to_numeric(df.loc[perm, col], errors="coerce").to_numpy(dtype=float)
            if src.size == g.shape[0]:
                g[col] = src
        out_frames.append(g)
    out = pd.concat(out_frames, ignore_index=True)
    for c in ("_month", "_hour", "_lat_bin"):
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def _fake_mirror_pairing(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    out = []
    for _, grp in df.groupby(["t", "L"], dropna=False):
        g = grp.copy()
        r_mask = g["hand"] == "R"
        r_vals = g.loc[r_mask, "O"].to_numpy(dtype=float)
        if r_vals.size > 1:
            rng.shuffle(r_vals)
            g.loc[r_mask, "O"] = r_vals
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _latitude_mirror_pairing(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    if "lat0" not in samples.columns:
        return _fake_mirror_pairing(samples, rng)
    df = samples.copy()
    out = []
    for _, grp in df.groupby(["t", "L"], dropna=False):
        g = grp.copy()
        r_mask = g["hand"] == "R"
        r_rows = g.loc[r_mask].copy()
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        r_rows["lat0"] = pd.to_numeric(r_rows["lat0"], errors="coerce")
        r_rows = r_rows.dropna(subset=["lat0"])
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        src = r_rows.sort_values("lat0")
        vals = src["O"].to_numpy(dtype=float)[::-1]
        tgt_idx = src.index.to_numpy()
        g.loc[tgt_idx, "O"] = vals
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _circular_lon_shift_pairing(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    if "lon0" not in samples.columns:
        return _fake_mirror_pairing(samples, rng)
    df = samples.copy()
    out = []
    for _, grp in df.groupby(["t", "L"], dropna=False):
        g = grp.copy()
        r_mask = g["hand"] == "R"
        r_rows = g.loc[r_mask].copy()
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        r_rows["lon0"] = pd.to_numeric(r_rows["lon0"], errors="coerce")
        r_rows = r_rows.dropna(subset=["lon0"])
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        src = r_rows.sort_values("lon0")
        vals = np.roll(src["O"].to_numpy(dtype=float), shift=1)
        tgt_idx = src.index.to_numpy()
        g.loc[tgt_idx, "O"] = vals
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _mirror_axis_jitter(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    if "lon0" not in samples.columns:
        return _fake_mirror_pairing(samples, rng)
    df = samples.copy()
    out: list[pd.DataFrame] = []
    for _, grp in df.groupby(["t", "L"], dropna=False):
        g = grp.copy()
        r_mask = g["hand"] == "R"
        r_rows = g.loc[r_mask].copy()
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        r_rows["lon0"] = pd.to_numeric(r_rows["lon0"], errors="coerce")
        r_rows = r_rows.dropna(subset=["lon0"])
        if r_rows.shape[0] <= 1:
            out.append(g)
            continue
        src = r_rows.sort_values("lon0")
        vals = src["O"].to_numpy(dtype=float)
        if vals.size <= 1:
            out.append(g)
            continue
        shift_deg = float(rng.uniform(0.5, 2.0))
        shift_steps = int(max(1, round(shift_deg / 0.25)))
        shift_steps = shift_steps % vals.size
        if shift_steps == 0:
            shift_steps = 1
        jittered = np.roll(vals, shift=shift_steps)
        tgt_idx = src.index.to_numpy()
        g.loc[tgt_idx, "O"] = jittered

        # Jitter odd channels to mimic mirror-axis mismatch before parity pairing.
        for col in [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio") if c in g.columns]:
            arr = pd.to_numeric(g.loc[tgt_idx, col], errors="coerce").to_numpy(dtype=float)
            if arr.size > 1:
                arr = np.roll(arr, shift=shift_steps)
                g.loc[tgt_idx, col] = arr
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _theta_roll_polar(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    polar_cols = [
        c
        for c in ("O_polar_left", "O_polar_right", "O_polar_chiral", "O_polar_spiral", "O_polar_eta", "O_polar_odd_ratio")
        if c in samples.columns
    ]
    if not polar_cols:
        # Without polar diagnostics available, fallback to pair-breaking mirror null.
        return _fake_mirror_pairing(samples, rng)

    df = samples.copy()
    out: list[pd.DataFrame] = []
    for _, grp in df.groupby(["case_id", "t", "L"], dropna=False):
        g = grp.copy()
        if g.empty:
            out.append(g)
            continue

        left_idx = g.loc[g["hand"] == "L"].index.to_numpy()
        right_idx = g.loc[g["hand"] == "R"].index.to_numpy()
        n_pairs = int(min(left_idx.size, right_idx.size))

        # Build the transform from paired rows so angular odd/even channels are disrupted upstream.
        if n_pairs > 0 and "O" in g.columns:
            l_vals = pd.to_numeric(g.loc[left_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            r_vals = pd.to_numeric(g.loc[right_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            pair_mu = 0.5 * (l_vals + r_vals)
            base_scale = float(np.nanmedian(np.abs(pair_mu[np.isfinite(pair_mu)]))) if np.isfinite(pair_mu).any() else 0.0
            sigma = max(1e-8, 0.02 * max(base_scale, 1e-6))
            noise = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
            l_new = np.where(np.isfinite(pair_mu), pair_mu + noise, l_vals)
            r_new = np.where(np.isfinite(pair_mu), pair_mu - noise, r_vals)
            g.loc[left_idx[:n_pairs], "O"] = np.maximum(l_new, 1e-8)
            g.loc[right_idx[:n_pairs], "O"] = np.maximum(r_new, 1e-8)

        # Polar handed channels collapse toward even symmetry under theta-roll.
        if n_pairs > 0 and ("O_polar_left" in g.columns and "O_polar_right" in g.columns):
            l_vals = pd.to_numeric(g.loc[left_idx[:n_pairs], "O_polar_left"], errors="coerce").to_numpy(dtype=float)
            r_vals = pd.to_numeric(g.loc[right_idx[:n_pairs], "O_polar_right"], errors="coerce").to_numpy(dtype=float)
            pair_mu = 0.5 * (l_vals + r_vals)
            sigma = max(1e-8, 0.01 * max(float(np.nanmedian(np.abs(pair_mu[np.isfinite(pair_mu)]))) if np.isfinite(pair_mu).any() else 0.0, 1e-6))
            noise = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
            g.loc[left_idx[:n_pairs], "O_polar_left"] = pair_mu + noise
            g.loc[right_idx[:n_pairs], "O_polar_right"] = pair_mu - noise

        odd_cols = [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio") if c in g.columns]
        for col in odd_cols:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if vals.size == 0:
                continue
            atten = float(rng.uniform(0.02, 0.15))
            shift = int(rng.integers(0, max(1, vals.size)))
            rolled = np.roll(vals, shift=shift)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=vals.size)
            g[col] = np.where(np.isfinite(rolled), atten * rolled * signs, rolled)

        # Spiral magnitude remains mostly radial; keep it but de-phase slightly.
        if "O_polar_spiral" in g.columns:
            vals = pd.to_numeric(g["O_polar_spiral"], errors="coerce").to_numpy(dtype=float)
            if vals.size > 1:
                shift = int(rng.integers(1, vals.size))
                rolled = np.roll(vals, shift=shift)
                g["O_polar_spiral"] = np.where(np.isfinite(rolled), 0.8 * rolled, rolled)
        if "O_polar_tangential" in g.columns and "O_polar_radial" in g.columns:
            vt = pd.to_numeric(g["O_polar_tangential"], errors="coerce").to_numpy(dtype=float)
            vr = pd.to_numeric(g["O_polar_radial"], errors="coerce").to_numpy(dtype=float)
            if vt.size > 0 and vr.size > 0:
                mix = 0.5 * (np.nan_to_num(vt, nan=0.0) + np.nan_to_num(vr, nan=0.0))
                noise = rng.normal(loc=0.0, scale=max(1e-8, 0.05 * float(np.nanmedian(np.abs(mix))) if np.isfinite(mix).any() else 1e-8), size=mix.size)
                g["O_polar_tangential"] = np.maximum(mix + noise, 1e-8)
                g["O_polar_radial"] = np.maximum(mix - noise, 1e-8)

        out.append(g)
    return pd.concat(out, ignore_index=True)


def _theta_scramble_within_ring(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    claim_cols = [
        c
        for c in (
            "O",
            "O_polar_chiral",
            "O_polar_eta",
            "O_polar_odd_ratio",
            "O_polar_left",
            "O_polar_right",
            "O_polar_tangential",
            "O_polar_radial",
        )
        if c in samples.columns
    ]
    if not claim_cols:
        return _time_permutation_within_lead(samples, rng)
    df = samples.copy()
    out_frames: list[pd.DataFrame] = []
    group_cols = [c for c in ("case_id", "t", "L") if c in df.columns]
    if not group_cols:
        group_cols = [c for c in ("case_id", "L") if c in df.columns]
    if not group_cols:
        return _time_permutation_within_lead(samples, rng)
    for _, grp in df.groupby(group_cols, dropna=False):
        g = grp.copy()
        left_idx = g.loc[g.get("hand", pd.Series(index=g.index)).astype(str) == "L"].index.to_numpy()
        right_idx = g.loc[g.get("hand", pd.Series(index=g.index)).astype(str) == "R"].index.to_numpy()
        n_pairs = int(min(left_idx.size, right_idx.size))
        if n_pairs <= 0:
            out_frames.append(g)
            continue
        if "O" in g.columns:
            o_l = pd.to_numeric(g.loc[left_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            o_r = pd.to_numeric(g.loc[right_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            pair_mu = 0.5 * (o_l + o_r)
            sigma = max(1e-8, 0.01 * max(float(np.nanmedian(np.abs(pair_mu[np.isfinite(pair_mu)]))) if np.isfinite(pair_mu).any() else 0.0, 1e-6))
            noise = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
            g.loc[left_idx[:n_pairs], "O"] = np.maximum(np.where(np.isfinite(pair_mu), pair_mu + noise, o_l), 1e-8)
            g.loc[right_idx[:n_pairs], "O"] = np.maximum(np.where(np.isfinite(pair_mu), pair_mu - noise, o_r), 1e-8)

        if "O_polar_left" in g.columns and "O_polar_right" in g.columns:
            lvals = pd.to_numeric(g.loc[left_idx[:n_pairs], "O_polar_left"], errors="coerce").to_numpy(dtype=float)
            rvals = pd.to_numeric(g.loc[right_idx[:n_pairs], "O_polar_right"], errors="coerce").to_numpy(dtype=float)
            pair_mu = 0.5 * (lvals + rvals)
            g.loc[left_idx[:n_pairs], "O_polar_left"] = pair_mu
            g.loc[right_idx[:n_pairs], "O_polar_right"] = pair_mu

        odd_cols = [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio") if c in g.columns]
        for col in odd_cols:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if vals.size > 0:
                atten = float(rng.uniform(0.02, 0.12))
                signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=vals.size)
                g[col] = np.where(np.isfinite(vals), atten * vals * signs, vals)

        if "O_polar_tangential" in g.columns and "O_polar_radial" in g.columns:
            vt = pd.to_numeric(g["O_polar_tangential"], errors="coerce").to_numpy(dtype=float)
            vr = pd.to_numeric(g["O_polar_radial"], errors="coerce").to_numpy(dtype=float)
            mix = 0.5 * (np.nan_to_num(vt, nan=0.0) + np.nan_to_num(vr, nan=0.0))
            g["O_polar_tangential"] = np.where(np.isfinite(mix), np.maximum(mix, 1e-8), mix)
            g["O_polar_radial"] = np.where(np.isfinite(mix), np.maximum(mix, 1e-8), mix)

        # Light shuffle of residual channels within ring group.
        for col in [c for c in claim_cols if c not in {"O", "O_polar_left", "O_polar_right", "O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio", "O_polar_tangential", "O_polar_radial"}]:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if vals.size > 1:
                rng.shuffle(vals)
                g[col] = vals
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def _radial_shuffle_polar(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return _radial_scramble_polar(samples, rng)


def _radial_scramble_polar(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    polar_cols = [
        c
        for c in (
            "O_polar_spiral",
            "O_polar_chiral",
            "O_polar_eta",
            "O_polar_odd_ratio",
            "O_polar_left",
            "O_polar_right",
        )
        if c in samples.columns
    ]
    if not polar_cols:
        return _spatial_shuffle_control(samples, rng)
    df = samples.copy()
    out: list[pd.DataFrame] = []
    for _, grp in df.groupby(["case_id", "t", "hand"], dropna=False):
        g = grp.copy()
        for col in polar_cols:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if vals.size > 1:
                rng.shuffle(vals)
                g[col] = vals
        if "O_polar_spiral" in g.columns and "O" in g.columns:
            vals = pd.to_numeric(g["O_polar_spiral"], errors="coerce").to_numpy(dtype=float)
            g["O"] = np.where(np.isfinite(vals), np.maximum(vals, 1e-8), pd.to_numeric(g["O"], errors="coerce").to_numpy(dtype=float))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _center_jitter_polar(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    cols = [c for c in ("O", "O_local_frame", "O_polar_spiral", "O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio", "O_polar_left", "O_polar_right") if c in df.columns]
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        df[col] = vals
    if "O" not in df.columns:
        return df
    out_frames: list[pd.DataFrame] = []
    for _, grp in df.groupby(["case_id", "t", "L"], dropna=False):
        g = grp.copy()
        left_idx = g.loc[g["hand"] == "L"].index.to_numpy()
        right_idx = g.loc[g["hand"] == "R"].index.to_numpy()
        n_pairs = int(min(left_idx.size, right_idx.size))
        if n_pairs <= 0:
            out_frames.append(g)
            continue
        o_l = pd.to_numeric(g.loc[left_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
        o_r = pd.to_numeric(g.loc[right_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
        pair_mu = 0.5 * (o_l + o_r)
        scale = float(np.nanmedian(np.abs(pair_mu[np.isfinite(pair_mu)]))) if np.isfinite(pair_mu).any() else 0.0
        sigma = max(1e-8, 0.01 * max(scale, 1e-6))
        noise = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
        l_new = np.where(np.isfinite(pair_mu), pair_mu + noise, o_l)
        r_new = np.where(np.isfinite(pair_mu), pair_mu - noise, o_r)
        g.loc[left_idx[:n_pairs], "O"] = np.maximum(l_new, 1e-8)
        g.loc[right_idx[:n_pairs], "O"] = np.maximum(r_new, 1e-8)

        if "O_polar_left" in g.columns and "O_polar_right" in g.columns:
            pl = pd.to_numeric(g.loc[left_idx[:n_pairs], "O_polar_left"], errors="coerce").to_numpy(dtype=float)
            pr = pd.to_numeric(g.loc[right_idx[:n_pairs], "O_polar_right"], errors="coerce").to_numpy(dtype=float)
            pmu = 0.5 * (pl + pr)
            ns = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
            g.loc[left_idx[:n_pairs], "O_polar_left"] = pmu + ns
            g.loc[right_idx[:n_pairs], "O_polar_right"] = pmu - ns

        for col in [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio") if c in g.columns]:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            atten = float(rng.uniform(0.01, 0.10))
            signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=vals.size)
            g[col] = np.where(np.isfinite(vals), atten * vals * signs, vals)

        if "O_polar_spiral" in g.columns:
            vals = pd.to_numeric(g["O_polar_spiral"], errors="coerce").to_numpy(dtype=float)
            g["O_polar_spiral"] = np.where(np.isfinite(vals), 0.7 * vals, vals)
        if "O_polar_tangential" in g.columns and "O_polar_radial" in g.columns:
            vt = pd.to_numeric(g["O_polar_tangential"], errors="coerce").to_numpy(dtype=float)
            vr = pd.to_numeric(g["O_polar_radial"], errors="coerce").to_numpy(dtype=float)
            blend = 0.5 * (np.nan_to_num(vt, nan=0.0) + np.nan_to_num(vr, nan=0.0))
            g["O_polar_tangential"] = np.where(np.isfinite(blend), np.maximum(0.8 * blend, 1e-8), blend)
            g["O_polar_radial"] = np.where(np.isfinite(blend), np.maximum(0.8 * blend, 1e-8), blend)

        if "O_local_frame" in g.columns:
            vals = pd.to_numeric(g["O_local_frame"], errors="coerce").to_numpy(dtype=float)
            g["O_local_frame"] = np.where(np.isfinite(vals), 0.5 * vals, vals)
        out_frames.append(g)
    df = pd.concat(out_frames, ignore_index=True)
    return df


def _haversine_km_pair(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not np.isfinite(lat1) or not np.isfinite(lon1) or not np.isfinite(lat2) or not np.isfinite(lon2):
        return np.nan
    r = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(max(a, 0.0))))


def _center_swap_polar(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    required_cols = {"case_id", "lead_bucket", "lat0", "lon0", "hand", "L"}
    if not required_cols.issubset(set(samples.columns)):
        return _spatial_shuffle_control(samples, rng)
    df = samples.copy()
    swap_cols = [
        c
        for c in (
            "O",
            "O_vector",
            "O_scalar",
            "O_raw",
            "O_local_frame",
            "O_vorticity",
            "O_meanflow",
            "O_lat_hour",
            "O_lat_day",
            "O_polar_spiral",
            "O_polar_chiral",
            "O_polar_left",
            "O_polar_right",
            "O_polar_odd_ratio",
            "O_polar_eta",
            "O_polar_tangential",
            "O_polar_radial",
        )
        if c in df.columns
    ]
    if not swap_cols:
        return _spatial_shuffle_control(samples, rng)

    case_meta = []
    for case_id, grp in df.groupby("case_id", dropna=False):
        strata = _derive_strata_labels(grp)
        label = str(strata.value_counts().idxmax()) if not strata.empty else "other"
        case_meta.append(
            {
                "case_id": str(case_id),
                "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else "",
                "lat0": float(pd.to_numeric(grp.get("lat0"), errors="coerce").median()),
                "lon0": float(pd.to_numeric(grp.get("lon0"), errors="coerce").median()),
                "split_stratum": label,
            }
        )
    meta = pd.DataFrame(case_meta)
    if meta.empty:
        return _spatial_shuffle_control(samples, rng)
    events = meta.loc[meta["split_stratum"].isin(["events", "near_storm"])].copy()
    far = meta.loc[meta["split_stratum"] == "far_nonstorm"].copy()
    if events.empty or far.empty:
        return _spatial_shuffle_control(samples, rng)

    mapping: dict[str, dict[str, Any]] = {}
    for _, ev in events.iterrows():
        lead = str(ev["lead_bucket"])
        donors = far.loc[far["lead_bucket"].astype(str) == lead].copy()
        if donors.empty:
            donors = far.copy()
        if donors.empty:
            continue
        donors = donors.assign(
            center_distance_km=[
                _haversine_km_pair(float(ev["lat0"]), float(ev["lon0"]), float(r["lat0"]), float(r["lon0"]))
                for _, r in donors.iterrows()
            ]
        )
        donors = donors.replace([np.inf, -np.inf], np.nan).dropna(subset=["center_distance_km"])
        if donors.empty:
            continue
        strong = donors.loc[donors["center_distance_km"] >= 800.0].copy()
        pick_pool = strong if not strong.empty else donors.sort_values("center_distance_km", ascending=False).head(12)
        if pick_pool.empty:
            continue
        if pick_pool.shape[0] >= 2:
            idxs = rng.choice(pick_pool.index.to_numpy(), size=2, replace=False)
            p_l = pick_pool.loc[idxs[0]]
            p_r = pick_pool.loc[idxs[1]]
        else:
            p = pick_pool.iloc[int(rng.integers(0, pick_pool.shape[0]))]
            p_l = p
            p_r = p
        mapping[str(ev["case_id"])] = {
            "donor_case_l": str(p_l["case_id"]),
            "donor_case_r": str(p_r["case_id"]),
            "dist_l_km": float(p_l["center_distance_km"]),
            "dist_r_km": float(p_r["center_distance_km"]),
            "dist_min_km": float(min(float(p_l["center_distance_km"]), float(p_r["center_distance_km"]))),
            "dist_mean_km": float(0.5 * (float(p_l["center_distance_km"]) + float(p_r["center_distance_km"]))),
        }

    if not mapping:
        return _spatial_shuffle_control(samples, rng)

    out = df.copy()
    out["center_swapped"] = False
    out["center_swap_center_distance_km"] = np.nan
    out["center_swap_center_distance_km_l"] = np.nan
    out["center_swap_center_distance_km_r"] = np.nan
    for event_case, donor_map in mapping.items():
        e_mask = out["case_id"].astype(str) == str(event_case)
        if not bool(e_mask.any()):
            continue
        donor_rows_l = df.loc[df["case_id"].astype(str) == str(donor_map.get("donor_case_l", ""))].copy()
        donor_rows_r = df.loc[df["case_id"].astype(str) == str(donor_map.get("donor_case_r", ""))].copy()
        if donor_rows_l.empty or donor_rows_r.empty:
            continue
        out.loc[e_mask, "center_swapped"] = True
        out.loc[e_mask, "center_swap_center_distance_km"] = float(donor_map.get("dist_mean_km", np.nan))
        out.loc[e_mask, "center_swap_center_distance_km_l"] = float(donor_map.get("dist_l_km", np.nan))
        out.loc[e_mask, "center_swap_center_distance_km_r"] = float(donor_map.get("dist_r_km", np.nan))

        event_rows = out.loc[e_mask].copy()
        for (hand_val, l_val), grp in event_rows.groupby(["hand", "L"], dropna=False):
            idx = grp.index.to_numpy()
            if idx.size == 0:
                continue
            donor_rows = donor_rows_l if str(hand_val) == "L" else donor_rows_r
            dsub = donor_rows.loc[
                (donor_rows["hand"].astype(str) == str(hand_val))
                & (pd.to_numeric(donor_rows["L"], errors="coerce") == float(l_val))
            ].copy()
            if dsub.empty:
                dsub = donor_rows.loc[donor_rows["hand"].astype(str) == str(hand_val)].copy()
            if dsub.empty:
                dsub = donor_rows.copy()
            if dsub.empty:
                continue
            if "t_rel_h" in grp.columns and "t_rel_h" in dsub.columns:
                e_t = pd.to_numeric(grp["t_rel_h"], errors="coerce").to_numpy(dtype=float)
                d_t = pd.to_numeric(dsub["t_rel_h"], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(d_t).any():
                    nearest = []
                    for val in e_t:
                        if not np.isfinite(val):
                            nearest.append(int(rng.integers(0, dsub.shape[0])))
                            continue
                        j = int(np.nanargmin(np.abs(d_t - float(val))))
                        nearest.append(j)
                    src = dsub.iloc[nearest].reset_index(drop=True)
                else:
                    src = dsub.sample(n=idx.size, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
            else:
                src = dsub.sample(n=idx.size, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

            for col in swap_cols:
                vals = pd.to_numeric(src.get(col), errors="coerce").to_numpy(dtype=float)
                if vals.size < idx.size:
                    if vals.size == 0:
                        continue
                    vals = np.resize(vals, idx.size)
                if col == "O":
                    vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-8), vals)
                out.loc[idx, col] = vals[: idx.size]

        # Force geometry break in the parity-carrying channel after swapped-center reparameterization.
        swapped = out.loc[e_mask].copy()
        for _, pair_grp in swapped.groupby(["t", "L"], dropna=False):
            left_idx = pair_grp.loc[pair_grp["hand"].astype(str) == "L"].index.to_numpy()
            right_idx = pair_grp.loc[pair_grp["hand"].astype(str) == "R"].index.to_numpy()
            n_pairs = int(min(left_idx.size, right_idx.size))
            if n_pairs <= 0:
                continue
            o_l = pd.to_numeric(out.loc[left_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            o_r = pd.to_numeric(out.loc[right_idx[:n_pairs], "O"], errors="coerce").to_numpy(dtype=float)
            pair_mu = 0.5 * (o_l + o_r)
            scale = float(np.nanmedian(np.abs(pair_mu[np.isfinite(pair_mu)]))) if np.isfinite(pair_mu).any() else 0.0
            sigma = max(1e-8, 0.005 * max(scale, 1e-6))
            noise = rng.normal(loc=0.0, scale=sigma, size=n_pairs)
            out.loc[left_idx[:n_pairs], "O"] = np.maximum(np.where(np.isfinite(pair_mu), pair_mu + noise, o_l), 1e-8)
            out.loc[right_idx[:n_pairs], "O"] = np.maximum(np.where(np.isfinite(pair_mu), pair_mu - noise, o_r), 1e-8)

            if "O_polar_tangential" in out.columns and "O_polar_radial" in out.columns:
                vt = pd.to_numeric(out.loc[left_idx[:n_pairs], "O_polar_tangential"], errors="coerce").to_numpy(dtype=float)
                vr = pd.to_numeric(out.loc[right_idx[:n_pairs], "O_polar_radial"], errors="coerce").to_numpy(dtype=float)
                mix = 0.5 * (np.nan_to_num(vt, nan=0.0) + np.nan_to_num(vr, nan=0.0))
                out.loc[left_idx[:n_pairs], "O_polar_tangential"] = np.maximum(0.75 * mix, 1e-8)
                out.loc[right_idx[:n_pairs], "O_polar_radial"] = np.maximum(0.75 * mix, 1e-8)

        for col in [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio") if c in out.columns]:
            vals = pd.to_numeric(out.loc[e_mask, col], errors="coerce").to_numpy(dtype=float)
            if vals.size > 0:
                atten = float(rng.uniform(0.05, 0.25))
                signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=vals.size)
                out.loc[e_mask, col] = np.where(np.isfinite(vals), atten * vals * signs, vals)
    return out


def _storm_track_reassignment(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Break storm-phase alignment while preserving broad climatology strata.

    For each event-like case, copy parity-bearing channels from a donor case in the same
    lead/month/hour strata but from a different storm/track group.
    """
    if samples.empty:
        return samples.copy()
    required_cols = {"case_id", "lead_bucket", "hand", "L"}
    if not required_cols.issubset(set(samples.columns)):
        return _time_permutation_within_lead(samples, rng)
    df = samples.copy()
    swap_cols = [
        c
        for c in (
            "O",
            "O_vector",
            "O_scalar",
            "O_raw",
            "O_local_frame",
            "O_vorticity",
            "O_meanflow",
            "O_lat_hour",
            "O_lat_day",
            "O_polar_spiral",
            "O_polar_chiral",
            "O_polar_left",
            "O_polar_right",
            "O_polar_odd_ratio",
            "O_polar_eta",
            "O_polar_tangential",
            "O_polar_radial",
        )
        if c in df.columns
    ]
    if not swap_cols:
        return _time_permutation_within_lead(samples, rng)

    ts = pd.to_datetime(df.get("t"), errors="coerce") if "t" in df.columns else pd.Series(pd.NaT, index=df.index)
    df["_month"] = ts.dt.month.astype("Int64")
    df["_hour"] = ts.dt.hour.astype("Int64")
    track_col = None
    for c in ("vortex_track_id", "storm_id", "track_id"):
        if c in df.columns:
            track_col = c
            break
    if track_col is None:
        df["_track_group"] = df["case_id"].astype(str)
    else:
        df["_track_group"] = (
            df[track_col]
            .fillna(df.get("storm_id", pd.Series(index=df.index)))
            .fillna(df["case_id"])
            .astype(str)
        )
    df["_phase_bucket"] = _phase_bucket_for_null(df)

    case_meta_rows: list[dict[str, Any]] = []
    for case_id, grp in df.groupby("case_id", dropna=False):
        strata = _derive_strata_labels(grp)
        label = str(strata.value_counts().idxmax()) if not strata.empty else "other"
        phase_vals = grp.get("_phase_bucket", pd.Series(index=grp.index, dtype=object)).fillna("unknown").astype(str)
        phase_mode = str(phase_vals.value_counts().idxmax()) if not phase_vals.empty else "unknown"
        case_meta_rows.append(
            {
                "case_id": str(case_id),
                "lead_bucket": str(grp.get("lead_bucket", pd.Series("", index=grp.index)).iloc[0]),
                "month": int(pd.to_numeric(grp.get("_month"), errors="coerce").dropna().iloc[0]) if pd.to_numeric(grp.get("_month"), errors="coerce").dropna().shape[0] > 0 else -1,
                "hour": int(pd.to_numeric(grp.get("_hour"), errors="coerce").dropna().iloc[0]) if pd.to_numeric(grp.get("_hour"), errors="coerce").dropna().shape[0] > 0 else -1,
                "phase_bucket": phase_mode,
                "track_group": str(grp["_track_group"].dropna().iloc[0]) if grp["_track_group"].dropna().shape[0] > 0 else str(case_id),
                "lat0": float(pd.to_numeric(grp.get("lat0"), errors="coerce").median()) if "lat0" in grp.columns else np.nan,
                "lon0": float(pd.to_numeric(grp.get("lon0"), errors="coerce").median()) if "lon0" in grp.columns else np.nan,
                "split_stratum": label,
            }
        )
    case_meta = pd.DataFrame(case_meta_rows)
    if case_meta.empty:
        return _time_permutation_within_lead(samples, rng)

    events = case_meta.loc[case_meta["split_stratum"].isin(["events", "near_storm"])].copy()
    donors_all = case_meta.loc[case_meta["split_stratum"].isin(["events", "near_storm", "far_nonstorm"])].copy()
    if events.empty or donors_all.empty:
        return _time_permutation_within_lead(samples, rng)

    mapping: dict[str, dict[str, Any]] = {}
    for _, ev in events.iterrows():
        lead = str(ev["lead_bucket"])
        track_key = str(ev["track_group"])
        phase_bucket = str(ev.get("phase_bucket", "unknown"))
        donors = donors_all.loc[
            (donors_all["lead_bucket"].astype(str) == lead)
            & (donors_all["case_id"].astype(str) != str(ev["case_id"]))
            & (donors_all["track_group"].astype(str) != track_key)
        ].copy()
        if donors.empty:
            continue
        # Prefer same month/hour strata for climatology matching.
        if int(ev["month"]) >= 0:
            m = donors["month"] == int(ev["month"])
            if bool(m.any()):
                donors = donors.loc[m].copy()
        if int(ev["hour"]) >= 0:
            h = donors["hour"] == int(ev["hour"])
            if bool(h.any()):
                donors = donors.loc[h].copy()
        if "phase_bucket" in donors.columns:
            p = donors["phase_bucket"].astype(str) == phase_bucket
            if bool(p.any()):
                donors = donors.loc[p].copy()
        if donors.empty:
            continue
        donors = donors.assign(
            center_distance_km=[
                _haversine_km_pair(float(ev["lat0"]), float(ev["lon0"]), float(r["lat0"]), float(r["lon0"]))
                for _, r in donors.iterrows()
            ]
        )
        donors = donors.replace([np.inf, -np.inf], np.nan).dropna(subset=["center_distance_km"])
        if donors.empty:
            continue
        strong = donors.loc[donors["center_distance_km"] >= 500.0].copy()
        pick_pool = strong if not strong.empty else donors.sort_values("center_distance_km", ascending=False).head(12)
        if pick_pool.empty:
            continue
        pick = pick_pool.iloc[int(rng.integers(0, pick_pool.shape[0]))]
        mapping[str(ev["case_id"])] = {
            "donor_case": str(pick["case_id"]),
            "donor_track_group": str(pick["track_group"]),
            "center_distance_km": float(pick["center_distance_km"]),
        }

    if not mapping:
        return _time_permutation_within_lead(samples, rng)

    out = df.copy()
    out["storm_track_reassigned"] = False
    out["storm_track_reassign_distance_km"] = np.nan
    out["storm_track_reassign_donor_case"] = None
    out["storm_track_reassign_donor_track_group"] = None

    for case_id, donor in mapping.items():
        e_mask = out["case_id"].astype(str) == str(case_id)
        if not bool(e_mask.any()):
            continue
        donor_rows = df.loc[df["case_id"].astype(str) == str(donor["donor_case"])].copy()
        if donor_rows.empty:
            continue
        out.loc[e_mask, "storm_track_reassigned"] = True
        out.loc[e_mask, "storm_track_reassign_distance_km"] = float(donor.get("center_distance_km", np.nan))
        out.loc[e_mask, "storm_track_reassign_donor_case"] = str(donor.get("donor_case"))
        out.loc[e_mask, "storm_track_reassign_donor_track_group"] = str(donor.get("donor_track_group"))

        event_rows = out.loc[e_mask].copy()
        for (hand_val, l_val), grp in event_rows.groupby(["hand", "L"], dropna=False):
            idx = grp.index.to_numpy()
            if idx.size <= 0:
                continue
            dsub = donor_rows.loc[
                (donor_rows["hand"].astype(str) == str(hand_val))
                & (pd.to_numeric(donor_rows["L"], errors="coerce") == float(l_val))
            ].copy()
            if dsub.empty:
                dsub = donor_rows.loc[donor_rows["hand"].astype(str) == str(hand_val)].copy()
            if dsub.empty:
                dsub = donor_rows.copy()
            if dsub.empty:
                continue
            if "t_rel_h" in grp.columns and "t_rel_h" in dsub.columns:
                e_t = pd.to_numeric(grp["t_rel_h"], errors="coerce").to_numpy(dtype=float)
                d_t = pd.to_numeric(dsub["t_rel_h"], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(d_t).any():
                    nearest = []
                    for val in e_t:
                        if not np.isfinite(val):
                            nearest.append(int(rng.integers(0, dsub.shape[0])))
                            continue
                        j = int(np.nanargmin(np.abs(d_t - float(val))))
                        nearest.append(j)
                    src = dsub.iloc[nearest].reset_index(drop=True)
                else:
                    src = dsub.sample(n=idx.size, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
            else:
                src = dsub.sample(n=idx.size, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

            for col in swap_cols:
                vals = pd.to_numeric(src.get(col), errors="coerce").to_numpy(dtype=float)
                if vals.size < idx.size:
                    if vals.size == 0:
                        continue
                    vals = np.resize(vals, idx.size)
                if col == "O":
                    vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-8), vals)
                out.loc[idx, col] = vals[: idx.size]

        # Additional angular de-phasing to ensure phase coherence is broken.
        for col in [c for c in ("O_polar_chiral", "O_polar_eta", "O_polar_odd_ratio", "O_polar_left", "O_polar_right") if c in out.columns]:
            vals = pd.to_numeric(out.loc[e_mask, col], errors="coerce").to_numpy(dtype=float)
            if vals.size > 1:
                shift = int(rng.integers(1, vals.size))
                rolled = np.roll(vals, shift=shift)
                atten = float(rng.uniform(0.05, 0.35))
                out.loc[e_mask, col] = np.where(np.isfinite(rolled), atten * rolled, rolled)

    return out.drop(columns=[c for c in ("_month", "_hour", "_track_group", "_phase_bucket") if c in out.columns], errors="ignore")


def _geometry_null_transform_by_name(name: str) -> NullTransform | None:
    mapping: dict[str, NullTransform] = {
        "theta_roll": _theta_roll_polar,
        "center_jitter": _center_jitter_polar,
        "center_swap": _center_swap_polar,
        "storm_track_reassignment": _storm_track_reassignment,
        "storm_track_reassignment_phase_matched": _storm_track_reassignment,
        "time_permutation_within_lead": _time_permutation_within_lead,
        "time_permutation_within_track_phase": _time_permutation_within_lead,
        "lat_band_shuffle_within_month_hour": _lat_band_shuffle_within_month_hour,
        "mirror_axis_jitter": _mirror_axis_jitter,
        "radial_scramble": _radial_scramble_polar,
        "radial_shuffle": _radial_shuffle_polar,
        "theta_scramble_within_ring": _theta_scramble_within_ring,
    }
    return mapping.get(str(name))


def _summarize_case_metrics_by_type(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if case_metrics is None or case_metrics.empty:
        return out
    if "case_type" not in case_metrics.columns:
        return {"unknown": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold)}
    for case_type, grp in case_metrics.groupby("case_type"):
        out[str(case_type)] = _aggregate_case_metrics(grp, p_lock_threshold=p_lock_threshold)
    return out


def _summarize_case_metrics_by_control_tier(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    if case_metrics.empty:
        return {}
    controls = case_metrics.loc[case_metrics["case_type"].astype(str) == "control"].copy()
    if controls.empty:
        return {}
    controls["control_tier"] = controls.get("control_tier", pd.Series(index=controls.index)).fillna("unknown").astype(str)
    out: dict[str, Any] = {}
    for tier, grp in controls.groupby("control_tier", dropna=False):
        out[str(tier)] = _aggregate_case_metrics(grp, p_lock_threshold=p_lock_threshold)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _summarize_case_metrics_by_center_source(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    if case_metrics.empty:
        return {}
    src = case_metrics.copy()
    src["center_source"] = src.get("center_source", pd.Series(index=src.index)).fillna("unknown").astype(str)
    out: dict[str, Any] = {}
    for source, grp in src.groupby("center_source", dropna=False):
        out[str(source)] = _aggregate_case_metrics(grp, p_lock_threshold=p_lock_threshold)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _summarize_case_metrics_by_strata(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    if case_metrics.empty:
        return {
            "all": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold),
            "events": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold),
            "near_storm_only": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold),
            "far_nonstorm": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold),
        }
    strata = _derive_strata_labels(case_metrics)
    events = case_metrics.loc[strata.isin(["events", "near_storm"])].copy()
    near_only = case_metrics.loc[strata == "near_storm"].copy()
    far_nonstorm = case_metrics.loc[strata == "far_nonstorm"].copy()
    return {
        "all": _aggregate_case_metrics(case_metrics, p_lock_threshold=p_lock_threshold),
        "events": _aggregate_case_metrics(events, p_lock_threshold=p_lock_threshold),
        "near_storm_only": _aggregate_case_metrics(near_only, p_lock_threshold=p_lock_threshold),
        "far_nonstorm": _aggregate_case_metrics(far_nonstorm, p_lock_threshold=p_lock_threshold),
    }


def _aggregate_case_metrics(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    if case_metrics.empty:
        return {
            "n_cases": 0,
            "knee_rate": 0.0,
            "parity_lock_rate": 0.0,
            "parity_signal_rate": 0.0,
            "parity_signal_rate_raw": 0.0,
            "alignment_fraction_rate": 0.0,
            "alignment_event_weighted_rate": 0.0,
            "alignment_far_weighted_rate": 0.0,
            "knee_confidence_mean": 0.0,
            "tf_knee_rate": 0.0,
            "run_error_rate": 0.0,
        }
    knee_series = (
        case_metrics["knee_detected_effective"]
        if "knee_detected_effective" in case_metrics.columns
        else case_metrics.get("knee_detected", pd.Series(dtype=bool))
    )
    parity_series_raw = (
        case_metrics["parity_signal_pass"]
        if "parity_signal_pass" in case_metrics.columns
        else pd.Series(dtype=bool)
    )
    parity_series_eff = (
        case_metrics["parity_signal_pass_effective"]
        if "parity_signal_pass_effective" in case_metrics.columns
        else parity_series_raw
    )
    return {
        "n_cases": int(case_metrics.shape[0]),
        "knee_rate": float(knee_series.mean()) if len(knee_series) else 0.0,
        "parity_lock_rate": float(
            (pd.to_numeric(case_metrics["P_lock"], errors="coerce") >= float(p_lock_threshold)).mean()
        ),
        "parity_signal_rate": float(parity_series_eff.mean()) if len(parity_series_eff) else 0.0,
        "parity_signal_rate_raw": float(parity_series_raw.mean()) if len(parity_series_raw) else 0.0,
        "alignment_fraction_rate": float(pd.to_numeric(case_metrics.get("A_align"), errors="coerce").mean())
        if "A_align" in case_metrics
        else 0.0,
        "alignment_event_weighted_rate": float(pd.to_numeric(case_metrics.get("A_align_event"), errors="coerce").mean())
        if "A_align_event" in case_metrics
        else 0.0,
        "alignment_far_weighted_rate": float(pd.to_numeric(case_metrics.get("A_align_far"), errors="coerce").mean())
        if "A_align_far" in case_metrics
        else 0.0,
        "parity_localization_pass_rate": float(pd.to_numeric(case_metrics.get("parity_localization_pass"), errors="coerce").fillna(0).mean())
        if "parity_localization_pass" in case_metrics
        else 0.0,
        "parity_tangential_pass_rate": float(pd.to_numeric(case_metrics.get("parity_tangential_pass"), errors="coerce").fillna(0).mean())
        if "parity_tangential_pass" in case_metrics
        else 0.0,
        "odd_energy_inner_outer_ratio_median": _safe_median(case_metrics.get("odd_energy_inner_outer_ratio")),
        "tangential_radial_ratio_median": _safe_median(case_metrics.get("tangential_radial_ratio")),
        "knee_confidence_mean": float(
            pd.to_numeric(case_metrics.get("knee_confidence", pd.Series(dtype=float)), errors="coerce").mean()
        ),
        "tf_knee_rate": float(case_metrics["tf_knee_detected"].mean()) if "tf_knee_detected" in case_metrics else 0.0,
        "run_error_rate": float(case_metrics["run_error"].notna().mean()) if "run_error" in case_metrics else 0.0,
    }


def _build_knee_trace(case_metrics: pd.DataFrame) -> dict[str, Any]:
    if case_metrics.empty:
        return {
            "n_cases": 0,
            "knee_detected_rate": 0.0,
            "no_candidate_count": 0,
            "rejection_reason_counts": {},
            "candidate_summary": {},
        }
    reasons = (
        case_metrics["knee_rejected_because"].fillna("").astype(str)
        if "knee_rejected_because" in case_metrics.columns
        else pd.Series(dtype=str)
    )
    reason_counts: dict[str, int] = {}
    no_candidate_count = 0
    for entry in reasons.tolist():
        if not entry:
            continue
        tokens = [t.strip() for t in entry.split(";") if t.strip()]
        if "no_candidate" in tokens:
            no_candidate_count += 1
        for token in tokens:
            reason_counts[token] = reason_counts.get(token, 0) + 1

    candidate = case_metrics.copy()
    candidate = candidate.loc[candidate["run_error"].isna()] if "run_error" in candidate.columns else candidate
    candidate_summary = {
        "candidate_count_proposed_median": _safe_median(candidate.get("knee_candidate_count_proposed")),
        "candidate_count_evaluated_median": _safe_median(candidate.get("knee_candidate_count_evaluated")),
        "candidate_count_sanity_pass_median": _safe_median(candidate.get("knee_candidate_count_sanity_pass")),
        "knee_delta_bic_median": _safe_median(candidate.get("knee_delta_bic")),
        "knee_strength_median": _safe_median(candidate.get("knee_strength")),
        "knee_p_median": _safe_median(candidate.get("knee_p")),
        "curvature_peak_ratio_median": _safe_median(candidate.get("knee_curvature_peak_ratio")),
        "curvature_alignment_median": _safe_median(candidate.get("knee_curvature_alignment")),
    }
    top_cols = [
        c
        for c in (
            "case_id",
            "knee_delta_bic",
            "knee_strength",
            "knee_p",
            "knee_rejected_because",
            "knee_candidate_count_proposed",
            "knee_candidate_count_evaluated",
            "knee_candidate_count_sanity_pass",
        )
        if c in candidate.columns
    ]
    top_candidates: list[dict[str, Any]] = []
    if top_cols:
        rank = candidate[top_cols].copy()
        rank["knee_delta_bic"] = pd.to_numeric(rank.get("knee_delta_bic"), errors="coerce")
        rank = rank.sort_values("knee_delta_bic", ascending=False).head(3)
        top_candidates = rank.to_dict(orient="records")
    return {
        "n_cases": int(case_metrics.shape[0]),
        "knee_detected_rate": float(case_metrics["knee_detected"].mean()) if "knee_detected" in case_metrics else 0.0,
        "knee_detected_effective_rate": float(case_metrics["knee_detected_effective"].mean())
        if "knee_detected_effective" in case_metrics
        else float(case_metrics["knee_detected"].mean()) if "knee_detected" in case_metrics else 0.0,
        "no_candidate_count": int(no_candidate_count),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_summary": candidate_summary,
        "top_candidates": top_candidates,
    }


def _build_slowtick_report(
    *,
    test_df: pd.DataFrame,
    template_dataset: Path,
    config_path: Path,
    null_transforms: dict[str, NullTransform],
    seed: int,
    delta_min: float,
    p_max: float,
) -> dict[str, Any]:
    if test_df.empty:
        return {"per_lead": {}, "notes": "No test rows available"}

    per_lead: dict[str, Any] = {}
    leads = sorted(test_df["lead_bucket"].astype(str).dropna().unique().tolist())
    for i, lead in enumerate(leads):
        lead_df = test_df.loc[test_df["lead_bucket"].astype(str) == lead].copy()
        strata = _sample_strata(lead_df)
        stratum_rows: dict[str, Any] = {}
        for s_idx, (stratum_name, stratum_df) in enumerate(strata.items()):
            obs = _run_partition_pipeline(
                stratum_df,
                template_dataset=template_dataset,
                config_path=config_path,
                seed=int(seed + i * 100 + s_idx),
            )
            null_rows: dict[str, dict[str, Any]] = {}
            null_r_omega: dict[str, float | None] = {}
            for j, (name, transform) in enumerate(null_transforms.items()):
                transformed = transform(
                    stratum_df,
                    rng=np.random.default_rng(int(seed) + 100 + i * 131 + s_idx * 17 + j),
                )
                row = _run_partition_pipeline(
                    transformed,
                    template_dataset=template_dataset,
                    config_path=config_path,
                    seed=int(seed + 1000 + i * 131 + s_idx * 17 + j),
                )
                null_rows[name] = row
                null_r_omega[name] = _to_float(row.get("R_Omega"))

            obs_r = _to_float(obs.get("R_Omega"))
            null_vals = np.asarray([v for v in null_r_omega.values() if v is not None], dtype=float)
            if obs_r is not None and null_vals.size > 0:
                null_median = float(np.median(null_vals))
                delta = float(obs_r - null_median)
                p_emp = float((1 + np.sum(null_vals >= obs_r)) / (null_vals.size + 1))
                survives = bool(delta >= float(delta_min) and p_emp <= float(p_max))
                z_effect = float(delta / (np.std(null_vals, ddof=0) + 1e-12))
            else:
                null_median = None
                delta = None
                p_emp = None
                survives = False
                z_effect = None

            stratum_rows[str(stratum_name)] = {
                "n_rows": int(stratum_df.shape[0]),
                "dominant_omega_hat": _to_float(obs.get("omega_k_hat")),
                "coherence_score": obs_r,
                "ridge_strength": _to_float(obs.get("ridge_strength")),
                "band_class_hat": obs.get("band_class_hat"),
                "null_R_Omega": null_r_omega,
                "null_median_R_Omega": null_median,
                "delta_R_Omega": delta,
                "delta_z": z_effect,
                "p_empirical": p_emp,
                "survives_nulls": survives,
                "run_error": obs.get("run_error"),
            }

        per_lead[str(lead)] = {
            "n_rows": int(lead_df.shape[0]),
            "strata": stratum_rows,
        }
    return {
        "criteria": {"delta_min": float(delta_min), "p_max": float(p_max)},
        "per_lead": per_lead,
    }


def _sample_strata(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if samples.empty:
        return {"all": samples.copy()}
    strata = _derive_strata_labels(samples)
    out = {
        "all": samples.copy(),
        "events": samples.loc[strata.isin(["events", "near_storm"])].copy(),
        "near_storm_only": samples.loc[strata == "near_storm"].copy(),
        "far_nonstorm": samples.loc[strata == "far_nonstorm"].copy(),
    }
    return out


def _test_case_coverage(*, case_meta: pd.DataFrame, test_cases: set[str]) -> dict[str, Any]:
    if case_meta.empty or not test_cases:
        return {
            "total_test_cases": 0,
            "test_leads": [],
            "events": {"total": 0, "by_lead": {}},
            "far_nonstorm": {"total": 0, "by_lead": {}},
        }
    test = case_meta.loc[case_meta["case_id"].astype(str).isin(test_cases)].copy()
    if test.empty:
        return {
            "total_test_cases": 0,
            "test_leads": [],
            "events": {"total": 0, "by_lead": {}},
            "far_nonstorm": {"total": 0, "by_lead": {}},
        }
    test["lead_bucket"] = test["lead_bucket"].astype(str)
    test_leads = sorted(test["lead_bucket"].dropna().astype(str).unique().tolist())
    strata = _derive_strata_labels(test)
    event_mask = strata.isin(["events", "near_storm"])
    far_mask = strata == "far_nonstorm"
    events = test.loc[event_mask].copy()
    far = test.loc[far_mask].copy()
    event_by_lead = events.groupby("lead_bucket")["case_id"].nunique().to_dict()
    far_by_lead = far.groupby("lead_bucket")["case_id"].nunique().to_dict()
    return {
        "total_test_cases": int(test["case_id"].nunique()),
        "test_leads": test_leads,
        "events": {
            "total": int(events["case_id"].nunique()),
            "by_lead": {str(k): int(v) for k, v in sorted(event_by_lead.items())},
        },
        "far_nonstorm": {
            "total": int(far["case_id"].nunique()),
            "by_lead": {str(k): int(v) for k, v in sorted(far_by_lead.items())},
        },
    }


def _strict_ib_coverage(
    *,
    samples: pd.DataFrame,
    case_ids: set[str],
    time_basis: str,
    use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> dict[str, Any]:
    return contract_strict_ib_coverage(
        samples=samples,
        case_ids=case_ids,
        time_basis=str(time_basis),
        use_flags=bool(use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )


def _strict_ib_case_flags(
    *,
    samples: pd.DataFrame,
    time_basis: str,
    use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> pd.DataFrame:
    return contract_strict_ib_case_flags(
        samples=samples,
        time_basis=str(time_basis),
        use_flags=bool(use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )


def _strict_coverage_ok(
    *,
    coverage: dict[str, Any],
    min_event_total: int,
    min_far_total: int,
    min_event_per_lead: int,
    min_far_per_lead: int,
) -> tuple[bool, list[str]]:
    return contract_strict_coverage_ok(
        coverage=coverage,
        min_event_total=int(min_event_total),
        min_far_total=int(min_far_total),
        min_event_per_lead=int(min_event_per_lead),
        min_far_per_lead=int(min_far_per_lead),
    )


def _build_evaluation_shard_manifest(
    *,
    samples: pd.DataFrame,
    split_date: pd.Timestamp,
    time_buffer_hours: float,
    case_shards: int,
    max_cases: int,
    stratify_by: tuple[str, ...],
    storm_id_col: str | None,
    strict_time_basis: str,
    strict_use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_event_test_total: int,
    min_far_test_total: int,
    min_event_test_per_lead: int,
    min_far_test_per_lead: int,
    min_event_train_total: int,
    min_far_train_total: int,
    min_event_train_per_lead: int,
    min_far_train_per_lead: int,
    min_train_cases: int,
    min_test_cases: int,
    current_shard_index: int,
    strict_far_min_row_quality_frac: float = 0.0,
    strict_far_min_kinematic_clean_frac: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> dict[str, Any]:
    n_shards = int(max(1, case_shards))
    rows: list[dict[str, Any]] = []
    for shard_idx in range(1, n_shards + 1):
        shard_df, info = _select_case_subset(
            samples,
            max_cases=int(max_cases),
            case_shards=n_shards,
            case_shard_index=int(shard_idx),
            stratify_by=tuple(stratify_by),
            storm_id_col=storm_id_col,
            split_date=pd.Timestamp(split_date),
            time_buffer_hours=float(time_buffer_hours),
            strict_time_basis=str(strict_time_basis),
            strict_use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(far_quality_tags),
            min_event_total=int(min_event_test_total),
            min_far_total=int(min_far_test_total),
            min_event_train_total=int(min_event_train_total),
            min_far_train_total=int(min_far_train_total),
            min_train=int(min_train_cases),
            min_test=int(min_test_cases),
            strict_far_min_row_quality_frac=float(strict_far_min_row_quality_frac),
            strict_far_min_kinematic_clean_frac=float(strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
        )
        case_meta = _build_case_meta(shard_df, storm_id_col=storm_id_col)
        split = _blocked_time_split(case_meta, split_date=split_date, buffer_hours=float(time_buffer_hours))
        train_cases = set(split.get("train_cases", set()))
        test_cases = set(split.get("test_cases", set()))
        strict_test_cov = _strict_ib_coverage(
            samples=shard_df,
            case_ids=test_cases,
            time_basis=str(strict_time_basis),
            use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(far_quality_tags),
            min_far_quality_fraction=float(strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
        )
        strict_train_cov = _strict_ib_coverage(
            samples=shard_df,
            case_ids=train_cases,
            time_basis=str(strict_time_basis),
            use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(far_quality_tags),
            min_far_quality_fraction=float(strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
        )
        test_ok, test_reasons = _strict_coverage_ok(
            coverage=strict_test_cov,
            min_event_total=int(min_event_test_total),
            min_far_total=int(min_far_test_total),
            min_event_per_lead=int(min_event_test_per_lead),
            min_far_per_lead=int(min_far_test_per_lead),
        )
        train_ok, train_reasons = _strict_coverage_ok(
            coverage=strict_train_cov,
            min_event_total=int(min_event_train_total),
            min_far_total=int(min_far_train_total),
            min_event_per_lead=int(min_event_train_per_lead),
            min_far_per_lead=int(min_far_train_per_lead),
        )
        split_ok = bool(
            (int(len(train_cases)) >= int(min_train_cases))
            and (int(len(test_cases)) >= int(min_test_cases))
        )
        reasons = [f"test:{v}" for v in test_reasons] + [f"train:{v}" for v in train_reasons]
        if int(len(train_cases)) < int(min_train_cases):
            reasons.append(f"split:train_cases<{int(min_train_cases)}")
        if int(len(test_cases)) < int(min_test_cases):
            reasons.append(f"split:test_cases<{int(min_test_cases)}")
        ok = bool(test_ok and train_ok and split_ok)
        rows.append(
            {
                "shard_index": int(shard_idx),
                "n_cases_selected": int(info.get("n_cases_selected", 0)),
                "n_rows_selected": int(info.get("n_rows_selected", 0)),
                "n_train_cases": int(len(train_cases)),
                "n_test_cases": int(len(test_cases)),
                "strict_event_train_cases": int(strict_train_cov.get("event_total", 0)),
                "strict_far_train_cases": int(strict_train_cov.get("far_total", 0)),
                "strict_event_train_by_lead": strict_train_cov.get("event_by_lead", {}),
                "strict_far_train_by_lead": strict_train_cov.get("far_by_lead", {}),
                "strict_event_test_cases": int(strict_test_cov.get("event_total", 0)),
                "strict_far_test_cases": int(strict_test_cov.get("far_total", 0)),
                "strict_event_test_by_lead": strict_test_cov.get("event_by_lead", {}),
                "strict_far_test_by_lead": strict_test_cov.get("far_by_lead", {}),
                "claim_coverage_ok": bool(ok),
                "coverage_reasons": [str(v) for v in reasons],
            }
        )
    current = next((r for r in rows if int(r.get("shard_index", 0)) == int(current_shard_index)), None)
    return {
        "enabled": bool(n_shards > 1),
        "case_shards": int(n_shards),
        "current_shard_index": int(current_shard_index),
        "current_shard_claim_coverage_ok": bool((current or {}).get("claim_coverage_ok", True)),
        "current_shard_coverage_reasons": [str(v) for v in ((current or {}).get("coverage_reasons", []) or [])],
        "max_cases": int(max_cases),
        "strict_time_basis": str(strict_time_basis),
        "strict_use_flags": bool(strict_use_flags),
        "far_quality_tags": [str(v) for v in (far_quality_tags or ())],
        "thresholds": {
            "min_event_test_total": int(min_event_test_total),
            "min_far_test_total": int(min_far_test_total),
            "min_event_test_per_lead": int(min_event_test_per_lead),
            "min_far_test_per_lead": int(min_far_test_per_lead),
            "min_event_train_total": int(min_event_train_total),
            "min_far_train_total": int(min_far_train_total),
            "min_event_train_per_lead": int(min_event_train_per_lead),
            "min_far_train_per_lead": int(min_far_train_per_lead),
            "min_train_cases": int(min_train_cases),
            "min_test_cases": int(min_test_cases),
        },
        "shards": rows,
    }


def _enforce_split_case_counts(
    *,
    train_cases: set[str],
    test_cases: set[str],
    min_train: int,
    min_test: int,
) -> None:
    n_train = int(len(train_cases))
    n_test = int(len(test_cases))
    if int(min_train) > 0 and n_train < int(min_train):
        raise ValueError(f"insufficient_train_cases:{n_train}<{int(min_train)}")
    if int(min_test) > 0 and n_test < int(min_test):
        raise ValueError(f"insufficient_test_cases:{n_test}<{int(min_test)}")


def _enforce_test_coverage(
    *,
    coverage: dict[str, Any],
    min_event_total: int,
    min_event_per_lead: int,
    min_total: int,
    min_per_lead: int,
) -> None:
    test_leads = [str(v) for v in (coverage.get("test_leads", []) or [])]

    events = (coverage.get("events") or {}).copy()
    event_total = int(events.get("total", 0))
    event_by_lead = {str(k): int(v) for k, v in (events.get("by_lead", {}) or {}).items()}
    if int(min_event_total) > 0 and event_total < int(min_event_total):
        raise ValueError(f"insufficient_event_coverage:total={event_total}<{int(min_event_total)}")
    if int(min_event_per_lead) > 0:
        missing_ev = {
            lead: int(event_by_lead.get(lead, 0))
            for lead in test_leads
            if int(event_by_lead.get(lead, 0)) < int(min_event_per_lead)
        }
        if missing_ev:
            detail = ",".join(f"{k}:{v}<{int(min_event_per_lead)}" for k, v in sorted(missing_ev.items()))
            raise ValueError(f"insufficient_event_coverage_by_lead:{detail}")

    far = (coverage.get("far_nonstorm") or {}).copy()
    far_total = int(far.get("total", 0))
    far_by_lead = {str(k): int(v) for k, v in (far.get("by_lead", {}) or {}).items()}
    if int(min_total) > 0 and far_total < int(min_total):
        raise ValueError(f"insufficient_far_nonstorm_coverage:total={far_total}<{int(min_total)}")
    if int(min_per_lead) > 0:
        missing_far = {
            lead: int(far_by_lead.get(lead, 0))
            for lead in test_leads
            if int(far_by_lead.get(lead, 0)) < int(min_per_lead)
        }
        if missing_far:
            detail = ",".join(f"{k}:{v}<{int(min_per_lead)}" for k, v in sorted(missing_far.items()))
            raise ValueError(f"insufficient_far_nonstorm_coverage_by_lead:{detail}")


def _build_knee_detector_comparison(case_metrics: pd.DataFrame) -> dict[str, Any]:
    if case_metrics.empty:
        return {
            "n_cases": 0,
            "both_detected": 0,
            "size_only": 0,
            "tf_only": 0,
            "neither": 0,
            "rates": {},
        }
    size = pd.to_numeric(case_metrics.get("knee_size_detected", case_metrics.get("knee_detected")), errors="coerce").fillna(0).astype(bool)
    tf = pd.to_numeric(case_metrics.get("knee_tf_detected", case_metrics.get("tf_knee_detected")), errors="coerce").fillna(0).astype(bool)
    both = int((size & tf).sum())
    size_only = int((size & ~tf).sum())
    tf_only = int((~size & tf).sum())
    neither = int((~size & ~tf).sum())
    n = int(case_metrics.shape[0])
    return {
        "n_cases": n,
        "both_detected": both,
        "size_only": size_only,
        "tf_only": tf_only,
        "neither": neither,
        "rates": {
            "both_detected": float(both / n) if n else 0.0,
            "size_only": float(size_only / n) if n else 0.0,
            "tf_only": float(tf_only / n) if n else 0.0,
            "neither": float(neither / n) if n else 0.0,
            "coincidence": float(both / max(1, int((size | tf).sum()))),
        },
    }


def _build_size_knee_diagnostic(samples: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
    if samples.empty:
        return {"n_cases": 0, "n_cases_with_curve": 0, "n_cases_with_candidates": 0, "top_candidates": []}
    candidates: list[dict[str, Any]] = []
    n_cases_curve = 0
    for case_id, grp in samples.groupby("case_id", dropna=False):
        curve = _case_eta_by_scale(grp)
        if curve is None:
            continue
        n_cases_curve += 1
        L = curve["L"]
        eta = curve["eta"]
        x = np.log(np.maximum(L, 1e-12))
        y = np.log(np.maximum(eta, 1e-12))
        n = int(x.size)
        if n < 5:
            continue
        rss0 = _linear_rss(x, y)
        bic0 = _bic(rss0, n=n, k=2)
        best_for_case: dict[str, Any] | None = None
        for idx in range(2, n - 2):
            rss1, slope_left, slope_right = _piecewise_rss_and_slopes(x, y, idx)
            bic1 = _bic(rss1, n=n, k=3)
            delta_bic = float(bic0 - bic1)
            resid_impr = float((rss0 - rss1) / max(rss0, 1e-12))
            slope_ok = bool(abs(slope_left) <= 8.0 and abs(slope_right) <= 8.0)
            rec = {
                "case_id": str(case_id),
                "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown",
                "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else None,
                "n_scales": int(n),
                "knee_L": float(L[idx]),
                "delta_bic": delta_bic,
                "resid_improvement": resid_impr,
                "slope_left": float(slope_left),
                "slope_right": float(slope_right),
                "slope_delta": float(abs(slope_right - slope_left)),
                "sanity_pass_relaxed": bool(slope_ok and resid_impr >= 0.0),
                "rejection_reason": ";".join(
                    [
                        *([] if slope_ok else ["slope_unphysical"]),
                        *([] if resid_impr >= 0.0 else ["resid_not_improved"]),
                    ]
                ),
            }
            if best_for_case is None or delta_bic > float(best_for_case["delta_bic"]):
                best_for_case = rec
        if best_for_case is not None:
            candidates.append(best_for_case)
    candidates_sorted = sorted(candidates, key=lambda r: float(r.get("delta_bic", -np.inf)), reverse=True)
    return {
        "n_cases": int(samples["case_id"].nunique()),
        "n_cases_with_curve": int(n_cases_curve),
        "n_cases_with_candidates": int(len(candidates)),
        "top_candidates": candidates_sorted[: int(max(1, top_k))],
    }


def _case_eta_by_scale(case_df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    if case_df.empty:
        return None
    pivot = case_df.pivot_table(index=["t", "L"], columns="hand", values="O", aggfunc="first")
    if "L" not in pivot.columns or "R" not in pivot.columns:
        return None
    pivot = pivot.rename(columns={"L": "O_L", "R": "O_R"}).reset_index()
    o_l = pd.to_numeric(pivot["O_L"], errors="coerce").to_numpy(dtype=float)
    o_r = pd.to_numeric(pivot["O_R"], errors="coerce").to_numpy(dtype=float)
    denom = 0.5 * (o_l + o_r)
    valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
    if not np.any(valid):
        return None
    eta = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
    L = pd.to_numeric(pivot.loc[valid, "L"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(L) & np.isfinite(eta) & (L > 0) & (eta > 0)
    if not np.any(ok):
        return None
    tmp = pd.DataFrame({"L": L[ok], "eta": eta[ok]})
    curve = tmp.groupby("L", as_index=False)["eta"].median().sort_values("L")
    if curve.shape[0] < 4:
        return None
    return {
        "L": pd.to_numeric(curve["L"], errors="coerce").to_numpy(dtype=float),
        "eta": pd.to_numeric(curve["eta"], errors="coerce").to_numpy(dtype=float),
    }


def _load_axis_robust_meta(
    *,
    dataset_dir: Path,
    axis_json_arg: str | None,
    std_threshold: float,
) -> dict[str, Any]:
    path: Path | None = None
    if axis_json_arg:
        path = Path(axis_json_arg)
    else:
        build_summary = dataset_dir / "build_summary.json"
        if build_summary.exists():
            try:
                info = json.loads(build_summary.read_text(encoding="utf-8"))
                prepared_root = info.get("prepared_root")
                if prepared_root:
                    cand = Path(str(prepared_root)) / "lon0_sensitivity.json"
                    if cand.exists():
                        path = cand
            except Exception:
                path = None

    if path is None or not path.exists():
        return {
            "axis_robust": True,
            "available": False,
            "reason": "missing_lon0_sensitivity_not_applied",
            "std_threshold": float(std_threshold),
        }

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "axis_robust": True,
            "available": False,
            "reason": "invalid_lon0_sensitivity_json_not_applied",
            "path": str(path.resolve()),
            "std_threshold": float(std_threshold),
        }

    per = obj.get("per_lon0", {})
    contrast_pairs: list[tuple[float, float]] = []
    for k, row in per.items():
        e = _to_float(row.get("eta_parity_mean_events"))
        b = _to_float(row.get("eta_parity_mean_background"))
        if e is None or b is None:
            continue
        try:
            lon_val = float(k)
        except Exception:
            lon_val = _to_float(row.get("lon0")) or np.nan
        if not np.isfinite(lon_val):
            continue
        contrast_pairs.append((float(lon_val), float(e - b)))

    contrasts = [c for _, c in contrast_pairs]

    if len(contrasts) < 2:
        return {
            "axis_robust": False,
            "available": True,
            "reason": "insufficient_lons",
            "path": str(path.resolve()),
            "std_threshold": float(std_threshold),
            "n_lons": int(len(contrasts)),
            "contrast_std": None,
            "sign_consistent": False,
        }

    contrast_arr = np.asarray(contrasts, dtype=float)
    sign_consistent = bool(np.all(contrast_arr > 0) or np.all(contrast_arr < 0))
    std_val = float(np.std(contrast_arr, ddof=0))
    axis_robust = bool(sign_consistent and std_val <= float(std_threshold))
    if len(contrast_pairs) >= 2:
        x = np.asarray([v[0] for v in contrast_pairs], dtype=float)
        y = np.asarray([v[1] for v in contrast_pairs], dtype=float)
        slope = float(np.polyfit(x, y, deg=1)[0])
        best_idx = int(np.argmax(y))
        worst_idx = int(np.argmin(y))
        best_lon0 = float(x[best_idx])
        worst_lon0 = float(x[worst_idx])
    else:
        slope = None
        best_lon0 = None
        worst_lon0 = None
    return {
        "axis_robust": axis_robust,
        "available": True,
        "path": str(path.resolve()),
        "std_threshold": float(std_threshold),
        "n_lons": int(len(contrasts)),
        "contrast_std": std_val,
        "sign_consistent": sign_consistent,
        "contrast_mean": float(np.mean(contrast_arr)),
        "contrast_slope_per_lon": slope,
        "best_lon0": best_lon0,
        "worst_lon0": worst_lon0,
        "contrast_by_lon0": {f"{k:.3f}": float(v) for k, v in sorted(contrast_pairs)},
    }


def _apply_axis_robust_gating(case_metrics: pd.DataFrame, axis_robust: bool) -> pd.DataFrame:
    if case_metrics.empty:
        return case_metrics.copy()
    out = case_metrics.copy()
    raw = out["parity_signal_pass"] if "parity_signal_pass" in out.columns else pd.Series(False, index=out.index)
    out["parity_signal_pass_effective"] = raw.astype(bool) & bool(axis_robust)
    out["axis_robust"] = bool(axis_robust)
    return out


def _build_parity_ablation_report(
    *,
    test_df: pd.DataFrame,
    dataset_dir: Path,
    config_path: Path,
    seed: int,
    p_lock_threshold: float,
    axis_robust: bool,
    parity_odd_inner_outer_min: float,
    parity_tangential_radial_min: float,
    parity_inner_ring_quantile: float,
) -> dict[str, Any]:
    variants = {
        "vectors_only": "O_vector",
        "scalars_only": "O_scalar",
        "raw": "O_raw",
        "local_frame": "O_local_frame",
        "vorticity": "O_vorticity",
        "meanflow_removed": "O_meanflow",
        "anomaly_lat_hour": "O_lat_hour",
        "anomaly_lat_day": "O_lat_day",
        "polar_spiral": "O_polar_spiral",
        "polar_chiral": "O_polar_chiral",
        "polar_left": "O_polar_left",
        "polar_right": "O_polar_right",
        "polar_odd_ratio": "O_polar_odd_ratio",
        "polar_eta": "O_polar_eta",
    }
    out: dict[str, Any] = {
        "axis_robust_applied": bool(axis_robust),
        "far_nonstorm": {},
        "variants": {},
    }
    for i, (name, col) in enumerate(variants.items()):
        if col not in test_df.columns:
            out["far_nonstorm"][name] = {"available": False, "reason": f"missing_column:{col}"}
            out["variants"][name] = {"available": False, "reason": f"missing_column:{col}"}
            continue
        dfv = test_df.copy()
        vals = pd.to_numeric(dfv[col], errors="coerce")
        dfv["O"] = vals
        dfv = dfv.loc[np.isfinite(dfv["O"]) & (dfv["O"] > 0)].copy()
        if dfv.empty:
            out["far_nonstorm"][name] = {"available": False, "reason": "no_valid_rows"}
            out["variants"][name] = {"available": False, "reason": "no_valid_rows"}
            continue
        metrics = _evaluate_cases(
            dfv,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(seed + i),
            enable_time_frequency_knee=False,
            tf_knee_bic_delta_min=6.0,
            parity_odd_inner_outer_min=float(parity_odd_inner_outer_min),
            parity_tangential_radial_min=float(parity_tangential_radial_min),
            parity_inner_ring_quantile=float(parity_inner_ring_quantile),
        )
        metrics = _apply_axis_robust_gating(metrics, axis_robust=bool(axis_robust))
        strata = _summarize_case_metrics_by_strata(metrics, p_lock_threshold=float(p_lock_threshold))
        events = strata.get("events", {})
        far = strata.get("far_nonstorm", {})
        event_rate = float(events.get("parity_signal_rate", 0.0))
        far_rate = float(far.get("parity_signal_rate", 0.0))
        contrast = float(event_rate - far_rate)
        rec = {
            "available": True,
            "n_cases_total": int(metrics.shape[0]),
            "n_cases_events": int(events.get("n_cases", 0)),
            "n_cases_far_nonstorm": int(far.get("n_cases", 0)),
            "event_parity_signal_rate": event_rate,
            "far_nonstorm_parity_signal_rate": far_rate,
            "event_minus_far_parity_rate": contrast,
            "confound_rate": float(far_rate / max(event_rate, 1e-9)),
            "knee_rate_events": float(events.get("knee_rate", 0.0)),
            "knee_rate_far_nonstorm": float(far.get("knee_rate", 0.0)),
        }
        out["variants"][name] = rec
        out["far_nonstorm"][name] = {
            "available": True,
            "n_cases_total": int(metrics.shape[0]),
            "n_cases_far_nonstorm": int(far.get("n_cases", 0)),
            "parity_signal_rate": far_rate,
            "parity_signal_rate_raw": float(far.get("parity_signal_rate_raw", 0.0)),
            "knee_rate": float(far.get("knee_rate", 0.0)),
        }

    candidates: list[tuple[str, float, float]] = []
    for mode, rec in out["variants"].items():
        if not rec.get("available"):
            continue
        ev = rec.get("event_parity_signal_rate")
        far = rec.get("far_nonstorm_parity_signal_rate")
        if ev is None or far is None:
            continue
        candidates.append((str(mode), float(ev - far), float(far)))
    if candidates:
        # Prefer maximal separation (events > far_nonstorm), then lower far_nonstorm background rate.
        best = sorted(candidates, key=lambda x: (-x[1], x[2], x[0]))[0]
        out["recommended_mode_by_contrast"] = {
            "mode": best[0],
            "event_minus_far_parity_rate": best[1],
            "far_nonstorm_parity_rate": best[2],
        }
    else:
        out["recommended_mode_by_contrast"] = None
    return out


def _build_anomaly_mode_ablation(
    *,
    test_df: pd.DataFrame,
    dataset_dir: Path,
    config_path: Path,
    seed: int,
    p_lock_threshold: float,
    axis_robust: bool,
    enable_time_frequency_knee: bool,
    tf_knee_bic_delta_min: float,
    parity_odd_inner_outer_min: float,
    parity_tangential_radial_min: float,
    parity_inner_ring_quantile: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    mode_to_col = {
        "none": "O_raw",
        "lat_hour": "O_lat_hour",
        "lat_day": "O_lat_day",
    }
    out: dict[str, Any] = {"axis_robust_applied": bool(axis_robust), "modes": {}}
    mode_case_metrics: dict[str, pd.DataFrame] = {}
    base_case_ids: np.ndarray | None = None
    base_pass: np.ndarray | None = None
    base_rank: pd.DataFrame | None = None
    for i, (mode, col) in enumerate(mode_to_col.items()):
        if col not in test_df.columns:
            out["modes"][mode] = {"available": False, "reason": f"missing_column:{col}"}
            continue
        dfv = test_df.copy()
        vals = pd.to_numeric(dfv[col], errors="coerce")
        dfv["O"] = vals
        dfv = dfv.loc[np.isfinite(dfv["O"]) & (dfv["O"] > 0)].copy()
        if dfv.empty:
            out["modes"][mode] = {"available": False, "reason": "no_valid_rows"}
            continue
        metrics = _evaluate_cases(
            dfv,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(seed + i),
            enable_time_frequency_knee=bool(enable_time_frequency_knee),
            tf_knee_bic_delta_min=float(tf_knee_bic_delta_min),
            parity_odd_inner_outer_min=float(parity_odd_inner_outer_min),
            parity_tangential_radial_min=float(parity_tangential_radial_min),
            parity_inner_ring_quantile=float(parity_inner_ring_quantile),
        )
        metrics = _apply_axis_robust_gating(metrics, axis_robust=bool(axis_robust))
        mode_case_metrics[mode] = metrics.copy()
        strata = _summarize_case_metrics_by_strata(metrics, p_lock_threshold=float(p_lock_threshold))
        det_cmp = _build_knee_detector_comparison(metrics)
        events = strata.get("events", {}) if isinstance(strata, dict) else {}
        far = strata.get("far_nonstorm", {}) if isinstance(strata, dict) else {}
        event_rate = float(events.get("parity_signal_rate", 0.0))
        far_rate = float(far.get("parity_signal_rate", 0.0))
        rank_df = _compute_case_eta_scores(dfv)
        null_survival_penalty = None
        if not rank_df.empty:
            mask = _event_mask_from_frame(rank_df)
            r_event = rank_df.loc[mask].copy()
            r_far = rank_df.loc[~mask].copy()
            far_vals = pd.to_numeric(r_far.get("eta_score"), errors="coerce").to_numpy(dtype=float)
            far_vals = far_vals[np.isfinite(far_vals)]
            if far_vals.size > 0:
                med = float(np.median(far_vals))
                mad = float(np.median(np.abs(far_vals - med)))
                eta_thresh = med + max(mad, 1e-8)
            else:
                all_vals = pd.to_numeric(rank_df.get("eta_score"), errors="coerce").to_numpy(dtype=float)
                all_vals = all_vals[np.isfinite(all_vals)]
                eta_thresh = float(np.quantile(all_vals, 0.75)) if all_vals.size > 0 else 0.0

            def _event_rate_from_rank(df_rank: pd.DataFrame) -> float:
                if df_rank.empty:
                    return 0.0
                ev_mask = _event_mask_from_frame(df_rank)
                if not bool(ev_mask.any()):
                    return 0.0
                score = pd.to_numeric(df_rank["eta_score"], errors="coerce")
                return float((score.loc[ev_mask] >= float(eta_thresh)).mean())

            base_rate = _event_rate_from_rank(rank_df)
            theta_rank = _compute_case_eta_scores(_theta_roll_polar(dfv, rng=np.random.default_rng(int(seed) + 100 + i)))
            center_rank = _compute_case_eta_scores(_center_jitter_polar(dfv, rng=np.random.default_rng(int(seed) + 200 + i)))
            theta_rate = _event_rate_from_rank(theta_rank)
            center_rate = _event_rate_from_rank(center_rank)
            if base_rate > 0:
                null_survival_penalty = float(
                    0.5 * (theta_rate / max(base_rate, 1e-9)) + 0.5 * (center_rate / max(base_rate, 1e-9))
                )
            else:
                null_survival_penalty = 0.0
        mode_rec = {
            "available": True,
            "n_cases_total": int(metrics.shape[0]),
            "strata": strata,
            "knee_detector_comparison": det_cmp,
            "event_parity_signal_rate": event_rate,
            "far_nonstorm_parity_signal_rate": far_rate,
            "event_minus_far_parity_rate": float(event_rate - far_rate),
            "mode_complexity": float({"none": 0.0, "lat_day": 1.0, "lat_hour": 2.0}.get(str(mode), 1.0)),
            "null_survival_penalty": _to_float(null_survival_penalty),
        }
        out["modes"][mode] = mode_rec

        pass_series = (
            metrics[["case_id", "parity_signal_pass_effective"]]
            .drop_duplicates(subset=["case_id"], keep="first")
            .sort_values("case_id")
        )
        ids = pass_series["case_id"].astype(str).to_numpy()
        vals_pass = pass_series["parity_signal_pass_effective"].astype(bool).to_numpy()
        if mode == "none":
            base_case_ids = ids
            base_pass = vals_pass
            mode_rec["parity_decision_agreement_vs_none"] = 1.0
            base_rank = rank_df.copy() if rank_df is not None else None
            mode_rec["parity_rank_agreement_vs_none"] = 1.0
        elif base_case_ids is not None and base_pass is not None:
            merged = pd.merge(
                pd.DataFrame({"case_id": base_case_ids, "base": base_pass}),
                pd.DataFrame({"case_id": ids, "curr": vals_pass}),
                on="case_id",
                how="inner",
            )
            if merged.empty:
                mode_rec["parity_decision_agreement_vs_none"] = None
            else:
                mode_rec["parity_decision_agreement_vs_none"] = float((merged["base"] == merged["curr"]).mean())
            mode_rec["parity_rank_agreement_vs_none"] = _rank_agreement_vs_base(base_rank, rank_df)
        else:
            mode_rec["parity_rank_agreement_vs_none"] = None

    agreements = [
        float(v.get("parity_decision_agreement_vs_none"))
        for k, v in out["modes"].items()
        if k != "none" and v.get("available") and v.get("parity_decision_agreement_vs_none") is not None
    ]
    rank_agreements = [
        float(v.get("parity_rank_agreement_vs_none"))
        for k, v in out["modes"].items()
        if k != "none" and v.get("available") and v.get("parity_rank_agreement_vs_none") is not None
    ]
    out["decision_stability"] = {
        "agreement_vs_none_min": float(min(agreements)) if agreements else None,
        "agreement_vs_none_mean": float(np.mean(agreements)) if agreements else None,
        "rank_agreement_vs_none_min": float(min(rank_agreements)) if rank_agreements else None,
        "rank_agreement_vs_none_mean": float(np.mean(rank_agreements)) if rank_agreements else None,
        "stable": bool(agreements and min(agreements) >= 0.8),
    }
    return out, mode_case_metrics


def _rank_anomaly_modes_for_canonical(
    anomaly_ablation: dict[str, Any],
    *,
    mode_agreement_min: float,
) -> list[str]:
    modes = (anomaly_ablation or {}).get("modes", {}) or {}
    candidates: list[tuple[str, float, float, float, float, float, float]] = []
    fallback: list[tuple[str, float, float, float, float, float, float]] = []
    w_margin = 1.0
    w_far = 1.0
    w_null = 1.0
    w_complexity = 0.05
    w_agree = 0.10
    w_rank = 0.10
    for mode, rec in modes.items():
        if not isinstance(rec, dict) or not bool(rec.get("available", False)):
            continue
        event_rate = _to_float(rec.get("event_parity_signal_rate")) or 0.0
        far_rate = _to_float(rec.get("far_nonstorm_parity_signal_rate")) or 0.0
        agree = _to_float(rec.get("parity_decision_agreement_vs_none"))
        if agree is None and str(mode) == "none":
            agree = 1.0
        rank_agree = _to_float(rec.get("parity_rank_agreement_vs_none"))
        if rank_agree is None and str(mode) == "none":
            rank_agree = 1.0
        margin = _to_float(rec.get("event_minus_far_parity_rate"))
        if margin is None:
            margin = float(event_rate - far_rate)
        null_survival = _to_float(rec.get("null_survival_penalty")) or 0.0
        complexity = _to_float(rec.get("mode_complexity")) or 0.0
        score = (
            w_margin * float(margin)
            - w_far * float(far_rate)
            - w_null * float(null_survival)
            - w_complexity * float(complexity)
            + w_agree * float(agree or 0.0)
            + w_rank * float(rank_agree or 0.0)
        )
        row = (
            str(mode),
            float(score),
            float(margin),
            float(far_rate),
            float(event_rate),
            float(agree or 0.0),
            float(rank_agree or 0.0),
        )
        fallback.append(row)
        if float(agree or 0.0) >= float(mode_agreement_min):
            candidates.append(row)
    if not candidates:
        candidates = fallback
    if not candidates:
        return ["none", "lat_hour", "lat_day"]
    # Maximize ranking score, then margin, then lower far background.
    ordered = sorted(candidates, key=lambda x: (-x[1], -x[2], x[3], -x[5], -x[6], x[0]))
    ranked = [m for m, _, _, _, _, _, _ in ordered]
    for fallback in ("none", "lat_hour", "lat_day"):
        if fallback not in ranked:
            ranked.append(fallback)
    return ranked


def _build_casewise_anomaly_mode_selection(
    *,
    mode_case_metrics: dict[str, pd.DataFrame],
    mode_priority: list[str],
    stability_min: float,
) -> dict[str, Any]:
    if not mode_case_metrics:
        empty = pd.DataFrame(
            columns=[
                "case_id",
                "case_type",
                "canonical_mode",
                "canonical_parity_signal_pass",
                "ensemble_parity_signal_pass",
                "stability_score",
                "stable_case",
                "n_modes_available",
                "n_positive_votes",
                "n_negative_votes",
                "majority_margin",
                "selection_reason",
            ]
        )
        return {
            "summary": {
                "n_cases": 0,
                "stable_case_rate": 0.0,
                "stability_score_mean": None,
                "stability_score_min": None,
                "canonical_mode_counts": {},
                "selection_disagreement_rate": 0.0,
            },
            "per_case": empty,
        }

    mode_tables: dict[str, pd.DataFrame] = {}
    all_case_ids: set[str] = set()
    all_modes = [str(m) for m in mode_case_metrics.keys()]
    for mode, metrics in mode_case_metrics.items():
        if metrics is None or metrics.empty:
            continue
        tab = metrics.copy()
        tab["case_id"] = tab["case_id"].astype(str)
        tab = tab.drop_duplicates(subset=["case_id"], keep="first").set_index("case_id", drop=False)
        mode_tables[str(mode)] = tab
        all_case_ids.update(tab.index.astype(str).tolist())

    ranked_modes = [m for m in mode_priority if m in mode_tables]
    for mode in sorted(mode_tables.keys()):
        if mode not in ranked_modes:
            ranked_modes.append(mode)
    if not ranked_modes:
        ranked_modes = sorted(mode_tables.keys())

    rows: list[dict[str, Any]] = []
    for cid in sorted(all_case_ids):
        mode_decisions: dict[str, bool] = {}
        mode_case_types: dict[str, str] = {}
        mode_storm_max: dict[str, int] = {}
        mode_near_max: dict[str, int] = {}
        mode_pregen_max: dict[str, int] = {}
        mode_storm_cohort: dict[str, str] = {}
        for mode in ranked_modes:
            tab = mode_tables.get(mode)
            if tab is None or cid not in tab.index:
                continue
            rec = tab.loc[cid]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            decision = bool(rec.get("parity_signal_pass_effective", rec.get("parity_signal_pass", False)))
            mode_decisions[mode] = decision
            mode_case_types[mode] = str(rec.get("case_type", "unknown"))
            mode_storm_max[mode] = int(pd.to_numeric(rec.get("storm_max", 0), errors="coerce") or 0)
            mode_near_max[mode] = int(pd.to_numeric(rec.get("near_storm_max", 0), errors="coerce") or 0)
            mode_pregen_max[mode] = int(pd.to_numeric(rec.get("pregen_max", 0), errors="coerce") or 0)
            mode_storm_cohort[mode] = str(rec.get("storm_distance_cohort", "") or "")

        if not mode_decisions:
            continue

        n_avail = int(len(mode_decisions))
        n_pos = int(sum(1 for v in mode_decisions.values() if v))
        n_neg = int(n_avail - n_pos)
        majority_decision = bool(n_pos > n_neg)
        if n_pos == n_neg:
            # Tie-break with highest-priority available mode.
            tie_mode = next((m for m in ranked_modes if m in mode_decisions), None)
            majority_decision = bool(mode_decisions.get(str(tie_mode), False)) if tie_mode is not None else False
        stability = float(max(n_pos, n_neg) / max(1, n_avail))
        stable_case = bool(stability >= float(stability_min))
        majority_margin = float(abs(n_pos - n_neg) / max(1, n_avail))

        if stable_case:
            canonical_mode = next(
                (m for m in ranked_modes if m in mode_decisions and mode_decisions[m] == majority_decision),
                next((m for m in ranked_modes if m in mode_decisions), sorted(mode_decisions.keys())[0]),
            )
            selection_reason = "stable_majority_priority"
        else:
            canonical_mode = next((m for m in ranked_modes if m in mode_decisions), sorted(mode_decisions.keys())[0])
            selection_reason = "unstable_priority_fallback"

        canonical_decision = bool(mode_decisions.get(canonical_mode, majority_decision))
        ensemble_decision = bool(majority_decision)
        case_type = mode_case_types.get(canonical_mode) or next(iter(mode_case_types.values()))
        storm_max = int(mode_storm_max.get(canonical_mode, max(mode_storm_max.values()) if mode_storm_max else 0))
        near_storm_max = int(mode_near_max.get(canonical_mode, max(mode_near_max.values()) if mode_near_max else 0))
        pregen_max = int(mode_pregen_max.get(canonical_mode, max(mode_pregen_max.values()) if mode_pregen_max else 0))
        storm_distance_cohort = mode_storm_cohort.get(canonical_mode) or next(iter(mode_storm_cohort.values()), "")

        row: dict[str, Any] = {
            "case_id": str(cid),
            "case_type": str(case_type),
            "storm_max": int(storm_max),
            "near_storm_max": int(near_storm_max),
            "pregen_max": int(pregen_max),
            "storm_distance_cohort": str(storm_distance_cohort),
            "canonical_mode": str(canonical_mode),
            "canonical_parity_signal_pass": bool(canonical_decision),
            "ensemble_parity_signal_pass": bool(ensemble_decision),
            "parity_pass_all_modes": bool(all(mode_decisions.values())),
            "parity_pass_any_modes": bool(any(mode_decisions.values())),
            "stability_score": float(stability),
            "stable_case": bool(stable_case),
            "n_modes_available": int(n_avail),
            "n_positive_votes": int(n_pos),
            "n_negative_votes": int(n_neg),
            "majority_margin": float(majority_margin),
            "selection_reason": str(selection_reason),
        }
        for mode in ranked_modes:
            key = f"parity_{mode}"
            row[key] = bool(mode_decisions[mode]) if mode in mode_decisions else None
        rows.append(row)

    per_case = pd.DataFrame(rows)
    if per_case.empty:
        return {
            "summary": {
                "n_cases": 0,
                "stable_case_rate": 0.0,
                "stability_score_mean": None,
                "stability_score_min": None,
                "canonical_mode_counts": {},
                "selection_disagreement_rate": 0.0,
            },
            "per_case": per_case,
        }

    mode_counts = (
        per_case["canonical_mode"].value_counts(dropna=False).to_dict()
        if "canonical_mode" in per_case.columns
        else {}
    )
    disagreement = float(
        (
            per_case["canonical_parity_signal_pass"].astype(bool)
            != per_case["ensemble_parity_signal_pass"].astype(bool)
        ).mean()
    )
    summary = {
        "n_cases": int(per_case.shape[0]),
        "stability_min_threshold": float(stability_min),
        "stable_case_rate": float(per_case["stable_case"].astype(bool).mean()),
        "stability_score_mean": float(pd.to_numeric(per_case["stability_score"], errors="coerce").mean()),
        "stability_score_min": _safe_median(pd.to_numeric(per_case["stability_score"], errors="coerce").sort_values().head(1)),
        "n_modes_available_median": float(pd.to_numeric(per_case["n_modes_available"], errors="coerce").median()),
        "canonical_mode_counts": {str(k): int(v) for k, v in mode_counts.items()},
        "canonical_positive_rate": float(per_case["canonical_parity_signal_pass"].astype(bool).mean()),
        "ensemble_positive_rate": float(per_case["ensemble_parity_signal_pass"].astype(bool).mean()),
        "all_modes_positive_rate": float(per_case["parity_pass_all_modes"].astype(bool).mean()),
        "any_mode_positive_rate": float(per_case["parity_pass_any_modes"].astype(bool).mean()),
        "selection_disagreement_rate": disagreement,
    }
    return {"summary": summary, "per_case": per_case}


def _build_mode_invariant_parity_summary(selection_df: pd.DataFrame) -> dict[str, Any]:
    if selection_df is None or selection_df.empty:
        return {
            "n_cases": 0,
            "stable_fraction": 0.0,
            "all_modes_positive_rate": 0.0,
            "any_mode_positive_rate": 0.0,
            "canonical_positive_rate": 0.0,
            "ensemble_positive_rate": 0.0,
        }
    stable = selection_df.get("stable_case", pd.Series(dtype=bool)).astype(bool)
    all_pos = selection_df.get("parity_pass_all_modes", pd.Series(dtype=bool)).astype(bool)
    any_pos = selection_df.get("parity_pass_any_modes", pd.Series(dtype=bool)).astype(bool)
    canon = selection_df.get("canonical_parity_signal_pass", pd.Series(dtype=bool)).astype(bool)
    ens = selection_df.get("ensemble_parity_signal_pass", pd.Series(dtype=bool)).astype(bool)
    return {
        "n_cases": int(selection_df.shape[0]),
        "stable_fraction": float(stable.mean()) if len(stable) else 0.0,
        "all_modes_positive_rate": float(all_pos.mean()) if len(all_pos) else 0.0,
        "any_mode_positive_rate": float(any_pos.mean()) if len(any_pos) else 0.0,
        "canonical_positive_rate": float(canon.mean()) if len(canon) else 0.0,
        "ensemble_positive_rate": float(ens.mean()) if len(ens) else 0.0,
    }


def _compute_case_eta_scores(samples: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty:
        return pd.DataFrame(columns=["case_id", "case_type", "storm_max", "near_storm_max", "pregen_max", "eta_score"])
    req = {"case_id", "t", "L", "hand", "O"}
    if not req.issubset(set(samples.columns)):
        return pd.DataFrame(columns=["case_id", "case_type", "storm_max", "near_storm_max", "pregen_max", "eta_score"])
    rows: list[dict[str, Any]] = []
    for case_id, grp in samples.groupby("case_id", dropna=False):
        pivot = grp.pivot_table(index=["t", "L"], columns="hand", values="O", aggfunc="first")
        if "L" not in pivot.columns or "R" not in pivot.columns:
            continue
        storm_series = pd.to_numeric(grp["storm"], errors="coerce").fillna(0) if "storm" in grp.columns else pd.Series(0.0, index=grp.index)
        near_series = pd.to_numeric(grp["near_storm"], errors="coerce").fillna(0) if "near_storm" in grp.columns else pd.Series(0.0, index=grp.index)
        pregen_series = pd.to_numeric(grp["pregen"], errors="coerce").fillna(0) if "pregen" in grp.columns else pd.Series(0.0, index=grp.index)
        o_l = pd.to_numeric(pivot["L"], errors="coerce").to_numpy(dtype=float)
        o_r = pd.to_numeric(pivot["R"], errors="coerce").to_numpy(dtype=float)
        denom = 0.5 * (o_l + o_r)
        valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
        if not np.any(valid):
            continue
        eta = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
        rows.append(
            {
                "case_id": str(case_id),
                "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown",
                "storm_max": int(storm_series.max()),
                "near_storm_max": int(near_series.max()),
                "pregen_max": int(pregen_series.max()),
                "eta_score": float(np.nanmedian(eta)),
            }
        )
    return pd.DataFrame(rows)


def _rank_agreement_vs_base(base_rank: pd.DataFrame | None, curr_rank: pd.DataFrame | None) -> float | None:
    if base_rank is None or curr_rank is None or base_rank.empty or curr_rank.empty:
        return None
    left = base_rank[["case_id", "eta_score"]].rename(columns={"eta_score": "eta_base"})
    right = curr_rank[["case_id", "eta_score"]].rename(columns={"eta_score": "eta_curr"})
    merged = pd.merge(left, right, on="case_id", how="inner")
    if merged.shape[0] < 3:
        return None
    tau = kendalltau(
        pd.to_numeric(merged["eta_base"], errors="coerce"),
        pd.to_numeric(merged["eta_curr"], errors="coerce"),
        nan_policy="omit",
    )
    return _to_float(getattr(tau, "correlation", None))


def _event_mask_from_case_type(case_type: pd.Series) -> pd.Series:
    ct = case_type.fillna("unknown").astype(str).str.lower()
    return (~ct.eq("control")) & (~ct.eq("far_nonstorm"))


def _event_mask_from_frame(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    strata = _derive_strata_labels(frame)
    return strata.isin(["events", "near_storm"]).reindex(frame.index, fill_value=False).astype(bool)


def _derive_strata_labels(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    idx = frame.index
    def _num_series(primary: str, fallback: str | None = None) -> pd.Series:
        if primary in frame.columns:
            return pd.to_numeric(frame[primary], errors="coerce").fillna(0.0)
        if fallback and fallback in frame.columns:
            return pd.to_numeric(frame[fallback], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=idx, dtype=float)

    case_type = frame.get("case_type", pd.Series(index=idx)).fillna("").astype(str).str.lower().str.strip()
    cohort = frame.get("storm_distance_cohort", pd.Series(index=idx)).fillna("").astype(str).str.lower().str.strip()
    storm = _num_series("storm_max", "storm")
    near = _num_series("near_storm_max", "near_storm")
    pregen = _num_series("pregen_max", "pregen")
    ib_event = _num_series("ib_event_strict_max", "ib_event_strict")
    ib_event = (ib_event > 0) | (_num_series("ib_event_max", "ib_event") > 0)
    ib_far = _num_series("ib_far_strict_max", "ib_far_strict")
    ib_far = (ib_far > 0) | (_num_series("ib_far_max", "ib_far") > 0)

    labels = pd.Series("other", index=idx, dtype=object)
    labels.loc[cohort.isin({"far_nonstorm", "far", "background", "quiet"})] = "far_nonstorm"
    labels.loc[cohort.isin({"near_storm", "near", "transition"})] = "near_storm"
    labels.loc[cohort.isin({"event", "events", "storm"})] = "events"

    control_like = case_type.isin({"control", "background", "far_nonstorm"})
    labels.loc[control_like] = "far_nonstorm"
    explicit_event_info = bool(((storm > 0) | (near > 0) | (pregen > 0) | ib_event | ib_far).any())
    if not explicit_event_info:
        event_like = case_type.isin({"storm", "event", "events"})
        near_like = case_type.isin({"near_storm", "near", "transition", "pregen"})
        labels.loc[event_like] = "events"
        labels.loc[near_like] = "near_storm"

    labels.loc[(labels == "other") & ib_far & (~ib_event)] = "far_nonstorm"
    labels.loc[(labels == "other") & ib_event & (~ib_far)] = "events"
    labels.loc[(labels == "other") & ib_event & ib_far] = "near_storm"

    labels.loc[(labels == "other") & ((storm > 0) | (pregen > 0))] = "events"
    labels.loc[(labels == "other") & (near > 0)] = "near_storm"
    labels.loc[(labels == "other") & (storm <= 0) & (near <= 0) & (pregen <= 0)] = "far_nonstorm"
    return labels.reindex(idx, fill_value="other").astype(str)


def _filter_claim_leads_by_far_coverage(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_far_per_lead: int,
    strict_time_basis: str = "selected",
    strict_use_flags: bool = True,
    far_quality_tags: tuple[str, ...] = (),
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if int(min_far_per_lead) <= 0:
        return train_df, test_df, {"enabled": False, "included_leads": sorted(test_df.get("lead_bucket", pd.Series(dtype=str)).astype(str).dropna().unique().tolist())}
    if test_df is None or test_df.empty or "lead_bucket" not in test_df.columns:
        return train_df, test_df, {"enabled": True, "included_leads": [], "excluded_leads": [], "far_cases_by_lead": {}}

    strict_for_leads = _strict_ib_case_flags(
        samples=test_df,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    if strict_for_leads.empty:
        case_meta = (
            test_df[["case_id", "lead_bucket"]]
            .drop_duplicates(subset=["case_id"], keep="first")
            .copy()
        )
        case_meta["lead_bucket"] = case_meta["lead_bucket"].astype(str)
        far_counts: dict[str, int] = {}
        leads_all = sorted(case_meta["lead_bucket"].dropna().astype(str).unique().tolist())
    else:
        strict_for_leads = strict_for_leads.copy()
        strict_for_leads["lead_bucket"] = strict_for_leads["lead_bucket"].astype(str)
        far_counts = (
            strict_for_leads.loc[pd.to_numeric(strict_for_leads["far_flag"], errors="coerce").fillna(0).astype(bool)]
            .groupby("lead_bucket")["case_id"]
            .nunique()
            .to_dict()
        )
        leads_all = sorted(strict_for_leads["lead_bucket"].dropna().astype(str).unique().tolist())
    included = [lead for lead in leads_all if int(far_counts.get(str(lead), 0)) >= int(min_far_per_lead)]
    excluded = [lead for lead in leads_all if lead not in included]
    if not included:
        included = leads_all
        excluded = []
    train_out = train_df.loc[train_df["lead_bucket"].astype(str).isin(included)].copy() if (train_df is not None and not train_df.empty and "lead_bucket" in train_df.columns) else train_df
    test_out = test_df.loc[test_df["lead_bucket"].astype(str).isin(included)].copy()

    # Avoid pathological filtering where far-per-lead pruning removes strict IB events needed
    # for train-time calibration and claim-time evaluation.
    fallback_reason = ""
    try:
        strict_train = _strict_ib_case_flags(
            samples=train_out if train_out is not None else pd.DataFrame(),
            time_basis=str(strict_time_basis),
            use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(far_quality_tags),
            min_far_quality_fraction=float(min_far_quality_fraction),
            min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
            strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
        )
        strict_test = _strict_ib_case_flags(
            samples=test_out if test_out is not None else pd.DataFrame(),
            time_basis=str(strict_time_basis),
            use_flags=bool(strict_use_flags),
            far_quality_tags=tuple(far_quality_tags),
            min_far_quality_fraction=float(min_far_quality_fraction),
            min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
            strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
        )
        n_ev_train = int(pd.to_numeric(strict_train.get("event_flag"), errors="coerce").fillna(0).astype(bool).sum()) if not strict_train.empty else 0
        n_ev_test = int(pd.to_numeric(strict_test.get("event_flag"), errors="coerce").fillna(0).astype(bool).sum()) if not strict_test.empty else 0
        if (n_ev_train <= 0) or (n_ev_test <= 0):
            train_out = train_df.copy() if train_df is not None else train_df
            test_out = test_df.copy() if test_df is not None else test_df
            included = leads_all
            excluded = []
            fallback_reason = f"disabled_due_event_drop:train={n_ev_train},test={n_ev_test}"
    except Exception:
        fallback_reason = "disabled_due_event_drop:strict_check_error"

    info = {
        "enabled": True,
        "min_far_per_lead": int(min_far_per_lead),
        "included_leads": included,
        "excluded_leads": excluded,
        "far_cases_by_lead": {str(k): int(v) for k, v in sorted(far_counts.items())},
        "strict_time_basis": str(strict_time_basis),
        "strict_use_flags": bool(strict_use_flags),
        "fallback_reason": str(fallback_reason),
    }
    return train_out, test_out, info


def _filter_claim_to_strict_far_controls(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allowed_quality_tags: tuple[str, ...] = ("A_strict_clean",),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    def _strict_far_series(df: pd.DataFrame) -> pd.Series:
        for col in ("ib_far_strict_max", "ib_far_strict", "ib_far_max", "ib_far"):
            if col in df.columns:
                return (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0)
        return pd.Series(False, index=df.index, dtype=bool)

    def _apply(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        if df is None or df.empty:
            return df, {"n_rows": 0, "n_cases_total": 0, "n_cases_kept": 0, "n_cases_dropped_non_strict_far": 0}
        strata = _derive_strata_labels(df)
        strict_far = _strict_far_series(df).reindex(df.index, fill_value=False).astype(bool)
        quality = (
            df.get("ib_far_quality_tag", pd.Series(index=df.index))
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        allowed = {str(v).strip().lower() for v in (allowed_quality_tags or ()) if str(v).strip()}
        if allowed:
            strict_far = strict_far & quality.isin(allowed)
        far_mask = strata == "far_nonstorm"
        keep = (~far_mask) | strict_far
        kept = df.loc[keep].copy()
        case_total = int(df["case_id"].astype(str).nunique()) if "case_id" in df.columns else 0
        case_keep = int(kept["case_id"].astype(str).nunique()) if "case_id" in kept.columns else 0
        dropped = 0
        if "case_id" in df.columns:
            case_far = (
                pd.DataFrame({"case_id": df["case_id"].astype(str), "far": far_mask.to_numpy(dtype=bool), "strict": strict_far.to_numpy(dtype=bool)})
                .groupby("case_id", as_index=False)
                .agg(far=("far", "max"), strict=("strict", "max"))
            )
            dropped = int(case_far.loc[(case_far["far"]) & (~case_far["strict"])].shape[0])
        info = {
            "n_rows": int(df.shape[0]),
            "n_rows_kept": int(kept.shape[0]),
            "n_cases_total": case_total,
            "n_cases_kept": case_keep,
            "n_cases_dropped_non_strict_far": dropped,
            "allowed_quality_tags": sorted(list(allowed)),
            "far_quality_tag_counts": quality.value_counts(dropna=False).to_dict(),
        }
        return kept, info

    train_out, train_info = _apply(train_df)
    test_out, test_info = _apply(test_df)
    return (
        train_out if train_out is not None else train_df,
        test_out if test_out is not None else test_df,
        {"enabled": True, "train": train_info, "test": test_info},
    )


def _apply_strict_claim_strata(
    *,
    case_metrics: pd.DataFrame,
    samples: pd.DataFrame,
    strict_time_basis: str,
    strict_use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if case_metrics is None or case_metrics.empty:
        return case_metrics, {
            "available": False,
            "reason": "empty_case_metrics",
            "n_cases_total": 0,
            "n_cases_event": 0,
            "n_cases_far": 0,
        }
    if samples is None or samples.empty:
        out_empty = case_metrics.copy()
        out_empty["split_stratum"] = _derive_strata_labels(out_empty).astype(str)
        return out_empty, {
            "available": False,
            "reason": "empty_samples",
            "n_cases_total": int(out_empty.shape[0]),
            "n_cases_event": int((out_empty["split_stratum"] == "events").sum()),
            "n_cases_far": int((out_empty["split_stratum"] == "far_nonstorm").sum()),
        }
    strict_case = _strict_ib_case_flags(
        samples=samples,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    if strict_case.empty:
        out_missing = case_metrics.copy()
        out_missing["split_stratum"] = _derive_strata_labels(out_missing).astype(str)
        return out_missing, {
            "available": False,
            "reason": "missing_strict_ib_flags",
            "n_cases_total": int(out_missing.shape[0]),
            "n_cases_event": int((out_missing["split_stratum"] == "events").sum()),
            "n_cases_far": int((out_missing["split_stratum"] == "far_nonstorm").sum()),
        }
    out = case_metrics.copy()
    out["case_id"] = out["case_id"].astype(str)
    strict_case = strict_case.copy()
    strict_case["case_id"] = strict_case["case_id"].astype(str)
    strict_idx = strict_case.set_index("case_id")
    ev_map = pd.to_numeric(strict_idx["event_flag"], errors="coerce").fillna(0).astype(bool).to_dict()
    fr_map = pd.to_numeric(strict_idx["far_flag"], errors="coerce").fillna(0).astype(bool).to_dict()
    out["strict_event_flag"] = out["case_id"].map(ev_map).fillna(False).astype(bool)
    out["strict_far_flag"] = out["case_id"].map(fr_map).fillna(False).astype(bool)
    out["split_stratum"] = "other"
    out.loc[out["strict_event_flag"] & (~out["strict_far_flag"]), "split_stratum"] = "events"
    out.loc[(~out["strict_event_flag"]) & out["strict_far_flag"], "split_stratum"] = "far_nonstorm"
    out.loc[out["strict_event_flag"] & out["strict_far_flag"], "split_stratum"] = "near_storm"
    info = {
        "available": True,
        "reason": "ok",
        "n_cases_total": int(out.shape[0]),
        "n_cases_event": int((out["split_stratum"] == "events").sum()),
        "n_cases_far": int((out["split_stratum"] == "far_nonstorm").sum()),
        "n_cases_near": int((out["split_stratum"] == "near_storm").sum()),
        "time_basis": str((strict_case.get("time_basis", pd.Series(["selected"])).iloc[0])),
        "event_col_used": str((strict_case.get("event_col_used", pd.Series([""])).iloc[0])),
        "far_col_used": str((strict_case.get("far_col_used", pd.Series([""])).iloc[0])),
        "far_quality_tags": sorted(
            {
                str(v).strip().lower()
                for v in (far_quality_tags or ())
                if str(v).strip()
            }
        ),
    }
    return out, info


def _strict_case_counts_from_samples(
    *,
    samples: pd.DataFrame,
    strict_time_basis: str,
    strict_use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> dict[str, Any]:
    if samples is None or samples.empty:
        return {
            "available": False,
            "reason": "empty_samples",
            "n_rows": 0,
            "n_cases_total": 0,
            "n_cases_event": 0,
            "n_cases_far": 0,
            "n_cases_near": 0,
            "n_cases_other": 0,
        }
    strict_case = _strict_ib_case_flags(
        samples=samples,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    if strict_case.empty:
        return {
            "available": False,
            "reason": "missing_strict_ib_flags",
            "n_rows": int(samples.shape[0]),
            "n_cases_total": int(samples["case_id"].astype(str).nunique()) if "case_id" in samples.columns else 0,
            "n_cases_event": 0,
            "n_cases_far": 0,
            "n_cases_near": 0,
            "n_cases_other": int(samples["case_id"].astype(str).nunique()) if "case_id" in samples.columns else 0,
        }
    ev = pd.to_numeric(strict_case.get("event_flag"), errors="coerce").fillna(0).astype(bool)
    fr = pd.to_numeric(strict_case.get("far_flag"), errors="coerce").fillna(0).astype(bool)
    return {
        "available": True,
        "reason": "ok",
        "n_rows": int(samples.shape[0]),
        "n_cases_total": int(strict_case.shape[0]),
        "n_cases_event": int(ev.sum()),
        "n_cases_far": int(fr.sum()),
        "n_cases_near": int((ev & fr).sum()),
        "n_cases_other": int((~ev & ~fr).sum()),
        "time_basis": str((strict_case.get("time_basis", pd.Series(["selected"])).iloc[0])),
    }


def _build_claim_retention_audit(
    *,
    train_stages: dict[str, dict[str, Any]],
    test_stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def _stage_rows(stage_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        prev: dict[str, Any] | None = None
        for name, rec in stage_map.items():
            row = {"stage": str(name), **(rec or {})}
            if prev is not None:
                row["delta_event_vs_prev"] = int(row.get("n_cases_event", 0)) - int(prev.get("n_cases_event", 0))
                row["delta_far_vs_prev"] = int(row.get("n_cases_far", 0)) - int(prev.get("n_cases_far", 0))
                row["delta_total_vs_prev"] = int(row.get("n_cases_total", 0)) - int(prev.get("n_cases_total", 0))
            prev = rec or {}
            rows.append(row)
        return rows

    train_rows = _stage_rows(train_stages)
    test_rows = _stage_rows(test_stages)
    return {
        "train": {"stages": train_rows},
        "test": {"stages": test_rows},
    }


def _calibrate_parity_thresholds_from_train(
    *,
    train_case_metrics: pd.DataFrame,
    default_odd_inner_outer_min: float,
    default_tangential_radial_min: float,
    event_min_floor: float = 0.05,
    constrain_min_thresholds: bool = False,
) -> dict[str, Any]:
    defaults = {
        "odd_inner_outer_min": float(default_odd_inner_outer_min),
        "tangential_radial_min": float(default_tangential_radial_min),
    }
    if train_case_metrics is None or train_case_metrics.empty:
        return {
            "available": False,
            "reason": "empty_train_case_metrics",
            "thresholds": defaults,
            "search": [],
        }
    df = train_case_metrics.copy()
    if "split_stratum" not in df.columns:
        df["split_stratum"] = _derive_strata_labels(df).astype(str)
    event_mask = df["split_stratum"].astype(str).isin(["events", "near_storm"])
    far_mask = df["split_stratum"].astype(str).eq("far_nonstorm")
    if (not bool(event_mask.any())) or (not bool(far_mask.any())):
        return {
            "available": False,
            "reason": "missing_event_or_far_train_cases",
            "thresholds": defaults,
            "search": [],
        }
    odd = pd.to_numeric(df.get("odd_energy_inner_outer_ratio"), errors="coerce")
    tan = pd.to_numeric(df.get("tangential_radial_ratio"), errors="coerce")
    base = pd.to_numeric(df.get("parity_signal_pass"), errors="coerce").fillna(0).astype(bool)
    valid = np.isfinite(odd) & np.isfinite(tan)
    d = df.loc[valid].copy()
    if d.empty:
        return {
            "available": False,
            "reason": "missing_parity_ratio_features",
            "thresholds": defaults,
            "search": [],
        }
    odd_vals = pd.to_numeric(d["odd_energy_inner_outer_ratio"], errors="coerce").to_numpy(dtype=float)
    tan_vals = pd.to_numeric(d["tangential_radial_ratio"], errors="coerce").to_numpy(dtype=float)
    base_vals = pd.to_numeric(d["parity_signal_pass"], errors="coerce").fillna(0).astype(bool).to_numpy()
    ev_vals = d["split_stratum"].astype(str).isin(["events", "near_storm"]).to_numpy()
    far_vals = d["split_stratum"].astype(str).eq("far_nonstorm").to_numpy()
    if (not bool(np.any(ev_vals))) or (not bool(np.any(far_vals))):
        return {
            "available": False,
            "reason": "missing_event_or_far_after_feature_filter",
            "thresholds": defaults,
            "search": [],
        }

    q = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    odd_candidates = sorted(
        {
            float(default_odd_inner_outer_min),
            *[float(np.nanquantile(odd_vals, qq)) for qq in q if np.isfinite(np.nanquantile(odd_vals, qq))],
        }
    )
    tan_candidates = sorted(
        {
            float(default_tangential_radial_min),
            *[float(np.nanquantile(tan_vals, qq)) for qq in q if np.isfinite(np.nanquantile(tan_vals, qq))],
        }
    )
    if bool(constrain_min_thresholds):
        odd_candidates = [v for v in odd_candidates if float(v) >= float(default_odd_inner_outer_min)]
        tan_candidates = [v for v in tan_candidates if float(v) >= float(default_tangential_radial_min)]
        if not odd_candidates:
            odd_candidates = [float(default_odd_inner_outer_min)]
        if not tan_candidates:
            tan_candidates = [float(default_tangential_radial_min)]

    rows: list[dict[str, Any]] = []
    for odd_thr in odd_candidates:
        for tan_thr in tan_candidates:
            pred = base_vals & (odd_vals >= float(odd_thr)) & (tan_vals >= float(tan_thr))
            event_rate = float(np.mean(pred[ev_vals])) if np.any(ev_vals) else 0.0
            far_rate = float(np.mean(pred[far_vals])) if np.any(far_vals) else 0.0
            margin = float(event_rate - far_rate)
            confound_ratio = float(far_rate / max(event_rate, 1e-9))
            floor_penalty = float(max(0.0, float(event_min_floor) - event_rate))
            safety_ok = bool((event_rate >= far_rate) and (event_rate >= float(event_min_floor)))
            # Storm-vs-far separation objective with explicit anti-no-op guard.
            score = float(
                margin
                - 0.50 * far_rate
                - 0.20 * confound_ratio
                - 2.00 * floor_penalty
                + 0.10 * event_rate
            )
            rows.append(
                {
                    "odd_inner_outer_min": float(odd_thr),
                    "tangential_radial_min": float(tan_thr),
                    "event_rate": event_rate,
                    "far_rate": far_rate,
                    "margin": margin,
                    "confound_ratio": confound_ratio,
                    "event_floor_penalty": floor_penalty,
                    "score": score,
                    "safety_ok": bool(safety_ok),
                }
            )
    if not rows:
        return {
            "available": False,
            "reason": "empty_threshold_search",
            "thresholds": defaults,
            "search": [],
        }
    search_df = pd.DataFrame(rows)
    safe_df = search_df.loc[search_df["safety_ok"]].copy()
    if safe_df.empty:
        best = search_df.sort_values(
            by=["margin", "score", "event_rate", "far_rate", "confound_ratio"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        safety_respected = False
    else:
        best = safe_df.sort_values(
            by=["score", "margin", "event_rate", "far_rate", "confound_ratio"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        safety_respected = True
    chosen = {
        "odd_inner_outer_min": float(best["odd_inner_outer_min"]),
        "tangential_radial_min": float(best["tangential_radial_min"]),
    }
    return {
        "available": True,
        "reason": "ok",
        "thresholds": chosen,
        "defaults": defaults,
        "safety_respected": bool(safety_respected),
        "best_metrics": {
            "event_rate": float(best["event_rate"]),
            "far_rate": float(best["far_rate"]),
            "margin": float(best["margin"]),
            "confound_ratio": float(best.get("confound_ratio", np.nan)),
            "event_floor_penalty": float(best.get("event_floor_penalty", 0.0)),
            "score": float(best["score"]),
        },
        "objective": {
            "event_min_floor": float(event_min_floor),
            "constrain_min_thresholds": bool(constrain_min_thresholds),
            "score_formula": "margin - 0.50*far_rate - 0.20*confound_ratio - 2.00*event_floor_penalty + 0.10*event_rate",
        },
        "search": search_df.sort_values(by=["score", "margin"], ascending=[False, False]).head(200).to_dict(orient="records"),
    }


def _parity_feature_distribution_table(case_metrics: pd.DataFrame) -> pd.DataFrame:
    if case_metrics is None or case_metrics.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "split_stratum",
                "lead_bucket",
                "parity_signal_pass",
                "odd_energy_inner_outer_ratio",
                "tangential_radial_ratio",
            ]
        )
    out = case_metrics.copy()
    cols = [
        c
        for c in [
            "case_id",
            "split_stratum",
            "lead_bucket",
            "case_type",
            "control_tier",
            "center_source",
            "parity_signal_pass",
            "odd_energy_inner_outer_ratio",
            "tangential_radial_ratio",
        ]
        if c in out.columns
    ]
    out = out.loc[:, cols].copy()
    if "split_stratum" not in out.columns:
        out["split_stratum"] = _derive_strata_labels(out).astype(str)
    return out


def _build_mode_invariant_claim_gate(
    *,
    selection_df: pd.DataFrame,
    event_minus_far_min: float,
    far_max: float,
) -> dict[str, Any]:
    if selection_df is None or selection_df.empty:
        return {
            "passed": False,
            "event_rate_all_modes": 0.0,
            "far_rate_all_modes": 0.0,
            "margin_all_modes": 0.0,
            "event_rate_any_modes": 0.0,
            "far_rate_any_modes": 0.0,
            "margin_any_modes": 0.0,
            "thresholds": {
                "event_minus_far_min": float(event_minus_far_min),
                "far_max": float(far_max),
            },
        }
    df = selection_df.copy()
    df["case_type"] = df.get("case_type", pd.Series(index=df.index)).fillna("unknown").astype(str)
    event_mask = _event_mask_from_frame(df)
    far_mask = ~event_mask
    all_modes = df.get("parity_pass_all_modes", pd.Series(False, index=df.index)).astype(bool)
    any_modes = df.get("parity_pass_any_modes", pd.Series(False, index=df.index)).astype(bool)
    event_all = float(all_modes.loc[event_mask].mean()) if bool(event_mask.any()) else 0.0
    far_all = float(all_modes.loc[far_mask].mean()) if bool(far_mask.any()) else 0.0
    event_any = float(any_modes.loc[event_mask].mean()) if bool(event_mask.any()) else 0.0
    far_any = float(any_modes.loc[far_mask].mean()) if bool(far_mask.any()) else 0.0
    margin_all = float(event_all - far_all)
    margin_any = float(event_any - far_any)
    passed = bool((event_all > 0.0) and (margin_all >= float(event_minus_far_min)) and (far_all <= float(far_max)))
    return {
        "passed": passed,
        "event_rate_all_modes": event_all,
        "far_rate_all_modes": far_all,
        "margin_all_modes": margin_all,
        "event_rate_any_modes": event_any,
        "far_rate_any_modes": far_any,
        "margin_any_modes": margin_any,
        "thresholds": {
            "event_minus_far_min": float(event_minus_far_min),
            "far_max": float(far_max),
        },
    }


def _build_mode_margin_gate(
    *,
    anomaly_ablation: dict[str, Any],
    event_minus_far_min: float,
    far_max: float,
) -> dict[str, Any]:
    modes = (anomaly_ablation or {}).get("modes", {}) or {}
    checks: dict[str, Any] = {}
    pass_all = True
    n_avail = 0
    for mode, rec in modes.items():
        if not isinstance(rec, dict) or not bool(rec.get("available", False)):
            continue
        n_avail += 1
        margin = _to_float(rec.get("event_minus_far_parity_rate")) or 0.0
        far = _to_float(rec.get("far_nonstorm_parity_signal_rate")) or 0.0
        ok = bool((margin >= float(event_minus_far_min)) and (far <= float(far_max)))
        checks[str(mode)] = {
            "margin": float(margin),
            "far_rate": float(far),
            "passed": bool(ok),
        }
        pass_all = pass_all and ok
    return {
        "passed": bool((n_avail > 0) and pass_all),
        "n_modes_available": int(n_avail),
        "thresholds": {
            "event_minus_far_min": float(event_minus_far_min),
            "far_max": float(far_max),
        },
        "by_mode": checks,
    }


def _safe_js_divergence(p: np.ndarray, q: np.ndarray) -> float | None:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return None
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps = float(p.sum())
    qs = float(q.sum())
    if ps <= 0.0 or qs <= 0.0:
        return None
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = float(np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0)))
    kl_qm = float(np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0)))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def _build_anomaly_mode_distribution_audit(selection_df: pd.DataFrame) -> dict[str, Any]:
    if selection_df is None or selection_df.empty:
        return {"n_cases": 0}
    df = selection_df.copy()
    df["case_type"] = df.get("case_type", pd.Series(index=df.index)).fillna("unknown").astype(str)
    df["canonical_mode"] = df.get("canonical_mode", pd.Series(index=df.index)).fillna("unknown").astype(str)
    event_mask = _event_mask_from_frame(df)
    event = df.loc[event_mask].copy()
    far = df.loc[~event_mask].copy()
    modes = sorted(df["canonical_mode"].dropna().unique().tolist())
    if not modes:
        return {"n_cases": int(df.shape[0])}
    event_counts = event["canonical_mode"].value_counts().reindex(modes, fill_value=0).to_numpy(dtype=float)
    far_counts = far["canonical_mode"].value_counts().reindex(modes, fill_value=0).to_numpy(dtype=float)
    event_prob = (event_counts / max(1.0, float(event_counts.sum()))).tolist()
    far_prob = (far_counts / max(1.0, float(far_counts.sum()))).tolist()
    js = _safe_js_divergence(np.asarray(event_prob), np.asarray(far_prob))
    return {
        "n_cases": int(df.shape[0]),
        "n_event_cases": int(event.shape[0]),
        "n_far_cases": int(far.shape[0]),
        "modes": modes,
        "event_mode_prob": {m: float(p) for m, p in zip(modes, event_prob)},
        "far_mode_prob": {m: float(p) for m, p in zip(modes, far_prob)},
        "event_far_mode_js_divergence": js,
    }


def _build_geometry_null_audit(
    *,
    samples: pd.DataFrame,
    transforms: dict[str, NullTransform],
    seed: int,
) -> dict[str, Any]:
    if samples is None or samples.empty:
        return {"n_rows": 0, "transforms": {}}
    families = {
        "active": [c for c in ("O",) if c in samples.columns],
        "scalar": [c for c in ("O_raw", "O_scalar", "O_lat_hour", "O_lat_day", "O_vorticity") if c in samples.columns],
        "vector": [c for c in ("O_vector", "O_local_frame", "O_meanflow") if c in samples.columns],
        "polar": [c for c in samples.columns if str(c).startswith("O_polar_")],
    }
    out: dict[str, Any] = {"n_rows": int(samples.shape[0]), "transforms": {}}
    base = samples.reset_index(drop=True).copy()
    for i, (name, transform) in enumerate(transforms.items()):
        tdf = transform(base.copy(), rng=np.random.default_rng(int(seed) + i)).reset_index(drop=True)
        rec: dict[str, Any] = {"n_rows": int(tdf.shape[0]), "feature_delta_by_family": {}, "invariant_fraction_by_family": {}}
        for fam, cols in families.items():
            cols = [c for c in cols if c in base.columns and c in tdf.columns]
            if not cols:
                rec["feature_delta_by_family"][fam] = None
                rec["invariant_fraction_by_family"][fam] = None
                continue
            deltas: list[np.ndarray] = []
            invariants: list[float] = []
            for col in cols:
                b = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
                t = pd.to_numeric(tdf[col], errors="coerce").to_numpy(dtype=float)
                n = min(b.size, t.size)
                if n <= 0:
                    continue
                d = np.abs(b[:n] - t[:n])
                mask = np.isfinite(d)
                if not np.any(mask):
                    continue
                dv = d[mask]
                deltas.append(dv)
                invariants.append(float(np.mean(dv <= 1e-12)))
            if deltas:
                all_d = np.concatenate(deltas)
                rec["feature_delta_by_family"][fam] = {
                    "mean_abs_delta": float(np.mean(all_d)),
                    "median_abs_delta": float(np.median(all_d)),
                }
                rec["invariant_fraction_by_family"][fam] = float(np.mean(invariants))
            else:
                rec["feature_delta_by_family"][fam] = None
                rec["invariant_fraction_by_family"][fam] = None
        out["transforms"][str(name)] = rec
    return out


def _build_geometry_null_by_mode(
    *,
    samples: pd.DataFrame,
    seed: int,
    modes: list[str] | None = None,
    null_names: list[str] | None = None,
) -> dict[str, Any]:
    if samples is None or samples.empty:
        return {"modes": {}}
    mode_list = modes or ["none", "lat_day", "lat_hour", "scalars_only", "polar_chiral"]
    chosen_nulls = null_names or [
        "theta_roll",
        "theta_scramble_within_ring",
        "center_swap",
        "storm_track_reassignment",
        "time_permutation_within_lead",
        "lat_band_shuffle_within_month_hour",
        "center_jitter",
        "mirror_axis_jitter",
        "radial_scramble",
    ]
    out: dict[str, Any] = {"modes": {}}
    for i, mode in enumerate(mode_list):
        col = _mode_to_column(str(mode))
        if col is None or col not in samples.columns:
            continue
        df = samples.copy()
        df["O"] = pd.to_numeric(df[col], errors="coerce")
        df = df.loc[np.isfinite(df["O"]) & (df["O"] > 0)].copy()
        obs_rank = _compute_case_eta_scores(df)
        if obs_rank.empty:
            continue
        mask = _event_mask_from_frame(obs_rank)
        far_vals = pd.to_numeric(obs_rank.loc[~mask, "eta_score"], errors="coerce").to_numpy(dtype=float)
        far_vals = far_vals[np.isfinite(far_vals)]
        if far_vals.size > 0:
            med = float(np.median(far_vals))
            mad = float(np.median(np.abs(far_vals - med)))
            thr = med + max(mad, 1e-8)
        else:
            all_vals = pd.to_numeric(obs_rank["eta_score"], errors="coerce").to_numpy(dtype=float)
            all_vals = all_vals[np.isfinite(all_vals)]
            thr = float(np.quantile(all_vals, 0.75)) if all_vals.size > 0 else 0.0

        def _rates(rank: pd.DataFrame) -> tuple[float, float]:
            if rank.empty:
                return 0.0, 0.0
            ev = _event_mask_from_frame(rank)
            fv = ~ev
            sc = pd.to_numeric(rank.get("eta_score"), errors="coerce")
            ev_rate = float((sc.loc[ev] >= thr).mean()) if bool(ev.any()) else 0.0
            far_rate = float((sc.loc[fv] >= thr).mean()) if bool(fv.any()) else 0.0
            return ev_rate, far_rate

        obs_event, obs_far = _rates(obs_rank)
        mode_row: dict[str, Any] = {
            "event_rate_observed": float(obs_event),
            "far_rate_observed": float(obs_far),
        }
        for j, null_name in enumerate(chosen_nulls):
            transform = _geometry_null_transform_by_name(str(null_name))
            if transform is None:
                continue
            null_rank = _compute_case_eta_scores(transform(df, rng=np.random.default_rng(int(seed) + 100 + i * 17 + j)))
            null_event, null_far = _rates(null_rank)
            mode_row[str(null_name)] = {
                "event_rate_null": float(null_event),
                "far_rate_null": float(null_far),
                "drop_relative": float((obs_event - null_event) / max(obs_event, 1e-9)),
                "drop_absolute": float(obs_event - null_event),
            }
        out["modes"][str(mode)] = mode_row
    return out


def _is_vortex_centered_context(*, samples: pd.DataFrame) -> bool:
    if samples is None or samples.empty:
        return False
    has_polar = any(str(c).startswith("O_polar_") for c in samples.columns)
    if not has_polar:
        return False
    if "center_source" not in samples.columns:
        return True
    src = samples["center_source"].dropna().astype(str).str.lower()
    if src.empty:
        return has_polar
    return bool(src.isin({"vortex", "hybrid", "ibtracs", "labels"}).any())


def _compute_case_angular_scores(samples: pd.DataFrame) -> pd.DataFrame:
    if samples is None or samples.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "track_group",
                "case_type",
                "storm_max",
                "near_storm_max",
                "pregen_max",
                "storm_distance_cohort",
                "ang_coh",
                "harmonic_score",
                "chiral_score",
                "odd_score",
                "eta_score",
                "tangential_dominance",
                "ring_witness",
                "witness_score",
            ]
        )
    req = {"case_id", "hand"}
    if not req.issubset(set(samples.columns)):
        return pd.DataFrame(
            columns=[
                "case_id",
                "track_group",
                "case_type",
                "storm_max",
                "near_storm_max",
                "pregen_max",
                "storm_distance_cohort",
                "ang_coh",
                "harmonic_score",
                "chiral_score",
                "odd_score",
                "eta_score",
                "tangential_dominance",
                "ring_witness",
                "witness_score",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for case_id, grp in samples.groupby("case_id", dropna=False):
        storm_series = pd.to_numeric(grp["storm"], errors="coerce").fillna(0) if "storm" in grp.columns else pd.Series(0.0, index=grp.index)
        near_series = pd.to_numeric(grp["near_storm"], errors="coerce").fillna(0) if "near_storm" in grp.columns else pd.Series(0.0, index=grp.index)
        pregen_series = pd.to_numeric(grp["pregen"], errors="coerce").fillna(0) if "pregen" in grp.columns else pd.Series(0.0, index=grp.index)
        left = pd.to_numeric(grp.get("O_polar_left"), errors="coerce").to_numpy(dtype=float) if "O_polar_left" in grp.columns else np.array([], dtype=float)
        right = pd.to_numeric(grp.get("O_polar_right"), errors="coerce").to_numpy(dtype=float) if "O_polar_right" in grp.columns else np.array([], dtype=float)
        chiral = pd.to_numeric(grp.get("O_polar_chiral"), errors="coerce").to_numpy(dtype=float) if "O_polar_chiral" in grp.columns else np.array([], dtype=float)
        odd = pd.to_numeric(grp.get("O_polar_odd_ratio"), errors="coerce").to_numpy(dtype=float) if "O_polar_odd_ratio" in grp.columns else np.array([], dtype=float)
        eta = pd.to_numeric(grp.get("O_polar_eta"), errors="coerce").to_numpy(dtype=float) if "O_polar_eta" in grp.columns else np.array([], dtype=float)
        vt = pd.to_numeric(grp.get("O_polar_tangential"), errors="coerce").to_numpy(dtype=float) if "O_polar_tangential" in grp.columns else np.array([], dtype=float)
        vr = pd.to_numeric(grp.get("O_polar_radial"), errors="coerce").to_numpy(dtype=float) if "O_polar_radial" in grp.columns else np.array([], dtype=float)
        harmonic = np.nan
        if left.size > 0 and right.size > 0:
            n = min(left.size, right.size)
            den = np.abs(left[:n]) + np.abs(right[:n]) + 1e-8
            harmonic = float(np.nanmedian(np.abs(left[:n] - right[:n]) / den))
        chiral_score = float(np.nanmedian(np.abs(chiral))) if chiral.size > 0 else np.nan
        odd_score = float(np.nanmedian(np.abs(odd))) if odd.size > 0 else np.nan
        eta_score = float(np.nanmedian(np.abs(eta))) if eta.size > 0 else np.nan
        tan_dom = np.nan
        if vt.size > 0 and vr.size > 0:
            n = min(vt.size, vr.size)
            denom = np.abs(vr[:n]) + 1e-8
            tan_dom = float(np.nanmedian(np.abs(vt[:n]) / denom))
        ring_witness = np.nan
        if ("L" in grp.columns) and vt.size > 0 and vr.size > 0:
            lvals = pd.to_numeric(grp["L"], errors="coerce").to_numpy(dtype=float)
            n = min(vt.size, vr.size, lvals.size)
            lvals = lvals[:n]
            ratio = np.abs(vt[:n]) / (np.abs(vr[:n]) + 1e-8)
            valid = np.isfinite(lvals) & np.isfinite(ratio)
            if np.any(valid):
                ring_series = pd.DataFrame({"L": lvals[valid], "ratio": ratio[valid]}).groupby("L")["ratio"].median()
                rv = ring_series.to_numpy(dtype=float)
                if rv.size >= 2:
                    ring_witness = float(np.nanmedian(rv) - np.nanstd(rv))
                elif rv.size == 1:
                    ring_witness = float(rv[0])
        vals = np.asarray([harmonic, chiral_score, odd_score, eta_score, tan_dom, ring_witness], dtype=float)
        vals = vals[np.isfinite(vals)]
        ang_coh = float(np.nanmedian(vals)) if vals.size > 0 else np.nan
        hs = harmonic if np.isfinite(harmonic) else np.nan
        os = odd_score if np.isfinite(odd_score) else np.nan
        td = tan_dom if np.isfinite(tan_dom) else np.nan
        rw = ring_witness if np.isfinite(ring_witness) else np.nan
        if np.isfinite(hs) and np.isfinite(os) and np.isfinite(td) and np.isfinite(rw):
            witness_score = float(hs + 0.35 * td + 0.20 * rw - 0.2 * os)
        elif np.isfinite(hs) and np.isfinite(os) and np.isfinite(td):
            # Storm-centric witness favors angular asymmetry plus tangential dominance.
            witness_score = float(hs + 0.35 * td - 0.2 * os)
        elif np.isfinite(hs) and np.isfinite(os):
            witness_score = float(hs - 0.2 * os)
        elif np.isfinite(hs) and np.isfinite(td):
            witness_score = float(hs + 0.35 * td)
        elif np.isfinite(hs):
            witness_score = float(hs)
        elif np.isfinite(os):
            witness_score = float(-0.2 * os)
        elif np.isfinite(td):
            witness_score = float(0.35 * td)
        else:
            witness_score = np.nan
        out_rows.append(
            {
                "case_id": str(case_id),
                "track_group": str(
                    next(
                        (
                            grp[c].dropna().astype(str).iloc[0]
                            for c in ("vortex_track_id", "track_id", "storm_id")
                            if c in grp.columns and grp[c].dropna().shape[0] > 0
                        ),
                        str(case_id),
                    )
                ),
                "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown",
                "storm_max": int(storm_series.max()),
                "near_storm_max": int(near_series.max()),
                "pregen_max": int(pregen_series.max()),
                "storm_distance_cohort": str(grp["storm_distance_cohort"].iloc[0]) if "storm_distance_cohort" in grp.columns and grp["storm_distance_cohort"].notna().any() else "",
                "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else "",
                "ang_coh": ang_coh,
                "harmonic_score": harmonic if np.isfinite(harmonic) else None,
                "chiral_score": chiral_score if np.isfinite(chiral_score) else None,
                "odd_score": odd_score if np.isfinite(odd_score) else None,
                "eta_score": eta_score if np.isfinite(eta_score) else None,
                "tangential_dominance": tan_dom if np.isfinite(tan_dom) else None,
                "ring_witness": ring_witness if np.isfinite(ring_witness) else None,
                "witness_score": witness_score if np.isfinite(witness_score) else None,
            }
        )
    return pd.DataFrame(out_rows)


def _cohen_d(event_vals: np.ndarray, far_vals: np.ndarray) -> float:
    if event_vals.size < 2 or far_vals.size < 2:
        return 0.0
    v1 = float(np.var(event_vals, ddof=1))
    v0 = float(np.var(far_vals, ddof=1))
    pooled = float(np.sqrt(((event_vals.size - 1) * v1 + (far_vals.size - 1) * v0) / max(event_vals.size + far_vals.size - 2, 1)))
    if not np.isfinite(pooled) or pooled <= 0.0:
        return 0.0
    return float((float(np.mean(event_vals)) - float(np.mean(far_vals))) / pooled)


def _permutation_pvalue_margin(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return 1.0
    x = scores[valid].astype(float)
    y = labels[valid].astype(int)
    g = groups[valid].astype(str)
    if x.size < 4 or np.unique(y).size < 2:
        return 1.0
    obs = float(np.mean(x[y == 1]) - np.mean(x[y == 0]))
    ge = 0
    total = 0
    for _ in range(int(max(1, n_perm))):
        yp = y.copy()
        for gg in np.unique(g):
            idx = np.where(g == gg)[0]
            if idx.size > 1:
                yp[idx] = yp[rng.permutation(idx)]
        if np.unique(yp).size < 2:
            continue
        stat = float(np.mean(x[yp == 1]) - np.mean(x[yp == 0]))
        if stat >= obs:
            ge += 1
        total += 1
    if total <= 0:
        return 1.0
    return float((ge + 1) / (total + 1))


def _build_angular_witness_report(
    *,
    observed_samples: pd.DataFrame,
    theta_roll_samples: pd.DataFrame,
    center_swap_samples: pd.DataFrame | None = None,
    center_jitter_samples: pd.DataFrame | None = None,
    margin_min: float,
    d_min: float,
    p_max: float,
    null_drop_min: float,
    null_abs_drop_min: float,
    null_margin_max: float,
    permutation_n: int,
    required: bool,
) -> dict[str, Any]:
    obs = _compute_case_angular_scores(observed_samples)
    theta = _compute_case_angular_scores(theta_roll_samples)
    center_swap_required = center_swap_samples is not None
    center_swap_input = center_swap_samples if center_swap_required else theta_roll_samples
    center_swap = _compute_case_angular_scores(center_swap_input)
    center_jitter = _compute_case_angular_scores(center_jitter_samples) if center_jitter_samples is not None else pd.DataFrame()
    if obs.empty:
        return {"gate": {"required": bool(required), "passed": False, "reason": "no_polar_scores"}}

    obs_scores_raw = pd.to_numeric(obs.get("witness_score"), errors="coerce").to_numpy(dtype=float)
    obs_scores_raw = obs_scores_raw[np.isfinite(obs_scores_raw)]
    center_ref = float(np.median(obs_scores_raw)) if obs_scores_raw.size > 0 else 0.0
    mad_ref = float(np.median(np.abs(obs_scores_raw - center_ref))) if obs_scores_raw.size > 0 else 0.0
    sd_ref = float(np.std(obs_scores_raw, ddof=1)) if obs_scores_raw.size > 1 else 0.0
    scale_ref = float(max(mad_ref, 0.25 * sd_ref, 1e-6))

    def _to_group_table(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["group_id", "is_event", "lead_bucket", "score"])
        ev = _event_mask_from_frame(df).to_numpy(dtype=bool)
        s_raw = pd.to_numeric(df.get("witness_score"), errors="coerce").to_numpy(dtype=float)
        s = (s_raw - center_ref) / max(scale_ref, 1e-9)
        group_id = (
            df.get("track_group", df.get("case_id", pd.Series("", index=df.index)))
            .fillna(df.get("case_id", pd.Series("", index=df.index)))
            .astype(str)
            .to_numpy(dtype=object)
        )
        lead = df.get("lead_bucket", pd.Series("", index=df.index)).fillna("").astype(str).to_numpy(dtype=object)
        tmp = pd.DataFrame({"group_id": group_id, "is_event": ev.astype(int), "lead_bucket": lead, "score": s})
        tmp = tmp.loc[np.isfinite(pd.to_numeric(tmp["score"], errors="coerce"))].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["group_id", "is_event", "lead_bucket", "score"])
        grp = (
            tmp.groupby(["group_id", "is_event", "lead_bucket"], dropna=False)["score"]
            .median()
            .reset_index()
        )
        return grp

    def _summary(group_df: pd.DataFrame) -> dict[str, float]:
        if group_df is None or group_df.empty:
            return {
                "event_mean": 0.0,
                "far_mean": 0.0,
                "margin": 0.0,
                "event_rate": 0.0,
                "far_rate": 0.0,
                "n_event_groups": 0,
                "n_far_groups": 0,
            }
        ev = pd.to_numeric(group_df.get("is_event"), errors="coerce").fillna(0).astype(int) == 1
        s = pd.to_numeric(group_df.get("score"), errors="coerce").to_numpy(dtype=float)
        se = s[ev.to_numpy(dtype=bool)] if bool(ev.any()) else np.array([], dtype=float)
        sf = s[(~ev).to_numpy(dtype=bool)] if bool((~ev).any()) else np.array([], dtype=float)
        se = se[np.isfinite(se)]
        sf = sf[np.isfinite(sf)]
        event_mean = float(np.nanmean(se)) if se.size > 0 else 0.0
        far_mean = float(np.nanmean(sf)) if sf.size > 0 else 0.0
        return {
            "event_mean": event_mean,
            "far_mean": far_mean,
            "margin": float(event_mean - far_mean),
            "event_rate": float(np.mean(se > 0.0)) if se.size > 0 else 0.0,
            "far_rate": float(np.mean(sf > 0.0)) if sf.size > 0 else 0.0,
            "n_event_groups": int(se.size),
            "n_far_groups": int(sf.size),
        }

    obs_group = _to_group_table(obs)
    theta_group = _to_group_table(theta)
    center_swap_group = _to_group_table(center_swap)
    center_jitter_group = _to_group_table(center_jitter)

    obs_sum = _summary(obs_group)
    theta_sum = _summary(theta_group)
    center_swap_sum = _summary(center_swap_group)
    center_jitter_sum = _summary(center_jitter_group)

    g_scores = pd.to_numeric(obs_group.get("score"), errors="coerce").to_numpy(dtype=float)
    g_labels = pd.to_numeric(obs_group.get("is_event"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    g_groups = obs_group.get("lead_bucket", pd.Series("", index=obs_group.index)).astype(str).to_numpy(dtype=object)
    ev_vals = g_scores[g_labels == 1]
    far_vals = g_scores[g_labels == 0]
    ev_vals = ev_vals[np.isfinite(ev_vals)]
    far_vals = far_vals[np.isfinite(far_vals)]
    d_obs = _cohen_d(ev_vals, far_vals)
    p_obs = _permutation_pvalue_margin(
        scores=g_scores,
        labels=g_labels,
        groups=g_groups,
        rng=np.random.default_rng(91013),
        n_perm=int(permutation_n),
    )

    margin_obs = float(obs_sum["margin"])
    margin_theta = float(theta_sum["margin"])
    margin_center_swap = float(center_swap_sum["margin"])
    drop_theta_abs = float(margin_obs - margin_theta)
    drop_theta_rel = float(drop_theta_abs / max(abs(margin_obs), 1e-9))
    drop_center_swap_abs = float(margin_obs - margin_center_swap)
    drop_center_swap_rel = float(drop_center_swap_abs / max(abs(margin_obs), 1e-9))
    margin_center_jitter = float(center_jitter_sum["margin"])
    drop_center_jitter_abs = float(margin_obs - margin_center_jitter)
    drop_center_jitter_rel = float(drop_center_jitter_abs / max(abs(margin_obs), 1e-9))
    if not required:
        gate_passed = True
        reason = "not_required"
    else:
        gate_passed = bool(
            (margin_obs >= float(margin_min))
            and (d_obs >= float(d_min))
            and (p_obs <= float(p_max))
            and (margin_theta <= float(null_margin_max))
            and (drop_theta_rel >= float(null_drop_min))
            and (drop_theta_abs >= float(null_abs_drop_min))
            and (
                (not center_swap_required)
                or (
                    (margin_center_swap <= float(null_margin_max))
                    and (drop_center_swap_rel >= float(null_drop_min))
                    and (drop_center_swap_abs >= float(null_abs_drop_min))
                )
            )
        )
        reason = "ok" if gate_passed else "threshold_fail"
    report: dict[str, Any] = {
        "observed": {
            "event_mean": float(obs_sum["event_mean"]),
            "far_mean": float(obs_sum["far_mean"]),
            "margin": float(margin_obs),
            "event_rate": float(obs_sum["event_rate"]),
            "far_rate": float(obs_sum["far_rate"]),
            "n_event_groups": int(obs_sum.get("n_event_groups", 0)),
            "n_far_groups": int(obs_sum.get("n_far_groups", 0)),
            "effect_size_d": float(d_obs),
            "p_value_perm": float(p_obs),
        },
        "theta_roll": {
            "event_mean": float(theta_sum["event_mean"]),
            "far_mean": float(theta_sum["far_mean"]),
            "margin": float(margin_theta),
            "event_rate": float(theta_sum["event_rate"]),
            "far_rate": float(theta_sum["far_rate"]),
            "drop_abs_event": drop_theta_abs,
            "drop_rel_event": drop_theta_rel,
        },
        "center_swap": {
            "required": bool(center_swap_required),
            "event_mean": float(center_swap_sum["event_mean"]),
            "far_mean": float(center_swap_sum["far_mean"]),
            "margin": float(margin_center_swap),
            "event_rate": float(center_swap_sum["event_rate"]),
            "far_rate": float(center_swap_sum["far_rate"]),
            "drop_abs_event": drop_center_swap_abs,
            "drop_rel_event": drop_center_swap_rel,
        },
        "gate": {
            "required": bool(required),
            "passed": bool(gate_passed),
            "reason": reason,
            "margin_min": float(margin_min),
            "d_min": float(d_min),
            "p_max": float(p_max),
            "null_drop_min": float(null_drop_min),
            "null_abs_drop_min": float(null_abs_drop_min),
            "null_margin_max": float(null_margin_max),
            "permutation_n": int(permutation_n),
        },
        "normalization": {
            "center_ref": float(center_ref),
            "scale_ref": float(scale_ref),
            "mad_ref": float(mad_ref),
            "std_ref": float(sd_ref),
        },
    }
    if center_jitter_samples is not None:
        report["center_jitter"] = {
            "event_mean": float(center_jitter_sum["event_mean"]),
            "far_mean": float(center_jitter_sum["far_mean"]),
            "margin": float(margin_center_jitter),
            "event_rate": float(center_jitter_sum["event_rate"]),
            "far_rate": float(center_jitter_sum["far_rate"]),
            "drop_abs_event": drop_center_jitter_abs,
            "drop_rel_event": drop_center_jitter_rel,
        }
    return report


def _assemble_case_metrics_from_mode_selection(
    *,
    mode_case_metrics: dict[str, pd.DataFrame],
    selection_df: pd.DataFrame,
    mode_col: str,
    decision_col: str,
) -> pd.DataFrame:
    if selection_df is None or selection_df.empty or not mode_case_metrics:
        return pd.DataFrame()

    mode_tables: dict[str, pd.DataFrame] = {}
    for mode, df in mode_case_metrics.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["case_id"] = tmp["case_id"].astype(str)
        tmp = tmp.drop_duplicates(subset=["case_id"], keep="first").set_index("case_id", drop=False)
        mode_tables[str(mode)] = tmp

    rows: list[dict[str, Any]] = []
    for _, sel in selection_df.iterrows():
        cid = str(sel.get("case_id"))
        mode = str(sel.get(mode_col, ""))
        chosen = mode_tables.get(mode)
        rec: pd.Series | None = None
        if chosen is not None and cid in chosen.index:
            got = chosen.loc[cid]
            rec = got.iloc[0] if isinstance(got, pd.DataFrame) else got
        if rec is None:
            for alt_mode, alt_df in mode_tables.items():
                if cid in alt_df.index:
                    got = alt_df.loc[cid]
                    rec = got.iloc[0] if isinstance(got, pd.DataFrame) else got
                    mode = alt_mode
                    break
        if rec is None:
            continue
        row = dict(rec.to_dict())
        row["case_id"] = cid
        row["anomaly_mode_selected"] = str(mode)
        if decision_col in sel:
            row["parity_signal_pass_effective"] = bool(sel[decision_col])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_knee_strategy_summary(case_metrics: pd.DataFrame) -> dict[str, Any]:
    if case_metrics is None or case_metrics.empty:
        return {
            "n_cases": 0,
            "size_knee_rate": 0.0,
            "tf_knee_rate": 0.0,
            "effective_knee_rate": 0.0,
            "recommended_primary": "tf",
            "size_knee_role": "secondary_conditional",
            "size_knee_claimable": False,
            "notes": ["no_cases"],
        }
    size = (
        pd.to_numeric(case_metrics.get("knee_size_detected", case_metrics.get("knee_detected")), errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    tf = (
        pd.to_numeric(case_metrics.get("knee_tf_detected", case_metrics.get("tf_knee_detected")), errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    eff = (
        pd.to_numeric(case_metrics.get("knee_detected_effective", case_metrics.get("knee_detected")), errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    n = int(case_metrics.shape[0])
    size_rate = float(size.mean()) if n else 0.0
    tf_rate = float(tf.mean()) if n else 0.0
    eff_rate = float(eff.mean()) if n else 0.0
    both = int((size & tf).sum())
    any_knee = int((size | tf).sum())
    cand_prop = pd.to_numeric(case_metrics.get("knee_candidate_count_proposed"), errors="coerce")
    cand_sanity = pd.to_numeric(case_metrics.get("knee_candidate_count_sanity_pass"), errors="coerce")
    size_candidate_rate = float((cand_prop.fillna(0) > 0).mean()) if not cand_prop.empty else 0.0
    size_sanity_rate = float((cand_sanity.fillna(0) > 0).mean()) if not cand_sanity.empty else 0.0
    # Weather policy: TF-knee is primary; size-knee is only claimable when curve support is sufficiently stable.
    size_claimable = bool(
        (size_candidate_rate >= 0.50)
        and (size_sanity_rate >= 0.30)
        and (size_rate > 0.0)
    )
    notes: list[str] = []
    notes.append("weather_policy_tf_primary")
    if not size_claimable:
        notes.append("size_knee_diagnostic_only_under_current_scale_support")
    if tf_rate <= 0.0:
        notes.append("tf_knee_not_detected")
    return {
        "n_cases": n,
        "size_knee_rate": size_rate,
        "tf_knee_rate": tf_rate,
        "effective_knee_rate": eff_rate,
        "both_knees_rate": float(both / n) if n else 0.0,
        "knee_agreement_rate": float(both / max(1, any_knee)),
        "recommended_primary": "tf",
        "size_knee_role": "secondary_conditional",
        "size_knee_claimable": bool(size_claimable),
        "size_candidate_rate": float(size_candidate_rate),
        "size_sanity_rate": float(size_sanity_rate),
        "notes": notes,
    }


def _build_parity_confound_dashboard(
    *,
    case_metrics: pd.DataFrame,
    samples: pd.DataFrame,
    case_meta: pd.DataFrame,
    axis_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if samples.empty:
        return pd.DataFrame(), {"n_cases": 0}
    metrics_by_case = {}
    if not case_metrics.empty and "case_id" in case_metrics.columns:
        metrics_by_case = {
            str(r["case_id"]): r
            for _, r in case_metrics.drop_duplicates(subset=["case_id"], keep="first").iterrows()
        }

    rows: list[dict[str, Any]] = []
    for case_id, grp in samples.groupby("case_id", dropna=False):
        cid = str(case_id)
        pivot = grp.pivot_table(index=["t", "L"], columns="hand", values="O", aggfunc="first")
        if "L" not in pivot.columns or "R" not in pivot.columns:
            continue
        o_l = pd.to_numeric(pivot["L"], errors="coerce").to_numpy(dtype=float)
        o_r = pd.to_numeric(pivot["R"], errors="coerce").to_numpy(dtype=float)
        denom = 0.5 * (o_l + o_r)
        valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
        if not np.any(valid):
            continue
        eta = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
        sgn = np.sign(o_l[valid] - o_r[valid])

        try:
            ks = ks_2samp(o_l[valid], o_r[valid], method="asymp")
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        except Exception:
            ks_stat = None
            ks_p = None

        t_vals = pd.to_datetime(grp["t"], errors="coerce")
        t_med = pd.Timestamp(t_vals.dropna().median()) if t_vals.notna().any() else pd.NaT
        lat_mean = _to_float(pd.to_numeric(grp.get("lat0"), errors="coerce").mean())
        lon_mean = _to_float(pd.to_numeric(grp.get("lon0"), errors="coerce").mean())
        lat_bin = float(np.round(lat_mean / 2.0) * 2.0) if lat_mean is not None else None
        rec = metrics_by_case.get(cid, {})
        rows.append(
            {
                "case_id": cid,
                "case_type": str(grp["case_type"].iloc[0]) if "case_type" in grp.columns else "unknown",
                "center_source": str(grp["center_source"].iloc[0]) if "center_source" in grp.columns and grp["center_source"].notna().any() else None,
                "control_tier": str(grp["control_tier"].iloc[0]) if "control_tier" in grp.columns and grp["control_tier"].notna().any() else None,
                "match_quality": str(grp["match_quality"].iloc[0]) if "match_quality" in grp.columns and grp["match_quality"].notna().any() else None,
                "storm_distance_cohort": str(grp["storm_distance_cohort"].iloc[0]) if "storm_distance_cohort" in grp.columns and grp["storm_distance_cohort"].notna().any() else None,
                "lead_bucket": str(grp["lead_bucket"].iloc[0]) if "lead_bucket" in grp.columns else None,
                "anomaly_mode_effective": str(grp["anomaly_mode_effective"].iloc[0]) if "anomaly_mode_effective" in grp.columns and grp["anomaly_mode_effective"].notna().any() else None,
                "anomaly_mode_requested": str(grp["anomaly_mode_requested"].iloc[0]) if "anomaly_mode_requested" in grp.columns and grp["anomaly_mode_requested"].notna().any() else None,
                "n_pairs": int(np.sum(valid)),
                "eta_mean": float(np.mean(eta)),
                "eta_abs_mean": float(np.mean(np.abs(eta))),
                "eta_sign_consistency": float(abs(np.mean(sgn))),
                "parity_signal_pass": bool(rec.get("parity_signal_pass", False)),
                "parity_signal_pass_effective": bool(rec.get("parity_signal_pass_effective", rec.get("parity_signal_pass", False))),
                "parity_localization_pass": bool(rec.get("parity_localization_pass", False)),
                "parity_tangential_pass": bool(rec.get("parity_tangential_pass", False)),
                "odd_energy_inner_outer_ratio": _to_float(rec.get("odd_energy_inner_outer_ratio")),
                "tangential_radial_ratio": _to_float(rec.get("tangential_radial_ratio")),
                "knee_size_detected": bool(rec.get("knee_detected", False)),
                "knee_tf_detected": bool(rec.get("tf_knee_detected", False)),
                "knee_tf_null_gate_pass": bool(rec.get("tf_knee_null_gate_pass", False)),
                "lat_mean": lat_mean,
                "lon_mean": lon_mean,
                "lat_bin_2deg": lat_bin,
                "dist_from_event_deg_mean": _to_float(pd.to_numeric(grp.get("dist_from_event_deg"), errors="coerce").mean()),
                "nearest_storm_distance_km_mean": _to_float(pd.to_numeric(grp.get("nearest_storm_distance_km"), errors="coerce").mean()),
                "dist_from_event_bin_deg_mode": (
                    _to_float(pd.to_numeric(grp.get("dist_from_event_bin_deg"), errors="coerce").mode(dropna=True).iloc[0])
                    if "dist_from_event_bin_deg" in grp.columns and not pd.to_numeric(grp.get("dist_from_event_bin_deg"), errors="coerce").dropna().empty
                    else None
                ),
                "month": int(t_med.month) if pd.notna(t_med) else None,
                "hour": int(t_med.hour) if pd.notna(t_med) else None,
                "mean_abs_O_vector": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_vector"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_raw": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_raw"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_vorticity": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_vorticity"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_local_frame": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_local_frame"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_polar_spiral": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_polar_spiral"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_polar_chiral": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_polar_chiral"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_polar_odd_ratio": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_polar_odd_ratio"), errors="coerce").to_numpy(dtype=float)))),
                "mean_abs_O_polar_eta": _to_float(np.nanmean(np.abs(pd.to_numeric(grp.get("O_polar_eta"), errors="coerce").to_numpy(dtype=float)))),
                "mirror_ks_stat_O": ks_stat,
                "mirror_ks_pvalue_O": ks_p,
                "axis_robust": bool(axis_meta.get("axis_robust", False)),
                "axis_std": _to_float(axis_meta.get("contrast_std")),
                "axis_slope": _to_float(axis_meta.get("contrast_slope_per_lon")),
                "axis_best_lon0": _to_float(axis_meta.get("best_lon0")),
                "axis_worst_lon0": _to_float(axis_meta.get("worst_lon0")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"n_cases": 0}
    summary = {
        "n_cases": int(out["case_id"].nunique()),
        "by_case_type": {},
        "by_center_source": {},
        "by_distance_cohort": {},
    }
    for k, grp in out.groupby("case_type"):
        summary["by_case_type"][str(k)] = {
            "n_cases": int(grp.shape[0]),
            "parity_signal_rate_effective": float(pd.to_numeric(grp["parity_signal_pass_effective"], errors="coerce").fillna(0).mean()),
            "eta_mean_median": float(np.nanmedian(pd.to_numeric(grp["eta_mean"], errors="coerce"))),
            "eta_sign_consistency_median": float(np.nanmedian(pd.to_numeric(grp["eta_sign_consistency"], errors="coerce"))),
            "mirror_ks_stat_median": float(np.nanmedian(pd.to_numeric(grp["mirror_ks_stat_O"], errors="coerce"))),
        }
    for k, grp in out.groupby("center_source", dropna=False):
        summary["by_center_source"][str(k)] = {
            "n_cases": int(grp.shape[0]),
            "parity_signal_rate_effective": float(pd.to_numeric(grp["parity_signal_pass_effective"], errors="coerce").fillna(0).mean()),
            "eta_mean_median": float(np.nanmedian(pd.to_numeric(grp["eta_mean"], errors="coerce"))),
        }
    for k, grp in out.groupby("storm_distance_cohort", dropna=False):
        summary["by_distance_cohort"][str(k)] = {
            "n_cases": int(grp.shape[0]),
            "parity_signal_rate_effective": float(pd.to_numeric(grp["parity_signal_pass_effective"], errors="coerce").fillna(0).mean()),
            "eta_mean_median": float(np.nanmedian(pd.to_numeric(grp["eta_mean"], errors="coerce"))),
        }
    return out, summary


def _build_parity_breakdown(confound_df: pd.DataFrame, *, axis_meta: dict[str, Any]) -> dict[str, Any]:
    if confound_df.empty:
        return {"n_cases": 0}

    def _agg(g: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_cases": int(g.shape[0]),
            "parity_signal_rate_effective": float(pd.to_numeric(g["parity_signal_pass_effective"], errors="coerce").fillna(0).mean()),
            "eta_mean_median": float(np.nanmedian(pd.to_numeric(g["eta_mean"], errors="coerce"))),
        }

    by_cohort = {str(k): _agg(g) for k, g in confound_df.groupby("case_type", dropna=False)}
    by_lead = {
        str(k): _agg(g)
        for k, g in confound_df.groupby(["case_type", "lead_bucket"], dropna=False)
    }
    by_tier = {
        str(k): _agg(g)
        for k, g in confound_df.groupby("control_tier", dropna=False)
    }
    by_anomaly = {
        str(k): _agg(g)
        for k, g in confound_df.groupby("anomaly_mode_effective", dropna=False)
    }
    return {
        "n_cases": int(confound_df.shape[0]),
        "by_cohort": by_cohort,
        "by_cohort_lead": by_lead,
        "by_control_tier": by_tier,
        "by_anomaly_mode": by_anomaly,
        "axis_sensitivity": {
            "axis_robust": bool(axis_meta.get("axis_robust", False)),
            "contrast_std": _to_float(axis_meta.get("contrast_std")),
            "contrast_slope_per_lon": _to_float(axis_meta.get("contrast_slope_per_lon")),
            "contrast_by_lon0": axis_meta.get("contrast_by_lon0"),
        },
    }


def _build_far_cleanliness_dashboard(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if samples is None or samples.empty:
        return pd.DataFrame(), {"n_cases": 0}
    frame = samples.copy()
    if "case_id" not in frame.columns:
        return pd.DataFrame(), {"n_cases": 0, "reason": "missing_case_id"}
    frame["split_stratum"] = _derive_strata_labels(frame)
    frame = frame.loc[frame["split_stratum"] == "far_nonstorm"].copy()
    if frame.empty:
        return pd.DataFrame(), {"n_cases": 0}

    rows: list[dict[str, Any]] = []
    for case_id, grp in frame.groupby("case_id", dropna=False):
        rows.append(
            {
                "case_id": str(case_id),
                "lead_bucket": str(grp.get("lead_bucket", pd.Series(index=grp.index)).iloc[0]) if "lead_bucket" in grp.columns else "",
                "control_tier": str(grp.get("control_tier", pd.Series(index=grp.index)).iloc[0]) if "control_tier" in grp.columns else "",
                "ib_far_quality_tag": str(grp.get("ib_far_quality_tag", pd.Series(index=grp.index)).iloc[0]) if "ib_far_quality_tag" in grp.columns else "missing",
                "ib_far_strict": bool((pd.to_numeric(grp.get("ib_far_strict", grp.get("ib_far", 0)), errors="coerce").fillna(0) > 0).any()),
                "ib_dist_km_median": _to_float(pd.to_numeric(grp.get("ib_dist_km", grp.get("nearest_storm_distance_km")), errors="coerce").median()),
                "ib_min_dist_window_km_median": _to_float(
                    pd.to_numeric(
                        grp.get("nearest_storm_min_dist_window_km", grp.get("nearest_storm_min_dist_window_km_source")),
                        errors="coerce",
                    ).median()
                ),
                "case_speed_bin": _to_float(pd.to_numeric(grp.get("case_speed_bin"), errors="coerce").median()),
                "case_zeta_bin": _to_float(pd.to_numeric(grp.get("case_zeta_bin"), errors="coerce").median()),
                "storm_distance_cohort": str(grp.get("storm_distance_cohort", pd.Series(index=grp.index)).iloc[0]) if "storm_distance_cohort" in grp.columns else "",
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "n_cases": int(out.shape[0]),
        "by_quality_tag": out.get("ib_far_quality_tag", pd.Series(dtype=object)).fillna("missing").astype(str).value_counts(dropna=False).to_dict(),
        "strict_far_rate": float(pd.to_numeric(out.get("ib_far_strict"), errors="coerce").fillna(0).mean()) if "ib_far_strict" in out.columns else 0.0,
        "by_lead": {
            str(k): int(v)
            for k, v in out.get("lead_bucket", pd.Series(dtype=object)).fillna("missing").astype(str).value_counts(dropna=False).to_dict().items()
        },
    }
    return out, summary


def _detect_time_frequency_knee(
    case_df: pd.DataFrame,
    *,
    bic_delta_min: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"detected": False, "reason": "disabled"}
    eta_series = _case_eta_timeseries(case_df)
    if eta_series is None or eta_series.shape[0] < 24:
        return {"detected": False, "reason": "insufficient_time_points"}
    t_hours = eta_series["t_hours"].to_numpy(dtype=float)
    eta = eta_series["eta"].to_numpy(dtype=float)
    dt = np.diff(t_hours)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return {"detected": False, "reason": "invalid_time_axis"}
    dt_h = float(np.median(dt))
    fs = float(1.0 / dt_h)
    if fs <= 0:
        return {"detected": False, "reason": "invalid_sample_rate"}

    nperseg = int(min(128, max(16, eta.size // 2)))
    freqs, power = welch(eta - float(np.mean(eta)), fs=fs, nperseg=nperseg)
    mask = (freqs > 0) & np.isfinite(freqs) & np.isfinite(power) & (power > 0)
    f = freqs[mask]
    p = power[mask]
    if f.size < 10:
        return {"detected": False, "reason": "insufficient_spectrum_points"}

    x = np.log(f)
    y = np.log(p)
    best = _segmented_knee_from_loglog(
        x=x,
        y=y,
        min_points_each_side=3,
    )
    if best is None:
        return {"detected": False, "reason": "no_candidate"}
    per_scale_freqs: list[float] = []
    for _, sg in case_df.groupby("L", dropna=False):
        sg_ts = _case_eta_timeseries(sg)
        if sg_ts is None or sg_ts.shape[0] < 24:
            continue
        sg_t = sg_ts["t_hours"].to_numpy(dtype=float)
        sg_eta = sg_ts["eta"].to_numpy(dtype=float)
        sg_dt = np.diff(sg_t)
        sg_dt = sg_dt[np.isfinite(sg_dt) & (sg_dt > 0)]
        if sg_dt.size == 0:
            continue
        sg_fs = float(1.0 / np.median(sg_dt))
        if sg_fs <= 0:
            continue
        sg_nperseg = int(min(128, max(16, sg_eta.size // 2)))
        sg_freq, sg_pow = welch(sg_eta - float(np.mean(sg_eta)), fs=sg_fs, nperseg=sg_nperseg)
        sg_mask = (sg_freq > 0) & np.isfinite(sg_freq) & np.isfinite(sg_pow) & (sg_pow > 0)
        sg_f = sg_freq[sg_mask]
        sg_p = sg_pow[sg_mask]
        if sg_f.size < 10:
            continue
        sg_best = _segmented_knee_from_loglog(
            x=np.log(sg_f),
            y=np.log(sg_p),
            min_points_each_side=3,
        )
        if sg_best is None:
            continue
        per_scale_freqs.append(float(sg_best["f_knee"]))

    consistency_cv = _cv(per_scale_freqs)
    consistent = bool(consistency_cv is not None and consistency_cv <= 0.35)
    detected = bool(best["delta_bic"] >= float(bic_delta_min))
    return {
        "detected": detected,
        "reason": None if detected else "bic_weak",
        "knee_freq_per_hour": float(best["f_knee"]),
        "delta_bic": float(best["delta_bic"]),
        "slope_left": float(best["slope_left"]),
        "slope_right": float(best["slope_right"]),
        "consistent": consistent,
        "consistency_cv": consistency_cv,
        "n_scales_considered": int(len(per_scale_freqs)),
    }


def _tf_knee_null_gate(
    *,
    case_df: pd.DataFrame,
    bic_delta_min: float,
    seed: int,
) -> bool:
    null_a = _fake_mirror_pairing(case_df, rng=np.random.default_rng(int(seed) + 1))
    null_b = _circular_lon_shift_pairing(case_df, rng=np.random.default_rng(int(seed) + 2))
    tf_a = _detect_time_frequency_knee(
        null_a,
        bic_delta_min=float(bic_delta_min),
        enabled=True,
    )
    tf_b = _detect_time_frequency_knee(
        null_b,
        bic_delta_min=float(bic_delta_min),
        enabled=True,
    )
    return bool((not bool(tf_a.get("detected", False))) and (not bool(tf_b.get("detected", False))))


def _case_eta_timeseries(case_df: pd.DataFrame) -> pd.DataFrame | None:
    if case_df.empty:
        return None
    pivot = case_df.pivot_table(
        index=["t", "L"],
        columns="hand",
        values="O",
        aggfunc="first",
    )
    if "L" not in pivot.columns or "R" not in pivot.columns:
        return None
    pivot = pivot.rename(columns={"L": "O_L", "R": "O_R"}).reset_index()
    o_l = pd.to_numeric(pivot["O_L"], errors="coerce").to_numpy(dtype=float)
    o_r = pd.to_numeric(pivot["O_R"], errors="coerce").to_numpy(dtype=float)
    denom = 0.5 * (o_l + o_r)
    valid = np.isfinite(o_l) & np.isfinite(o_r) & np.isfinite(denom) & (denom > 1e-12)
    if not np.any(valid):
        return None
    eta = np.abs(o_l[valid] - o_r[valid]) / denom[valid]
    t_vals = pd.to_datetime(pivot.loc[valid, "t"], errors="coerce")
    ts = pd.DataFrame({"t": t_vals, "eta": eta}).dropna(subset=["t", "eta"])
    if ts.empty:
        return None
    ts = ts.groupby("t", as_index=False)["eta"].median().sort_values("t")
    t0 = pd.Timestamp(ts["t"].iloc[0])
    ts["t_hours"] = (pd.to_datetime(ts["t"], errors="coerce") - t0).dt.total_seconds() / 3600.0
    ts = ts.dropna(subset=["t_hours", "eta"])
    return ts if not ts.empty else None


def _segmented_knee_from_loglog(
    *,
    x: np.ndarray,
    y: np.ndarray,
    min_points_each_side: int,
) -> dict[str, float] | None:
    n = int(x.size)
    if n < (2 * int(min_points_each_side) + 2):
        return None
    rss0 = _linear_rss(x, y)
    bic0 = _bic(rss0, n=n, k=2)
    best: dict[str, float] | None = None
    for idx in range(int(min_points_each_side), n - int(min_points_each_side)):
        rss1, slope_left, slope_right = _piecewise_rss_and_slopes(x, y, idx)
        bic1 = _bic(rss1, n=n, k=3)
        delta_bic = float(bic0 - bic1)
        if best is None or delta_bic > best["delta_bic"]:
            best = {
                "idx": float(idx),
                "delta_bic": delta_bic,
                "f_knee": float(np.exp(x[idx])),
                "slope_left": float(slope_left),
                "slope_right": float(slope_right),
            }
    return best


def _linear_rss(x: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return float(np.sum((y - pred) ** 2))


def _piecewise_rss_and_slopes(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float, float]:
    xk = x[idx]
    d = x - xk
    X = np.column_stack([np.ones_like(x), np.minimum(d, 0.0), np.maximum(d, 0.0)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    return rss, float(beta[1]), float(beta[2])


def _bic(rss: float, n: int, k: int) -> float:
    rss_clamped = max(float(rss), 1e-12)
    return float(n * np.log(rss_clamped / max(1, n)) + k * np.log(max(2, n)))


def _safe_median(series: pd.Series | Any) -> float | None:
    if series is None:
        return None
    try:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def _summary_numeric(values: Any) -> dict[str, float] | None:
    try:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _summary_hist(values: Any, *, bins: list[float]) -> dict[str, Any] | None:
    try:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    edges = np.asarray(bins, dtype=float)
    if edges.size < 2:
        return None
    hist, bin_edges = np.histogram(arr, bins=edges)
    return {
        "bins": [float(v) for v in bin_edges.tolist()],
        "counts": [int(v) for v in hist.tolist()],
        "n": int(arr.size),
    }


def _cv(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    med = float(np.median(arr))
    if abs(med) <= 1e-12:
        return None
    return float(np.std(arr, ddof=0) / abs(med))


def _build_ibtracs_strict_eval(
    *,
    case_metrics: pd.DataFrame,
    samples: pd.DataFrame,
    radius_km: float,
    far_min_km: float,
    time_hours: float,
    p_lock_threshold: float,
    use_flags: bool = False,
    time_basis: str = "selected",
) -> dict[str, Any]:
    if case_metrics is None or case_metrics.empty:
        return {"available": False, "reason": "no_case_metrics", "time_basis": str(time_basis)}

    case_df = case_metrics.copy()
    case_df["case_id"] = case_df["case_id"].astype(str)

    # Fill distance/time metadata from samples if missing in case_metrics.
    basis = str(time_basis or "selected").strip().lower()
    if basis not in {"source", "valid", "selected"}:
        basis = "selected"
    if basis == "source":
        dist_pref = ("ib_dist_km_source", "nearest_storm_distance_km_source", "storm_dist_km")
        time_pref = (
            "ib_dt_hours_source",
            "nearest_storm_time_delta_h_source",
            "nearest_storm_time_delta_hours_source",
            "storm_time_delta_h_source",
            "storm_time_delta_hours_source",
            "ibtracs_time_delta_h_source",
            "ibtracs_time_delta_hours_source",
        )
        event_flag_pref = ("ib_event_strict_source", "ib_event_source", "ib_event_strict", "ib_event")
        far_flag_pref = ("ib_far_strict_source", "ib_far_source", "ib_far_strict", "ib_far")
    elif basis == "valid":
        dist_pref = ("ib_dist_km_valid", "nearest_storm_distance_km_valid", "storm_dist_km")
        time_pref = (
            "ib_dt_hours_valid",
            "nearest_storm_time_delta_h_valid",
            "nearest_storm_time_delta_hours_valid",
            "storm_time_delta_h_valid",
            "storm_time_delta_hours_valid",
            "ibtracs_time_delta_h_valid",
            "ibtracs_time_delta_hours_valid",
        )
        event_flag_pref = ("ib_event_strict_valid", "ib_event_valid", "ib_event_strict", "ib_event")
        far_flag_pref = ("ib_far_strict_valid", "ib_far_valid", "ib_far_strict", "ib_far")
    else:
        dist_pref = ("ib_dist_km", "nearest_storm_distance_km", "storm_dist_km")
        time_pref = (
            "ib_dt_hours",
            "nearest_storm_time_delta_h",
            "nearest_storm_time_delta_hours",
            "storm_time_delta_h",
            "storm_time_delta_hours",
            "ibtracs_time_delta_h",
            "ibtracs_time_delta_hours",
        )
        event_flag_pref = ("ib_event_strict", "ib_event", "ib_event_strict_max", "ib_event_max")
        far_flag_pref = ("ib_far_strict", "ib_far", "ib_far_strict_max", "ib_far_max")

    dist_col_candidates = [c for c in dist_pref if c in samples.columns]
    meta_cols = [c for c in ("case_id",) if c in samples.columns]
    if dist_col_candidates:
        meta_cols.append(dist_col_candidates[0])
    time_delta_candidates = [c for c in time_pref if c in samples.columns]
    if time_delta_candidates:
        meta_cols.append(time_delta_candidates[0])
    for col in ("ib_event_strict", "ib_far_strict", "ib_event", "ib_far", "ib_far_quality_tag"):
        if col in samples.columns:
            meta_cols.append(col)
    if "storm_id_nearest" in samples.columns:
        meta_cols.append("storm_id_nearest")
    elif "storm_id" in samples.columns:
        meta_cols.append("storm_id")
    if len(meta_cols) > 1:
        meta = (
            samples[meta_cols]
            .copy()
            .assign(case_id=lambda d: d["case_id"].astype(str))
            .groupby("case_id", as_index=False)
            .agg(
                {
                    **({dist_col_candidates[0]: "median"} if dist_col_candidates else {}),
                    **({time_delta_candidates[0]: "median"} if time_delta_candidates else {}),
                    **({"ib_event_strict": "max"} if "ib_event_strict" in meta_cols else {}),
                    **({"ib_far_strict": "max"} if "ib_far_strict" in meta_cols else {}),
                    **({"ib_event": "max"} if "ib_event" in meta_cols else {}),
                    **({"ib_far": "max"} if "ib_far" in meta_cols else {}),
                    **({"ib_far_quality_tag": "first"} if "ib_far_quality_tag" in meta_cols else {}),
                    **({"storm_id_nearest": "first"} if "storm_id_nearest" in meta_cols else {}),
                    **({"storm_id": "first"} if "storm_id" in meta_cols else {}),
                }
            )
        )
        case_df = case_df.merge(meta, on="case_id", how="left", suffixes=("", "_sample"))

    dist_case_col = dist_col_candidates[0] if dist_col_candidates else "nearest_storm_distance_km"
    dist = pd.to_numeric(case_df.get(dist_case_col), errors="coerce")
    if dist.isna().all():
        return {"available": False, "reason": "missing_nearest_storm_distance_km", "time_basis": basis}

    event_mask = dist <= float(radius_km)
    if time_delta_candidates:
        td = pd.to_numeric(case_df.get(time_delta_candidates[0]), errors="coerce").abs()
        event_mask = event_mask & (td <= float(time_hours))
    far_mask = dist >= float(far_min_km)

    event_flag_cols = [c for c in event_flag_pref if c in case_df.columns]
    if not event_flag_cols:
        event_flag_cols = [c for c in ("ib_event_strict_max", "ib_event_strict", "ib_event_max", "ib_event") if c in case_df.columns]
    far_flag_cols = [c for c in far_flag_pref if c in case_df.columns]
    if not far_flag_cols:
        far_flag_cols = [c for c in ("ib_far_strict_max", "ib_far_strict", "ib_far_max", "ib_far") if c in case_df.columns]
    if bool(use_flags) and event_flag_cols and far_flag_cols:
        event_mask = pd.to_numeric(case_df.get(event_flag_cols[0]), errors="coerce").fillna(0) > 0
        far_mask = pd.to_numeric(case_df.get(far_flag_cols[0]), errors="coerce").fillna(0) > 0
    event_df = case_df.loc[event_mask].copy()
    far_df = case_df.loc[far_mask].copy()
    if event_df.empty or far_df.empty:
        return {
            "available": False,
            "reason": "insufficient_strict_or_far_cases",
            "time_basis": basis,
            "n_event_cases": int(event_df.shape[0]),
            "n_far_cases": int(far_df.shape[0]),
            "dist_km_summary": _summary_numeric(dist),
        }

    event_summary = _aggregate_case_metrics(event_df, p_lock_threshold=float(p_lock_threshold))
    far_summary = _aggregate_case_metrics(far_df, p_lock_threshold=float(p_lock_threshold))
    event_rate = float(event_summary.get("parity_signal_rate", 0.0))
    far_rate = float(far_summary.get("parity_signal_rate", 0.0))

    by_lead: dict[str, Any] = {}
    ev_lead = event_df["lead_bucket"].astype(str) if "lead_bucket" in event_df.columns else pd.Series("", index=event_df.index, dtype=str)
    far_lead = far_df["lead_bucket"].astype(str) if "lead_bucket" in far_df.columns else pd.Series("", index=far_df.index, dtype=str)
    lead_values = sorted(set(ev_lead.dropna().tolist()) | set(far_lead.dropna().tolist()))
    for lead in lead_values:
        ev_l = event_df.loc[ev_lead == str(lead)].copy()
        far_l = far_df.loc[far_lead == str(lead)].copy()
        by_lead[str(lead)] = {
            "events": _aggregate_case_metrics(ev_l, p_lock_threshold=float(p_lock_threshold)),
            "far_controls": _aggregate_case_metrics(far_l, p_lock_threshold=float(p_lock_threshold)),
        }

    return {
        "available": True,
        "time_basis": basis,
        "radius_km": float(radius_km),
        "far_min_km": float(far_min_km),
        "time_hours": float(time_hours),
        "n_event_cases": int(event_df.shape[0]),
        "n_far_cases": int(far_df.shape[0]),
        "events": event_summary,
        "far_controls": far_summary,
        "event_minus_far_parity_rate": float(event_rate - far_rate),
        "confound_rate": float(far_rate / max(event_rate, 1e-9)),
        "distance_col_used": str(dist_case_col),
        "time_col_used": str(time_delta_candidates[0]) if time_delta_candidates else None,
        "event_flag_col_used": str(event_flag_cols[0]) if (bool(use_flags) and event_flag_cols) else None,
        "far_flag_col_used": str(far_flag_cols[0]) if (bool(use_flags) and far_flag_cols) else None,
        "strict_mask_source": "flags" if (bool(use_flags) and event_flag_cols and far_flag_cols) else "distance_time",
        "dist_km_summary": _summary_numeric(dist),
        "dt_hours_summary": (
            _summary_numeric(pd.to_numeric(case_df.get(time_delta_candidates[0]), errors="coerce").abs())
            if time_delta_candidates
            else None
        ),
        "dist_km_hist": _summary_hist(dist, bins=[0, 100, 300, 500, 800, 1200, 2000, 4000, 8000, 16000]),
        "dt_hours_hist": (
            _summary_hist(
                pd.to_numeric(case_df.get(time_delta_candidates[0]), errors="coerce").abs(),
                bins=[0, 1, 3, 6, 12, 24, 48, 96],
            )
            if time_delta_candidates
            else None
        ),
        "far_quality_tag_counts": (
            case_df.get("ib_far_quality_tag", pd.Series(dtype=object)).fillna("missing").astype(str).value_counts(dropna=False).to_dict()
            if "ib_far_quality_tag" in case_df.columns
            else {}
        ),
        "center_source_counts": (
            case_df.get("center_source", pd.Series(dtype=object)).fillna("unknown").astype(str).value_counts(dropna=False).to_dict()
            if "center_source" in case_df.columns
            else {}
        ),
        "by_lead": by_lead,
    }


def _build_ibtracs_alignment_gate(
    *,
    strict_eval: dict[str, Any],
    min_event_cases: int,
    min_far_cases: int,
    margin_min: float,
    confound_max: float,
    extra_reasons: list[str] | None = None,
) -> dict[str, Any]:
    info = strict_eval or {}
    reasons: list[str] = []
    available = bool(info.get("available", False))
    n_event = int(info.get("n_event_cases", 0) or 0)
    n_far = int(info.get("n_far_cases", 0) or 0)
    margin = _to_float(info.get("event_minus_far_parity_rate"))
    confound = _to_float(info.get("confound_rate"))
    time_hours = _to_float(info.get("time_hours")) or 0.0
    radius_km = _to_float(info.get("radius_km")) or 0.0
    dist_summary = info.get("dist_km_summary") if isinstance(info.get("dist_km_summary"), dict) else None
    dt_summary = info.get("dt_hours_summary") if isinstance(info.get("dt_hours_summary"), dict) else None
    dist_p50 = _to_float((dist_summary or {}).get("p50")) if dist_summary else None
    dist_p95 = _to_float((dist_summary or {}).get("p95")) if dist_summary else None
    dist_p05 = _to_float((dist_summary or {}).get("p05")) if dist_summary else None
    dist_max = _to_float((dist_summary or {}).get("max")) if dist_summary else None
    dt_p50 = _to_float((dt_summary or {}).get("p50")) if dt_summary else None
    dt_p95 = _to_float((dt_summary or {}).get("p95")) if dt_summary else None

    suspect_time_basis = bool(
        (dt_p50 is not None)
        and (time_hours > 0.0)
        and (dt_p50 >= 0.75 * time_hours)
        and (margin is None or margin < float(margin_min))
    )
    suspect_distance_units = bool(
        ((dist_p50 is not None) and (dist_p50 > 8_000.0))
        or ((dist_max is not None) and (dist_max < 5.0) and n_event > 0)
    )
    suspect_lon_wrap = bool(
        (dist_p95 is not None)
        and (dist_p05 is not None)
        and (dist_p95 > 6_000.0)
        and (dist_p05 < 500.0)
    )
    suspect_center_mismatch = bool(
        (dist_p50 is not None)
        and (radius_km > 0.0)
        and (dist_p50 > 1.5 * radius_km)
        and (margin is None or margin < float(margin_min))
    )

    if not available:
        reasons.append(str(info.get("reason") or "strict_overlay_unavailable"))
    if n_event < int(min_event_cases):
        reasons.append(f"too_few_ibtracs_event_cases:{n_event}<{int(min_event_cases)}")
    if n_far < int(min_far_cases):
        reasons.append(f"too_few_ibtracs_far_cases:{n_far}<{int(min_far_cases)}")
    if margin is None:
        reasons.append("missing_event_minus_far_margin")
    elif margin < float(margin_min):
        reasons.append(f"negative_or_low_margin:{margin:.6f}<{float(margin_min):.6f}")
    if confound is None:
        reasons.append("missing_confound_rate")
    elif confound > float(confound_max):
        reasons.append(f"confound_exceeds_max:{confound:.6f}>{float(confound_max):.6f}")
    if suspect_time_basis:
        reasons.append("time_basis_mismatch_suspected")
    if suspect_distance_units:
        reasons.append("distance_units_suspected")
    if suspect_lon_wrap:
        reasons.append("lon_wrap_suspected")
    if suspect_center_mismatch:
        reasons.append("center_definition_mismatch")
    for r in (extra_reasons or []):
        if r:
            reasons.append(str(r))
    passed = bool(len(reasons) == 0)
    actions: list[str] = []
    if not passed:
        if any("too_few_ibtracs_event_cases" in r or "too_few_ibtracs_far_cases" in r for r in reasons):
            actions.append("increase_strict_overlay_coverage_or_adjust_split")
        if any("margin" in r or "confound" in r for r in reasons):
            actions.append("audit_ibtracs_join_time_tolerance_lon_wrap_and_event_radius")
            actions.append("verify_event_vs_far_cohort_definition_and_distance_thresholds")
        if any("time_basis_mismatch_suspected" in r for r in reasons):
            actions.append("use_valid_time_for_ibtracs_matching")
        if any("distance_units_suspected" in r for r in reasons):
            actions.append("audit_distance_units_and_haversine_inputs")
        if any("lon_wrap_suspected" in r for r in reasons):
            actions.append("normalize_longitudes_to_minus180_180_before_matching")
        if any("center_definition_mismatch" in r for r in reasons):
            actions.append("calibrate_vortex_center_definition_against_ibtracs")
        if any("basis_flip_changes_label" in r for r in reasons):
            actions.append("compare_source_vs_valid_time_basis_and_lock_one_contract")
    reason_code_counts = pd.Series(reasons, dtype=object).value_counts(dropna=False).to_dict() if reasons else {}
    remediation_hints = {
        str(code): [
            "increase_strict_overlay_coverage_or_adjust_split"
            if "too_few_ibtracs_" in str(code)
            else "use_valid_time_for_ibtracs_matching"
            if "time_basis_mismatch" in str(code)
            else "normalize_longitudes_to_minus180_180_before_matching"
            if "lon_wrap" in str(code)
            else "audit_distance_units_and_haversine_inputs"
            if "distance_units" in str(code)
            else "calibrate_vortex_center_definition_against_ibtracs"
            if "center_definition" in str(code)
            else "verify_event_vs_far_cohort_definition_and_distance_thresholds"
        ]
        for code in reason_code_counts.keys()
    }
    return {
        "passed": bool(passed),
        "available": bool(available),
        "n_event_cases": int(n_event),
        "n_far_cases": int(n_far),
        "event_minus_far_parity_rate": margin,
        "confound_rate": confound,
        "thresholds": {
            "min_event_cases": int(min_event_cases),
            "min_far_cases": int(min_far_cases),
            "margin_min": float(margin_min),
            "confound_max": float(confound_max),
        },
        "diagnostics": {
            "dist_km_summary": dist_summary,
            "dt_hours_summary": dt_summary,
            "dist_km_hist": info.get("dist_km_hist"),
            "dt_hours_hist": info.get("dt_hours_hist"),
            "distance_col_used": info.get("distance_col_used"),
            "time_col_used": info.get("time_col_used"),
            "time_hours": float(time_hours),
            "radius_km": float(radius_km),
            "suspect_flags": {
                "time_basis_mismatch_suspected": bool(suspect_time_basis),
                "distance_units_suspected": bool(suspect_distance_units),
                "lon_wrap_suspected": bool(suspect_lon_wrap),
                "center_definition_mismatch": bool(suspect_center_mismatch),
            },
        },
        "reason_codes": reasons,
        "reason_code_counts": reason_code_counts,
        "actions": actions,
        "remediation_hints": remediation_hints,
    }


def _build_time_basis_audit_from_samples(samples: pd.DataFrame) -> dict[str, Any]:
    if samples is None or samples.empty:
        return {"available": False, "n_rows": 0}
    s = samples.copy()
    out = {
        "available": True,
        "n_rows": int(s.shape[0]),
        "time_basis_used_counts": s.get("ib_time_basis_used", s.get("time_basis", pd.Series(index=s.index))).fillna("unknown").astype(str).value_counts(dropna=False).to_dict(),
    }
    if {"ib_event_source", "ib_event_valid"}.issubset(set(s.columns)):
        ev_s = (pd.to_numeric(s["ib_event_source"], errors="coerce").fillna(0) > 0).astype(int)
        ev_v = (pd.to_numeric(s["ib_event_valid"], errors="coerce").fillna(0) > 0).astype(int)
        out["event_label_flip_rate"] = float((ev_s != ev_v).mean())
    if {"ib_far_source", "ib_far_valid"}.issubset(set(s.columns)):
        far_s = (pd.to_numeric(s["ib_far_source"], errors="coerce").fillna(0) > 0).astype(int)
        far_v = (pd.to_numeric(s["ib_far_valid"], errors="coerce").fillna(0) > 0).astype(int)
        out["far_label_flip_rate"] = float((far_s != far_v).mean())
    if {"nearest_storm_id_source", "nearest_storm_id_valid"}.issubset(set(s.columns)):
        sid_s = s["nearest_storm_id_source"].fillna("").astype(str)
        sid_v = s["nearest_storm_id_valid"].fillna("").astype(str)
        out["nearest_storm_id_flip_rate"] = float((sid_s != sid_v).mean())
    return out


def _build_center_definition_audit_from_samples(samples: pd.DataFrame) -> dict[str, Any]:
    if samples is None or samples.empty:
        return {"available": False, "n_rows": 0}
    s = samples.copy()
    dist = pd.to_numeric(s.get("nearest_storm_distance_km"), errors="coerce")
    out: dict[str, Any] = {
        "available": True,
        "n_rows": int(s.shape[0]),
        "center_to_storm_dist_km_summary": _summary_numeric(dist),
        "center_mismatch_gt200km_rate": float((dist > 200.0).mean()) if dist.notna().any() else 0.0,
    }
    if "center_source" in s.columns:
        by_src: dict[str, Any] = {}
        for src, grp in s.groupby(s["center_source"].fillna("unknown").astype(str), dropna=False):
            d = pd.to_numeric(grp.get("nearest_storm_distance_km"), errors="coerce")
            by_src[str(src)] = {
                "n_rows": int(grp.shape[0]),
                "dist_km_summary": _summary_numeric(d),
                "mismatch_gt200km_rate": float((d > 200.0).mean()) if d.notna().any() else 0.0,
            }
        out["by_center_source"] = by_src
    return out


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(int(chunk_size))
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _dataset_hash(dataset_dir: Path) -> str:
    files = [dataset_dir / "dataset.yaml", dataset_dir / "samples.parquet"]
    hasher = hashlib.sha256()
    for path in files:
        if not path.exists():
            continue
        rel = path.name.encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(_file_sha256(path).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _config_hash(config_path: Path) -> str:
    if not config_path.exists():
        return ""
    return _file_sha256(config_path)


def _git_commit_hash(repo_dir: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        out = str(proc.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _build_eval_command_tokens(args: argparse.Namespace, *, out_path: Path) -> list[str]:
    tokens = [
        "python",
        "examples/weather_real_minipilot/evaluate_minipilot.py",
    ]
    for key, value in vars(args).items():
        opt = f"--{str(key).replace('_', '-')}"
        if key == "out":
            value = str(out_path)
        if isinstance(value, bool):
            if value:
                tokens.append(opt)
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            tokens.append(opt)
            tokens.extend([str(v) for v in value])
            continue
        if value is None:
            continue
        tokens.extend([opt, str(value)])
    return tokens


def _build_claim_contract(
    *,
    summary: dict[str, Any],
    args: argparse.Namespace,
    dataset_dir: Path,
    config_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parents[2]
    applied_parity = (summary.get("parity_thresholds_applied") or {}) if isinstance(summary, dict) else {}
    contract = {
        "schema_version": int(getattr(args, "claim_contract_schema_version", 1)),
        "generated_at_utc": utc_now_iso(),
        "dataset_path": str(dataset_dir.resolve()),
        "dataset_hash": _dataset_hash(dataset_dir),
        "config_path": str(config_path.resolve()),
        "config_hash": _config_hash(config_path),
        "git_commit": _git_commit_hash(repo_dir),
        "seed": int(getattr(args, "seed", 0)),
        "split": {
            "split_date": str(getattr(args, "split_date", "")),
            "time_buffer_hours": float(getattr(args, "time_buffer_hours", 0.0)),
            "policy": "blocked_time_with_storm_id_containment",
            "storm_id_col": summary.get("storm_id_col"),
        },
        "claim_mode": str(summary.get("claim_mode", "")),
        "claim_mode_applied": str(summary.get("claim_mode_applied", "")),
        "canonical_anomaly_mode_by_lead": (
            (summary.get("anomaly_mode_selection") or {}).get("canonical_mode_by_lead", {}) or {}
        ),
        "anomaly_mode_default": (summary.get("anomaly_mode_selection") or {}).get("default_mode"),
        "thresholds": {
            "parity_confound_gate": {
                "margin_min": float(getattr(args, "parity_event_minus_far_min", 0.0)),
                "far_max": float(getattr(args, "parity_far_max", 0.0)),
                "ratio_max": float(getattr(args, "parity_confound_max_ratio", 0.0)),
                "use_alignment_fraction": bool(getattr(args, "parity_use_alignment_fraction", True)),
                "alignment_eta_threshold": float(getattr(args, "alignment_eta_threshold", 0.10)),
                "alignment_block_hours": float(getattr(args, "alignment_block_hours", 3.0)),
                "odd_inner_outer_min": float(
                    _to_float(applied_parity.get("odd_inner_outer_min"))
                    if _to_float(applied_parity.get("odd_inner_outer_min")) is not None
                    else float(getattr(args, "parity_odd_inner_outer_min", 0.0))
                ),
                "tangential_radial_min": float(
                    _to_float(applied_parity.get("tangential_radial_min"))
                    if _to_float(applied_parity.get("tangential_radial_min")) is not None
                    else float(getattr(args, "parity_tangential_radial_min", 0.0))
                ),
                "inner_ring_quantile": float(
                    _to_float(applied_parity.get("inner_ring_quantile"))
                    if _to_float(applied_parity.get("inner_ring_quantile")) is not None
                    else float(getattr(args, "parity_inner_ring_quantile", 0.5))
                ),
                "calibration_event_min_floor": float(getattr(args, "parity_calibration_event_min_floor", 0.05)),
                "calibration_constrain_min_thresholds": bool(getattr(args, "parity_calibration_constrain_min_thresholds", False)),
                "far_quality_tags": [str(v) for v in (summary.get("claim_far_quality_tags", []) or [])],
                "strict_far_min_row_quality_frac": float(getattr(args, "strict_far_min_row_quality_frac", 0.0)),
                "strict_far_min_kinematic_clean_frac": float(getattr(args, "strict_far_min_kinematic_clean_frac", 0.0)),
                "strict_far_min_any_storm_km": float(getattr(args, "strict_far_min_any_storm_km", 0.0)),
                "strict_far_min_nearest_storm_km": float(getattr(args, "strict_far_min_nearest_storm_km", 0.0)),
            },
            "anomaly_gate": {
                "selected_all_leads_required": True,
                "robust_far_all_leads_required": True,
                "agreement_mean_min": float(getattr(args, "anomaly_agreement_mean_min", 0.0)),
                "agreement_min_min": float(getattr(args, "anomaly_agreement_min_min", 0.0)),
            },
            "geometry_null_collapse": {
                "required_checks": [str(v) for v in (summary.get("geometry_required_checks", []) or [])],
                "relative_drop_min": float(getattr(args, "null_collapse_min_drop", 0.0)),
                "absolute_drop_min": float(getattr(args, "null_collapse_min_abs_drop", 0.0)),
                "null_no_signal_max_rate": float(getattr(args, "null_no_signal_max_rate", 0.0)),
            },
            "angular_witness": {
                "margin_min": float(getattr(args, "angular_witness_margin_min", 0.0)),
                "effect_size_d_min": float(getattr(args, "angular_witness_d_min", 0.0)),
                "p_value_max": float(getattr(args, "angular_witness_p_max", 1.0)),
                "permutation_n": int(getattr(args, "angular_witness_permutation_n", 0)),
                "null_drop_min": float(getattr(args, "angular_witness_null_drop_min", 0.0)),
                "null_abs_drop_min": float(getattr(args, "angular_witness_null_abs_drop_min", 0.0)),
                "null_margin_max": float(getattr(args, "angular_witness_null_margin_max", 0.0)),
            },
            "ibtracs_alignment": {
                "min_event_cases": int(getattr(args, "ibtracs_alignment_min_event_cases", 0)),
                "min_far_cases": int(getattr(args, "ibtracs_alignment_min_far_cases", 0)),
                "margin_min": float(getattr(args, "ibtracs_alignment_margin_min", 0.0)),
                "confound_max": float(getattr(args, "ibtracs_alignment_confound_max", 1.0)),
                "strict_use_flags": bool(getattr(args, "ibtracs_strict_use_flags", False)),
            },
        },
        "knee_policy": {
            "primary": "tf",
            "size_role": "secondary_conditional",
        },
        "evaluation_output": str(out_path.resolve()),
        "command": _build_eval_command_tokens(args, out_path=out_path),
    }
    return contract


def _validate_claim_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["contract_not_dict"]
    required_top = (
        "schema_version",
        "dataset_hash",
        "git_commit",
        "seed",
        "split",
        "canonical_anomaly_mode_by_lead",
        "claim_mode",
        "thresholds",
        "command",
    )
    for key in required_top:
        if key not in contract:
            errors.append(f"missing:{key}")
    if not isinstance(contract.get("schema_version"), int):
        errors.append("schema_version_not_int")
    if not str(contract.get("dataset_hash", "")).strip():
        errors.append("dataset_hash_empty")
    if not str(contract.get("claim_mode", "")).strip():
        errors.append("claim_mode_empty")
    split = contract.get("split")
    if not isinstance(split, dict):
        errors.append("split_not_dict")
    else:
        for key in ("split_date", "time_buffer_hours", "policy"):
            if key not in split:
                errors.append(f"missing:split.{key}")
    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds_not_dict")
    else:
        for gate in ("parity_confound_gate", "anomaly_gate", "geometry_null_collapse", "angular_witness", "ibtracs_alignment"):
            if gate not in thresholds:
                errors.append(f"missing:thresholds.{gate}")
    command = contract.get("command")
    if not isinstance(command, list) or len(command) < 3:
        errors.append("command_invalid")
    return errors


def _write_reproduce_script(
    *,
    path: Path,
    args: argparse.Namespace,
    dataset_dir: Path,
    config_path: Path,
    out_path: Path,
    claim_contract: dict[str, Any],
) -> None:
    dataset_hash = str(claim_contract.get("dataset_hash", ""))
    git_commit = str(claim_contract.get("git_commit") or "")
    out_eval = out_path.resolve()
    cmd_tokens = _build_eval_command_tokens(args, out_path=out_eval)
    cmd = " ".join(f"\"{t}\"" if (" " in t or ":" in t or "\\" in t) else t for t in cmd_tokens)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "REPO_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\"",
        "cd \"$REPO_ROOT\"",
        "",
    ]
    if git_commit:
        lines.extend(
            [
                f"EXPECTED_COMMIT=\"{git_commit}\"",
                "CURRENT_COMMIT=\"$(git rev-parse HEAD 2>/dev/null || true)\"",
                "if [[ \"$CURRENT_COMMIT\" != \"$EXPECTED_COMMIT\" ]]; then",
                "  echo \"Commit mismatch: expected $EXPECTED_COMMIT got $CURRENT_COMMIT\"",
                "  echo \"Run: git checkout $EXPECTED_COMMIT\"",
                "  exit 1",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            f"EXPECTED_DATASET_HASH=\"{dataset_hash}\"",
            "ACTUAL_DATASET_HASH=\"$(python - <<'PY'",
            "import hashlib",
            "from pathlib import Path",
            f"root = Path(r'''{str(dataset_dir.resolve())}''')",
            "files = [root / 'dataset.yaml', root / 'samples.parquet']",
            "h = hashlib.sha256()",
            "for p in files:",
            "    if not p.exists():",
            "        continue",
            "    h.update(p.name.encode('utf-8')); h.update(b'\\0')",
            "    hp = hashlib.sha256()",
            "    with p.open('rb') as fh:",
            "        while True:",
            "            c = fh.read(1024 * 1024)",
            "            if not c:",
            "                break",
            "            hp.update(c)",
            "    h.update(hp.hexdigest().encode('utf-8')); h.update(b'\\0')",
            "print(h.hexdigest())",
            "PY",
            ")\"",
            "if [[ \"$ACTUAL_DATASET_HASH\" != \"$EXPECTED_DATASET_HASH\" ]]; then",
            "  echo \"Dataset hash mismatch: expected $EXPECTED_DATASET_HASH got $ACTUAL_DATASET_HASH\"",
            "  exit 1",
            "fi",
            "",
            f"mkdir -p \"{str(out_eval.parent)}\"",
            cmd,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        fv = float(value)
        if not np.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
