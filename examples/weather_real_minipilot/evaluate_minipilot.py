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
        "--claim-contract-schema-version",
        type=int,
        default=1,
        help="Schema version written to claim_contract.json",
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

    samples = pd.read_parquet(dataset_dir / "samples.parquet")
    samples["t"] = pd.to_datetime(samples["t"], errors="coerce")
    if "onset_time" in samples.columns:
        samples["onset_time"] = pd.to_datetime(samples["onset_time"], errors="coerce")
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
    )
    test_case_metrics = _evaluate_cases(
        test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 1,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
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
        "n_cases_train": int(len(train_cases)),
        "n_cases_test": int(len(test_cases)),
        "n_cases_buffer_excluded": int(len(buffer_cases)),
        "test_coverage": test_cov,
        "far_nonstorm_coverage_test": test_cov.get("far_nonstorm", {}),
        "event_coverage_test": test_cov.get("events", {}),
        "min_train_cases": int(args.min_train_cases),
        "min_test_cases": int(args.min_test_cases),
        "min_event_test_cases": int(args.min_event_test_cases),
        "min_event_test_per_lead": int(args.min_event_test_per_lead),
        "min_far_nonstorm_test_cases": int(args.min_far_nonstorm_test_cases),
        "min_far_nonstorm_test_per_lead": int(args.min_far_nonstorm_test_per_lead),
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
        "fake_mirror_pairing": _fake_mirror_pairing,
        "latitude_mirror_pairing": _latitude_mirror_pairing,
        "circular_lon_shift_pairing": _circular_lon_shift_pairing,
        "theta_roll": _theta_roll_polar,
        "radial_shuffle": _radial_shuffle_polar,
        "radial_scramble": _radial_scramble_polar,
        "mirror_axis_jitter": _mirror_axis_jitter,
        "center_jitter": _center_jitter_polar,
    }
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
        )
        metrics = _apply_axis_robust_gating(metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
        null_case_metrics[name] = metrics
        null_strata[name] = _summarize_case_metrics_by_strata(metrics, p_lock_threshold=float(args.p_lock_threshold))
    summary["null_controls"] = null_controls
    summary["null_strata"] = null_strata
    geometry_null_audit = _build_geometry_null_audit(
        samples=test_df,
        transforms={
            "theta_roll": null_transforms["theta_roll"],
            "radial_shuffle": null_transforms["radial_shuffle"],
            "radial_scramble": null_transforms["radial_scramble"],
            "mirror_axis_jitter": null_transforms["mirror_axis_jitter"],
            "center_jitter": null_transforms["center_jitter"],
        },
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

    summary["claim_mode_applied"] = claim_mode
    claim_train_metrics = _evaluate_cases(
        claim_train_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 9000,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
    )
    claim_test_metrics = _evaluate_cases(
        claim_test_df,
        dataset_dir=dataset_dir,
        config_path=config_path,
        seed=int(args.seed) + 9001,
        enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
        tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
    )
    claim_train_metrics = _apply_axis_robust_gating(claim_train_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
    claim_test_metrics = _apply_axis_robust_gating(claim_test_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))

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
    )
    summary["parity_confound_gate"] = parity_confound_gate
    geometry_null_strata: dict[str, dict[str, Any]] = {}
    geometry_null_names = ("theta_roll", "center_jitter", "mirror_axis_jitter", "radial_scramble")
    for j, name in enumerate(geometry_null_names):
        transform = null_transforms[name]
        transformed_claim = transform(claim_test_df, rng=np.random.default_rng(int(args.seed) + 9600 + j))
        geom_metrics = _evaluate_cases(
            transformed_claim,
            dataset_dir=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 9700 + j,
            enable_time_frequency_knee=bool(args.enable_time_frequency_knee),
            tf_knee_bic_delta_min=float(args.tf_knee_bic_delta_min),
        )
        geom_metrics = _apply_axis_robust_gating(geom_metrics, axis_robust=bool(axis_meta.get("axis_robust", False)))
        geometry_null_strata[name] = _summarize_case_metrics_by_strata(geom_metrics, p_lock_threshold=float(args.p_lock_threshold))
    summary["geometry_null_strata"] = geometry_null_strata
    geometry_null_collapse = _build_geometry_null_collapse_gate(
        observed_strata=claim_strata.get("test", {}),
        null_strata=geometry_null_strata,
        min_rel_drop=float(args.null_collapse_min_drop),
        min_abs_drop=float(args.null_collapse_min_abs_drop),
        required_checks=("theta_roll", "center_jitter"),
    )
    summary["geometry_null_collapse"] = geometry_null_collapse
    summary["geometry_null_by_mode"] = _build_geometry_null_by_mode(
        samples=test_df,
        seed=int(args.seed) + 9900,
    )
    vortex_centered = _is_vortex_centered_context(samples=test_df)
    angular_witness = _build_angular_witness_report(
        observed_samples=claim_test_df,
        theta_roll_samples=null_transforms["theta_roll"](claim_test_df, rng=np.random.default_rng(int(args.seed) + 9801)),
        center_jitter_samples=null_transforms["center_jitter"](claim_test_df, rng=np.random.default_rng(int(args.seed) + 9802)),
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
    claim_reason_codes: list[str] = []
    if not anomaly_mode_gate.get("passed", False):
        claim_reason_codes.append("anomaly_canonical_fail")
    if not parity_confound_gate.get("passed", False):
        claim_reason_codes.append("mode_weak_margin")
    if not geometry_null_collapse.get("passed", False):
        claim_reason_codes.append("mode_null_fail")
    if not angular_witness.get("gate", {}).get("passed", False):
        claim_reason_codes.append("non_angular_signal")
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
        ),
        "reasons": [
            *([] if anomaly_mode_gate.get("passed", False) else ["anomaly_canonical_fail"]),
            *([] if parity_confound_gate.get("passed", False) else ["parity_confound_high"]),
            *([] if geometry_null_collapse.get("passed", False) else ["geometry_nulls_do_not_collapse_signal"]),
            *([] if angular_witness.get("gate", {}).get("passed", False) else ["angular_witness_fail"]),
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
    ibtracs_strict_eval = _build_ibtracs_strict_eval(
        case_metrics=claim_test_metrics,
        samples=claim_test_df,
        radius_km=float(args.ibtracs_strict_radius_km),
        far_min_km=float(args.ibtracs_strict_far_min_km),
        time_hours=float(args.ibtracs_strict_time_hours),
        p_lock_threshold=float(args.p_lock_threshold),
    )
    summary["ibtracs_strict_eval"] = ibtracs_strict_eval

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
    geom_audit_path = out_path.with_name("geometry_null_audit.json")
    geom_audit_path.write_text(json.dumps(geometry_null_audit, indent=2, sort_keys=True), encoding="utf-8")
    summary["geometry_null_audit_path"] = str(geom_audit_path.resolve())
    confound_path = out_path.with_name("parity_confound_dashboard.parquet")
    confound_path.parent.mkdir(parents=True, exist_ok=True)
    confound_df.to_parquet(confound_path, index=False)
    summary["parity_confound_dashboard_path"] = str(confound_path.resolve())
    knee_table_path = out_path.with_name("knee_candidate_table.parquet")
    knee_candidate_table.to_parquet(knee_table_path, index=False)
    summary["knee_candidate_table_path"] = str(knee_table_path.resolve())
    knee_weather_path = out_path.with_name("knee_weather_acceptance.parquet")
    knee_weather["per_case"].to_parquet(knee_weather_path, index=False)
    summary["knee_weather_acceptance_path"] = str(knee_weather_path.resolve())
    ibtracs_strict_path = out_path.with_name("ibtracs_strict_eval.json")
    ibtracs_strict_path.write_text(json.dumps(ibtracs_strict_eval, indent=2, sort_keys=True), encoding="utf-8")
    summary["ibtracs_strict_eval_path"] = str(ibtracs_strict_path.resolve())

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
    out["O"] = vals
    out = out.loc[np.isfinite(out["O"]) & (out["O"] > 0)].copy()
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
    o_vals = pd.to_numeric(out.get("O"), errors="coerce")
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
    out["O"] = o_vals
    out = out.loc[np.isfinite(out["O"]) & (out["O"] > 0)].copy()
    return out


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
) -> dict[str, Any]:
    events = (strata_test or {}).get("events", {}) or {}
    far = (strata_test or {}).get("far_nonstorm", {}) or {}
    event_rate = float(events.get("parity_signal_rate", 0.0))
    far_rate = float(far.get("parity_signal_rate", 0.0))
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
    }


def _build_geometry_null_collapse_gate(
    *,
    observed_strata: dict[str, Any],
    null_strata: dict[str, dict[str, Any]],
    min_rel_drop: float,
    min_abs_drop: float,
    required_checks: tuple[str, ...] = ("theta_roll", "center_jitter"),
) -> dict[str, Any]:
    events = (observed_strata or {}).get("events", {}) or {}
    observed = float(events.get("parity_signal_rate", 0.0))
    rows: dict[str, Any] = {}
    passed = True
    check_names = [str(k) for k in (null_strata or {}).keys()]
    for name in check_names:
        strata = (null_strata or {}).get(name, {}) or {}
        null_events = (strata or {}).get("events", {}) or {}
        null_rate = float(null_events.get("parity_signal_rate", 0.0))
        abs_drop = float(observed - null_rate)
        rel_drop = float(abs_drop / max(observed, 1e-9))
        ok = bool((abs_drop >= float(min_abs_drop)) and (rel_drop >= float(min_rel_drop)))
        required = str(name) in set(str(v) for v in required_checks)
        rows[name] = {
            "event_parity_rate_observed": observed,
            "event_parity_rate_null": null_rate,
            "absolute_drop": abs_drop,
            "relative_drop": rel_drop,
            "required": bool(required),
            "passed": ok,
        }
        if required:
            passed = passed and ok
    return {
        "passed": bool(passed),
        "min_relative_drop": float(min_rel_drop),
        "min_absolute_drop": float(min_abs_drop),
        "required_checks": [str(v) for v in required_checks],
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
        storm_distance_cohort = (
            str(grp["storm_distance_cohort"].iloc[0])
            if "storm_distance_cohort" in grp.columns and grp["storm_distance_cohort"].notna().any()
            else None
        )
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
                "storm_id": storm_id,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["anchor_time"] = pd.to_datetime(out["anchor_time"], errors="coerce")
    return out.dropna(subset=["anchor_time"]).reset_index(drop=True)


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
    storm = pd.to_numeric(sub.get("storm_max"), errors="coerce").fillna(0).astype(int)
    near = pd.to_numeric(sub.get("near_storm_max"), errors="coerce").fillna(0).astype(int)
    pregen = pd.to_numeric(sub.get("pregen_max"), errors="coerce").fillna(0).astype(int)
    labels = np.where(
        (storm == 1) | (pregen == 1),
        "events",
        np.where(near == 1, "near_storm", np.where((storm == 0) & (near == 0) & (pregen == 0), "far_nonstorm", "other")),
    )
    sub["split_stratum"] = labels
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


def _evaluate_cases(
    samples: pd.DataFrame,
    *,
    dataset_dir: Path,
    config_path: Path,
    seed: int,
    enable_time_frequency_knee: bool,
    tf_knee_bic_delta_min: float,
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
        out_rows.append(
            {
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
                    str(case_df["storm_distance_cohort"].iloc[0])
                    if "storm_distance_cohort" in case_df.columns and case_df["storm_distance_cohort"].notna().any()
                    else None
                ),
                "nearest_storm_distance_km": _to_float(pd.to_numeric(case_df.get("nearest_storm_distance_km"), errors="coerce").mean()),
                "storm_max": int(pd.to_numeric(case_df.get("storm", 0), errors="coerce").fillna(0).max()),
                "near_storm_max": int(pd.to_numeric(case_df.get("near_storm", 0), errors="coerce").fillna(0).max()),
                "pregen_max": int(pd.to_numeric(case_df.get("pregen", 0), errors="coerce").fillna(0).max()),
                "knee_detected": bool(result.get("knee_detected", False)),
                "knee_size_detected": bool(result.get("knee_detected", False)),
                "knee_confidence": _to_float(result.get("knee_confidence")),
                "P_lock": _to_float(result.get("P_lock")),
                "parity_signal_pass": bool(result.get("parity_signal_pass", False)),
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
        )
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


def _time_permutation_within_lead(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    df = samples.copy()
    out = []
    for _, grp in df.groupby(["lead_bucket", "L", "hand"], dropna=False):
        g = grp.copy()
        vals = g["O"].to_numpy(dtype=float)
        rng.shuffle(vals)
        g["O"] = vals
        out.append(g)
    return pd.concat(out, ignore_index=True)


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

        out.append(g)
    return pd.concat(out, ignore_index=True)


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

        if "O_local_frame" in g.columns:
            vals = pd.to_numeric(g["O_local_frame"], errors="coerce").to_numpy(dtype=float)
            g["O_local_frame"] = np.where(np.isfinite(vals), 0.5 * vals, vals)
        out_frames.append(g)
    df = pd.concat(out_frames, ignore_index=True)
    return df


def _geometry_null_transform_by_name(name: str) -> NullTransform | None:
    mapping: dict[str, NullTransform] = {
        "theta_roll": _theta_roll_polar,
        "center_jitter": _center_jitter_polar,
        "mirror_axis_jitter": _mirror_axis_jitter,
        "radial_scramble": _radial_scramble_polar,
        "radial_shuffle": _radial_shuffle_polar,
    }
    return mapping.get(str(name))


def _summarize_case_metrics_by_type(case_metrics: pd.DataFrame, p_lock_threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
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
    events = case_metrics.loc[
        (pd.to_numeric(case_metrics["storm_max"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(case_metrics["near_storm_max"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(case_metrics["pregen_max"], errors="coerce").fillna(0) == 1)
    ]
    near_only = case_metrics.loc[
        (pd.to_numeric(case_metrics["storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(case_metrics["near_storm_max"], errors="coerce").fillna(0) == 1)
    ]
    far_nonstorm = case_metrics.loc[
        (pd.to_numeric(case_metrics["storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(case_metrics["near_storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(case_metrics["pregen_max"], errors="coerce").fillna(0) == 0)
    ]
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
    storm = pd.to_numeric(samples.get("storm", 0), errors="coerce").fillna(0)
    near = pd.to_numeric(samples.get("near_storm", 0), errors="coerce").fillna(0)
    pregen = pd.to_numeric(samples.get("pregen", 0), errors="coerce").fillna(0)
    out = {
        "all": samples.copy(),
        "events": samples.loc[(storm == 1) | (near == 1) | (pregen == 1)].copy(),
        "near_storm_only": samples.loc[(storm == 0) & (near == 1)].copy(),
        "far_nonstorm": samples.loc[(storm == 0) & (near == 0) & (pregen == 0)].copy(),
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
    event_mask = (
        (pd.to_numeric(test["storm_max"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(test["near_storm_max"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(test["pregen_max"], errors="coerce").fillna(0) == 1)
    )
    far_mask = (
        (pd.to_numeric(test["storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(test["near_storm_max"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(test["pregen_max"], errors="coerce").fillna(0) == 0)
    )
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
            "axis_robust": False,
            "available": False,
            "reason": "missing_lon0_sensitivity",
            "std_threshold": float(std_threshold),
        }

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "axis_robust": False,
            "available": False,
            "reason": "invalid_lon0_sensitivity_json",
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
    idx = frame.index
    if {"storm_max", "near_storm_max", "pregen_max"}.issubset(set(frame.columns)):
        storm = pd.to_numeric(frame.get("storm_max"), errors="coerce").fillna(0)
        near = pd.to_numeric(frame.get("near_storm_max"), errors="coerce").fillna(0)
        pregen = pd.to_numeric(frame.get("pregen_max"), errors="coerce").fillna(0)
        mask = ((storm >= 1) | (near >= 1) | (pregen >= 1)).reindex(idx, fill_value=False).astype(bool)
        if bool(mask.any()) or ("case_type" not in frame.columns):
            return mask
    if "storm_distance_cohort" in frame.columns:
        cohort = frame["storm_distance_cohort"].fillna("unknown").astype(str).str.lower()
        mask = cohort.isin({"event", "events", "near_storm", "near"}).reindex(idx, fill_value=False).astype(bool)
        if bool(mask.any()) or ("case_type" not in frame.columns):
            return mask
    if "case_type" in frame.columns:
        return _event_mask_from_case_type(frame["case_type"]).reindex(idx, fill_value=False).astype(bool)
    return pd.Series(False, index=idx, dtype=bool)


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
    chosen_nulls = null_names or ["theta_roll", "center_jitter", "mirror_axis_jitter", "radial_scramble"]
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
                "witness_score",
            ]
        )
    req = {"case_id", "hand"}
    if not req.issubset(set(samples.columns)):
        return pd.DataFrame(
            columns=[
                "case_id",
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
        harmonic = np.nan
        if left.size > 0 and right.size > 0:
            n = min(left.size, right.size)
            den = np.abs(left[:n]) + np.abs(right[:n]) + 1e-8
            harmonic = float(np.nanmedian(np.abs(left[:n] - right[:n]) / den))
        chiral_score = float(np.nanmedian(np.abs(chiral))) if chiral.size > 0 else np.nan
        odd_score = float(np.nanmedian(np.abs(odd))) if odd.size > 0 else np.nan
        eta_score = float(np.nanmedian(np.abs(eta))) if eta.size > 0 else np.nan
        vals = np.asarray([harmonic, chiral_score, odd_score, eta_score], dtype=float)
        vals = vals[np.isfinite(vals)]
        ang_coh = float(np.nanmedian(vals)) if vals.size > 0 else np.nan
        hs = harmonic if np.isfinite(harmonic) else np.nan
        os = odd_score if np.isfinite(odd_score) else np.nan
        if np.isfinite(hs) and np.isfinite(os):
            # Favor angular asymmetry while penalizing even-ish odd-ratio inflation.
            witness_score = float(hs - 0.2 * os)
        elif np.isfinite(hs):
            witness_score = float(hs)
        elif np.isfinite(os):
            witness_score = float(-0.2 * os)
        else:
            witness_score = np.nan
        out_rows.append(
            {
                "case_id": str(case_id),
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
    center_jitter_samples: pd.DataFrame,
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
    center = _compute_case_angular_scores(center_jitter_samples)
    if obs.empty:
        return {"gate": {"required": bool(required), "passed": False, "reason": "no_polar_scores"}}

    obs_scores_raw = pd.to_numeric(obs.get("witness_score"), errors="coerce").to_numpy(dtype=float)
    obs_scores_raw = obs_scores_raw[np.isfinite(obs_scores_raw)]
    center_ref = float(np.median(obs_scores_raw)) if obs_scores_raw.size > 0 else 0.0
    mad_ref = float(np.median(np.abs(obs_scores_raw - center_ref))) if obs_scores_raw.size > 0 else 0.0
    sd_ref = float(np.std(obs_scores_raw, ddof=1)) if obs_scores_raw.size > 1 else 0.0
    scale_ref = float(max(mad_ref, 0.25 * sd_ref, 1e-6))

    def _summary(df: pd.DataFrame) -> dict[str, float]:
        if df is None or df.empty:
            return {"event_mean": 0.0, "far_mean": 0.0, "margin": 0.0, "event_rate": 0.0, "far_rate": 0.0}
        ev = _event_mask_from_frame(df)
        fv = ~ev
        s_raw = pd.to_numeric(df.get("witness_score"), errors="coerce").to_numpy(dtype=float)
        s = (s_raw - center_ref) / max(scale_ref, 1e-9)
        se = s[ev.to_numpy(dtype=bool)] if bool(ev.any()) else np.array([], dtype=float)
        sf = s[fv.to_numpy(dtype=bool)] if bool(fv.any()) else np.array([], dtype=float)
        se = se[np.isfinite(se)]
        sf = sf[np.isfinite(sf)]
        event_mean = float(np.mean(se)) if se.size > 0 else 0.0
        far_mean = float(np.mean(sf)) if sf.size > 0 else 0.0
        return {
            "event_mean": event_mean,
            "far_mean": far_mean,
            "margin": float(event_mean - far_mean),
            "event_rate": float(np.mean(se > 0.0)) if se.size > 0 else 0.0,
            "far_rate": float(np.mean(sf > 0.0)) if sf.size > 0 else 0.0,
        }

    obs_sum = _summary(obs)
    theta_sum = _summary(theta)
    center_sum = _summary(center)

    ev_mask = _event_mask_from_frame(obs)
    w_obs = pd.to_numeric(obs.get("witness_score"), errors="coerce").to_numpy(dtype=float)
    ev_vals = w_obs[ev_mask.to_numpy(dtype=bool)]
    far_vals = w_obs[(~ev_mask).to_numpy(dtype=bool)]
    ev_vals = ev_vals[np.isfinite(ev_vals)]
    far_vals = far_vals[np.isfinite(far_vals)]
    d_obs = _cohen_d(ev_vals, far_vals)
    labels = ev_mask.astype(int).to_numpy(dtype=int)
    groups = obs.get("lead_bucket", pd.Series("", index=obs.index)).astype(str).to_numpy(dtype=object)
    p_obs = _permutation_pvalue_margin(
        scores=w_obs,
        labels=labels,
        groups=groups,
        rng=np.random.default_rng(91013),
        n_perm=int(permutation_n),
    )

    margin_obs = float(obs_sum["margin"])
    margin_theta = float(theta_sum["margin"])
    margin_center = float(center_sum["margin"])
    drop_theta_abs = float(margin_obs - margin_theta)
    drop_theta_rel = float(drop_theta_abs / max(abs(margin_obs), 1e-9))
    drop_center_abs = float(margin_obs - margin_center)
    drop_center_rel = float(drop_center_abs / max(abs(margin_obs), 1e-9))
    if not required:
        gate_passed = True
        reason = "not_required"
    else:
        gate_passed = bool(
            (margin_obs >= float(margin_min))
            and (d_obs >= float(d_min))
            and (p_obs <= float(p_max))
            and (margin_theta <= float(null_margin_max))
            and (margin_center <= float(null_margin_max))
            and (drop_theta_rel >= float(null_drop_min))
            and (drop_theta_abs >= float(null_abs_drop_min))
            and (drop_center_rel >= float(null_drop_min))
            and (drop_center_abs >= float(null_abs_drop_min))
        )
        reason = "ok" if gate_passed else "threshold_fail"
    return {
        "observed": {
            "event_mean": float(obs_sum["event_mean"]),
            "far_mean": float(obs_sum["far_mean"]),
            "margin": float(margin_obs),
            "event_rate": float(obs_sum["event_rate"]),
            "far_rate": float(obs_sum["far_rate"]),
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
        "center_jitter": {
            "event_mean": float(center_sum["event_mean"]),
            "far_mean": float(center_sum["far_mean"]),
            "margin": float(margin_center),
            "event_rate": float(center_sum["event_rate"]),
            "far_rate": float(center_sum["far_rate"]),
            "drop_abs_event": drop_center_abs,
            "drop_rel_event": drop_center_rel,
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
    recommended = "tf" if (tf_rate >= size_rate) else "size"
    notes: list[str] = []
    if size_rate < 0.15 and tf_rate > 0.0:
        notes.append("size_knee_sparse_tf_primary_recommended")
    if tf_rate < 0.15 and size_rate >= 0.15:
        notes.append("tf_knee_sparse_size_primary_recommended")
    if not notes:
        notes.append("dual_knee_reporting_recommended")
    return {
        "n_cases": n,
        "size_knee_rate": size_rate,
        "tf_knee_rate": tf_rate,
        "effective_knee_rate": eff_rate,
        "both_knees_rate": float(both / n) if n else 0.0,
        "knee_agreement_rate": float(both / max(1, any_knee)),
        "recommended_primary": recommended,
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
) -> dict[str, Any]:
    if case_metrics is None or case_metrics.empty:
        return {"available": False, "reason": "no_case_metrics"}

    case_df = case_metrics.copy()
    case_df["case_id"] = case_df["case_id"].astype(str)

    # Fill distance/time metadata from samples if missing in case_metrics.
    meta_cols = [c for c in ("case_id", "nearest_storm_distance_km") if c in samples.columns]
    time_delta_candidates = [
        c
        for c in (
            "nearest_storm_time_delta_h",
            "nearest_storm_time_delta_hours",
            "storm_time_delta_h",
            "storm_time_delta_hours",
            "ibtracs_time_delta_h",
            "ibtracs_time_delta_hours",
        )
        if c in samples.columns
    ]
    if time_delta_candidates:
        meta_cols.append(time_delta_candidates[0])
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
                    **({"nearest_storm_distance_km": "median"} if "nearest_storm_distance_km" in meta_cols else {}),
                    **({time_delta_candidates[0]: "median"} if time_delta_candidates else {}),
                    **({"storm_id_nearest": "first"} if "storm_id_nearest" in meta_cols else {}),
                    **({"storm_id": "first"} if "storm_id" in meta_cols else {}),
                }
            )
        )
        case_df = case_df.merge(meta, on="case_id", how="left", suffixes=("", "_sample"))

    dist = pd.to_numeric(case_df.get("nearest_storm_distance_km"), errors="coerce")
    if dist.isna().all():
        return {"available": False, "reason": "missing_nearest_storm_distance_km"}

    event_mask = dist <= float(radius_km)
    if time_delta_candidates:
        td = pd.to_numeric(case_df.get(time_delta_candidates[0]), errors="coerce").abs()
        event_mask = event_mask & (td <= float(time_hours))
    far_mask = dist >= float(far_min_km)
    event_df = case_df.loc[event_mask].copy()
    far_df = case_df.loc[far_mask].copy()
    if event_df.empty or far_df.empty:
        return {
            "available": False,
            "reason": "insufficient_strict_or_far_cases",
            "n_event_cases": int(event_df.shape[0]),
            "n_far_cases": int(far_df.shape[0]),
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
        "radius_km": float(radius_km),
        "far_min_km": float(far_min_km),
        "time_hours": float(time_hours),
        "n_event_cases": int(event_df.shape[0]),
        "n_far_cases": int(far_df.shape[0]),
        "events": event_summary,
        "far_controls": far_summary,
        "event_minus_far_parity_rate": float(event_rate - far_rate),
        "confound_rate": float(far_rate / max(event_rate, 1e-9)),
        "by_lead": by_lead,
    }


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
            },
            "anomaly_gate": {
                "selected_all_leads_required": True,
                "robust_far_all_leads_required": True,
                "agreement_mean_min": float(getattr(args, "anomaly_agreement_mean_min", 0.0)),
                "agreement_min_min": float(getattr(args, "anomaly_agreement_min_min", 0.0)),
            },
            "geometry_null_collapse": {
                "required_checks": ["theta_roll", "center_jitter"],
                "relative_drop_min": float(getattr(args, "null_collapse_min_drop", 0.0)),
                "absolute_drop_min": float(getattr(args, "null_collapse_min_abs_drop", 0.0)),
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
        for gate in ("parity_confound_gate", "anomaly_gate", "geometry_null_collapse", "angular_witness"):
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
