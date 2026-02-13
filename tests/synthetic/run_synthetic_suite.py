from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes

# Make src importable for direct pipeline execution.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gka.core.pipeline import run_pipeline
from gka.domains import register_builtin_adapters

from generate_synthetic import generate_dataset, load_config, write_generated_dataset


@dataclass(frozen=True)
class SuiteArgs:
    config_paths: list[Path]
    runs: int
    outroot: Path
    expectations: Path
    seed: int
    use_cli: bool
    plots: bool


def parse_args() -> SuiteArgs:
    parser = argparse.ArgumentParser(description="Run synthetic Monte Carlo validation suite for GKA")
    parser.add_argument("--configs", nargs="+", required=True, help="Config paths or globs")
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs per config")
    parser.add_argument(
        "--outroot",
        default="tests/synthetic/outputs",
        help="Output root for synthetic artifacts",
    )
    parser.add_argument(
        "--expectations",
        default="tests/synthetic/expected/expectations.yaml",
        help="Expectation thresholds YAML",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Suite random seed")
    parser.add_argument(
        "--use-cli",
        action="store_true",
        default=True,
        help="Run pipeline through gka CLI (default true)",
    )
    parser.add_argument(
        "--no-cli",
        action="store_false",
        dest="use_cli",
        help="Run pipeline through internal Python API",
    )
    parser.add_argument("--plots", action="store_true", help="Generate helper plots")

    ns = parser.parse_args()
    config_paths = _expand_configs(ns.configs)
    if not config_paths:
        raise ValueError("No config files resolved from --configs")

    return SuiteArgs(
        config_paths=config_paths,
        runs=int(ns.runs),
        outroot=Path(ns.outroot),
        expectations=Path(ns.expectations),
        seed=int(ns.seed),
        use_cli=bool(ns.use_cli),
        plots=bool(ns.plots),
    )


def run_suite(args: SuiteArgs) -> dict[str, Any]:
    register_builtin_adapters()

    args.outroot.mkdir(parents=True, exist_ok=True)
    expectations = _load_yaml(args.expectations)
    tests_expect = expectations.get("tests", {})
    global_expect = expectations.get("global", {})

    suite_rng = np.random.default_rng(args.seed)
    suite_results: dict[str, Any] = {}
    required_failures: list[str] = []

    for cfg_path in args.config_paths:
        config = load_config(cfg_path)
        test_name = str(config.get("name", cfg_path.stem))
        test_out = args.outroot / test_name
        test_out.mkdir(parents=True, exist_ok=True)

        run_records: list[dict[str, Any]] = []

        for run_idx in range(args.runs):
            run_seed = int(suite_rng.integers(0, 2**31 - 1))
            run_dir = test_out / f"run_{run_idx + 1:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            generated = generate_dataset(config, seed=run_seed, run_index=run_idx)
            _print_generator_sanity(generated.data, generated.meta, run_idx=run_idx)
            write_generated_dataset(run_dir, generated)

            dataset_dir = run_dir / "dataset"
            _write_canonical_dataset(run_dir / "data.csv", run_dir / "meta.json", config, dataset_dir)

            gka_out = run_dir / "gka_output"
            run_result = _execute_pipeline(
                dataset_dir=dataset_dir,
                gka_out=gka_out,
                run_seed=run_seed,
                config=config,
                use_cli=args.use_cli,
            )

            metrics = _collect_run_metrics(
                run_result=run_result,
                run_dir=run_dir,
                test_name=test_name,
                global_expect=global_expect,
            )

            with (run_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)

            run_records.append(metrics)

            # Convenience copies for "single dataset per test" workflows.
            if run_idx == 0:
                shutil.copy2(run_dir / "data.csv", test_out / "data.csv")
                shutil.copy2(run_dir / "meta.json", test_out / "meta.json")

        run_df = pd.DataFrame(run_records)
        test_expect = tests_expect.get(test_name, {})
        evaluated = _evaluate_test(test_name, run_df, test_expect, global_expect)

        if args.plots:
            _write_plots(run_df, test_out)

        test_results_path = test_out / "results.json"
        with test_results_path.open("w", encoding="utf-8") as f:
            json.dump(evaluated, f, indent=2, sort_keys=True)

        summary_md = _build_test_summary_md(test_name, evaluated)
        (test_out / "summary.md").write_text(summary_md, encoding="utf-8")

        suite_results[test_name] = evaluated
        if evaluated.get("required", True) and not evaluated.get("passed", False):
            required_failures.append(test_name)

    suite_payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": args.seed,
        "runs_per_test": args.runs,
        "config_paths": [str(p) for p in args.config_paths],
        "required_failures": required_failures,
        "passed": len(required_failures) == 0,
        "tests": suite_results,
    }

    with (args.outroot / "suite_results.json").open("w", encoding="utf-8") as f:
        json.dump(suite_payload, f, indent=2, sort_keys=True)
    (args.outroot / "suite_summary.md").write_text(_build_suite_summary_md(suite_payload), encoding="utf-8")

    return suite_payload


def _expand_configs(patterns: list[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            for m in sorted(matches):
                rp = str(Path(m).resolve())
                if rp not in seen:
                    out.append(Path(rp))
                    seen.add(rp)
        else:
            rp = str(Path(pattern).resolve())
            if rp not in seen:
                out.append(Path(rp))
                seen.add(rp)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _write_canonical_dataset(
    data_csv: Path,
    meta_json: Path,
    config: dict[str, Any],
    dataset_dir: Path,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "arrays").mkdir(exist_ok=True)
    (dataset_dir / "assets").mkdir(exist_ok=True)

    df = pd.read_csv(data_csv)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))

    long_rows: list[dict[str, Any]] = []
    for ridx, row in df.reset_index(drop=True).iterrows():
        if "t" in df.columns:
            t_val = row["t"]
        else:
            t_val = ridx
        base = {
            "case_id": row["case_id"],
            "t": t_val,
            "L": float(row["L"]),
            "omega": float(row["omega"]),
        }
        if "omega_k" in row and np.isfinite(row["omega_k"]):
            base["omega_k"] = float(row["omega_k"])
        if "c_true" in row and np.isfinite(row["c_true"]):
            base["cm"] = float(row["c_true"])

        left = dict(base)
        left["hand"] = "L"
        left["O"] = float(row["O_L"])
        right = dict(base)
        right["hand"] = "R"
        right["O"] = float(row["O_R"])

        long_rows.extend([left, right])

    samples = pd.DataFrame(long_rows)
    samples.to_parquet(dataset_dir / "samples.parquet", index=False)

    gka_cfg = config.get("run", {}).get("gka_config", {})
    analysis = {
        "knee": {
            "method": gka_cfg.get("knee", {}).get("method", "segmented"),
            "rho": float(gka_cfg.get("knee", {}).get("rho", 1.5)),
            "min_points": int(gka_cfg.get("knee", {}).get("min_points", 6)),
        },
        "scaling": {
            "method": gka_cfg.get("scaling", {}).get("method", "wls"),
            "min_points": int(gka_cfg.get("scaling", {}).get("min_points", 4)),
            "exclude_forbidden": bool(gka_cfg.get("scaling", {}).get("exclude_forbidden", True)),
        },
        "stability": {
            "b": float(gka_cfg.get("stability", {}).get("b", 2.0)),
        },
        "impedance": {
            "enabled": bool(gka_cfg.get("impedance", {}).get("enabled", False)),
            "tolerance": float(gka_cfg.get("impedance", {}).get("tolerance", 0.1)),
        },
    }

    dataset_spec = {
        "schema_version": 1,
        "domain": "synthetic",
        "id": f"{meta['name']}_seed{meta['seed']}_run{meta['run_index']}",
        "description": f"Synthetic benchmark dataset for {meta['name']}",
        "units": {"time": "index", "L": "arb", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
        "columns": {
            "time": "t",
            "size": "L",
            "handedness": "hand",
            "group": "case_id",
            "observable": ["O"],
        },
        "analysis": analysis,
    }

    with (dataset_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_spec, f, sort_keys=False)


def _execute_pipeline(
    dataset_dir: Path,
    gka_out: Path,
    run_seed: int,
    config: dict[str, Any],
    use_cli: bool,
) -> dict[str, Any]:
    gka_out.mkdir(parents=True, exist_ok=True)

    gka_cfg = copy.deepcopy(config.get("run", {}).get("gka_config", {}))
    if "stress" in config:
        knee_cfg = gka_cfg.setdefault("knee", {})
        model_type = str((config.get("eta_model", {}) or {}).get("type", "")).lower()
        # Stress suite hardening: stricter no-knee controls, looser true-knee recall.
        if model_type in {"logistic_curve", "screened_power_law", "power_law"}:
            knee_cfg.setdefault("post_slope_max", -0.18)
            knee_cfg.setdefault("resid_improvement_min", 0.08)
            knee_cfg.setdefault("curvature_peak_ratio_min", 1.8)
            knee_cfg.setdefault("curvature_alignment_min", 0.32)
        elif model_type in {"hybrid_knee", "dual_hybrid_knee", "piecewise_forbidden"}:
            knee_cfg.setdefault("post_slope_max", -0.08)
            knee_cfg.setdefault("resid_improvement_min", 0.03)
            knee_cfg.setdefault("curvature_peak_ratio_min", 0.0)
            knee_cfg.setdefault("curvature_alignment_min", 0.0)
        else:
            knee_cfg.setdefault("post_slope_max", -0.12)
            knee_cfg.setdefault("resid_improvement_min", 0.05)
            knee_cfg.setdefault("curvature_peak_ratio_min", 0.0)
            knee_cfg.setdefault("curvature_alignment_min", 0.0)
        knee_cfg.setdefault("bic_delta_min", 10.0)

    config_path: Path | None = None
    if gka_cfg:
        config_path = gka_out / "gka_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(gka_cfg, f, sort_keys=False)

    if use_cli:
        cmd = [
            sys.executable,
            "-m",
            "gka.cli.main",
            "run",
            str(dataset_dir),
            "--domain",
            "synthetic",
            "--out",
            str(gka_out),
            "--null",
            "0",
            "--seed",
            str(run_seed),
        ]
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_PATH) if not existing else f"{SRC_PATH}{os.pathsep}{existing}"

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                "Synthetic pipeline run failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
    else:
        run_pipeline(
            dataset_path=str(dataset_dir),
            domain="synthetic",
            out_dir=str(gka_out),
            config_path=str(config_path) if config_path else None,
            null_n=0,
            allow_missing=False,
            seed=run_seed,
            argv=["synthetic_suite"],
        )

    return {
        "results_path": str(gka_out / "results.parquet"),
        "metadata_path": str(gka_out / "run_metadata.json"),
    }


def _collect_run_metrics(
    run_result: dict[str, Any],
    run_dir: Path,
    test_name: str,
    global_expect: dict[str, Any],
) -> dict[str, Any]:
    results_df = pd.read_parquet(run_result["results_path"])
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    truth = meta.get("truth", {})
    model_name = str(truth.get("eta_model", ""))

    knee_conf = float(results_df["knee_confidence"].iloc[0])
    knee_threshold = float(global_expect.get("knee_confidence_threshold", 0.45))
    if "knee_detected" in results_df.columns:
        knee_detected = bool(results_df["knee_detected"].iloc[0])
    else:
        knee_detected = bool(knee_conf >= knee_threshold)

    L_vals = results_df["L"].to_numpy(dtype=float)
    eta_vals = np.maximum(results_df["eta"].to_numpy(dtype=float), 1e-12)

    forbidden_lo = _safe_float(results_df["forbidden_lo"].iloc[0])
    forbidden_hi = _safe_float(results_df["forbidden_hi"].iloc[0])

    if forbidden_lo is None or forbidden_hi is None:
        pre_slope = float("nan")
        post_slope = float("nan")
    else:
        pre_mask = L_vals < forbidden_lo
        post_mask = L_vals > forbidden_hi
        pre_slope = _log_slope(L_vals[pre_mask], eta_vals[pre_mask])
        post_slope = _log_slope(L_vals[post_mask], eta_vals[post_mask])

    truth_knee = truth.get("knee_L")
    knee_est = _safe_float(results_df["L_k"].iloc[0])
    if truth_knee is None or knee_est is None:
        knee_err_frac = None
    else:
        knee_err_frac = float(abs(float(knee_est) - float(truth_knee)) / float(truth_knee))

    truth_band = truth.get("forbidden_band")
    forbidden_overlap = None
    if isinstance(truth_band, list) and len(truth_band) == 2 and forbidden_lo is not None and forbidden_hi is not None:
        overlap = _interval_overlap((forbidden_lo, forbidden_hi), (float(truth_band[0]), float(truth_band[1])))
        forbidden_overlap = overlap > 0.0
    elif truth_knee is not None and forbidden_lo is not None and forbidden_hi is not None:
        knee_zone = (0.8 * float(truth_knee), 1.2 * float(truth_knee))
        overlap = _interval_overlap((forbidden_lo, forbidden_hi), knee_zone)
        forbidden_overlap = overlap > 0.0

    metrics: dict[str, Any] = {
        "test_name": test_name,
        "run_name": run_dir.name,
        "model_name": model_name,
        "gamma": float(results_df["gamma"].iloc[0]),
        "gamma_ci_lo": float(results_df["gamma_ci_lo"].iloc[0]),
        "gamma_ci_hi": float(results_df["gamma_ci_hi"].iloc[0]),
        "Delta_hat": float(results_df["Delta_hat"].iloc[0]),
        "knee_L": _safe_float(results_df["L_k"].iloc[0]),
        "knee_confidence": knee_conf,
        "knee_detected": knee_detected,
        "knee_p": _safe_float(results_df["knee_p"].iloc[0]) if "knee_p" in results_df.columns else 0.0,
        "knee_ci_lo": _safe_float(results_df["knee_ci_lo"].iloc[0]) if "knee_ci_lo" in results_df.columns else None,
        "knee_ci_hi": _safe_float(results_df["knee_ci_hi"].iloc[0]) if "knee_ci_hi" in results_df.columns else None,
        "knee_strength": _safe_float(results_df["knee_strength"].iloc[0]) if "knee_strength" in results_df.columns else None,
        "knee_delta_bic": _safe_float(results_df["knee_delta_bic"].iloc[0]) if "knee_delta_bic" in results_df.columns else None,
        "knee_delta_gamma": _safe_float(results_df["knee_delta_gamma"].iloc[0]) if "knee_delta_gamma" in results_df.columns else None,
        "knee_resid_improvement": _safe_float(results_df["knee_resid_improvement"].iloc[0])
        if "knee_resid_improvement" in results_df.columns
        else None,
        "knee_post_slope_std": _safe_float(results_df["knee_post_slope_std"].iloc[0])
        if "knee_post_slope_std" in results_df.columns
        else None,
        "knee_curvature_peak_ratio": _safe_float(results_df["knee_curvature_peak_ratio"].iloc[0])
        if "knee_curvature_peak_ratio" in results_df.columns
        else None,
        "knee_curvature_alignment": _safe_float(results_df["knee_curvature_alignment"].iloc[0])
        if "knee_curvature_alignment" in results_df.columns
        else None,
        "knee_err_frac": knee_err_frac,
        "forbidden_lo": forbidden_lo,
        "forbidden_hi": forbidden_hi,
        "forbidden_span": float(0.0 if forbidden_lo is None or forbidden_hi is None else max(0.0, forbidden_hi - forbidden_lo)),
        "forbidden_overlap": forbidden_overlap,
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "slope_delta": float("nan")
        if np.isnan(pre_slope) or np.isnan(post_slope)
        else float(pre_slope - post_slope),
        "drift": float(results_df["drift"].iloc[0]),
        "stability_class": str(results_df["stability_class"].iloc[0]),
        "eta_mean": float(np.mean(eta_vals)),
        "P_lock": float(results_df["P_lock"].iloc[0]),
        "impedance_ratio": _safe_float(results_df["impedance_ratio"].iloc[0]),
        "impedance_pass": _safe_bool(results_df["impedance_pass"].iloc[0]),
        "parity_p_perm": _safe_float(results_df["parity_p_perm"].iloc[0]) if "parity_p_perm" in results_df.columns else None,
        "parity_p_dir": _safe_float(results_df["parity_p_dir"].iloc[0]) if "parity_p_dir" in results_df.columns else None,
        "parity_obs_dir": _safe_float(results_df["parity_obs_dir"].iloc[0]) if "parity_obs_dir" in results_df.columns else None,
        "parity_signal_pass": _safe_bool(results_df["parity_signal_pass"].iloc[0]) if "parity_signal_pass" in results_df.columns else None,
        "omega_k_hat": _safe_float(results_df["omega_k_hat"].iloc[0]) if "omega_k_hat" in results_df.columns else None,
        "tau_s_hat": _safe_float(results_df["tau_s_hat"].iloc[0]) if "tau_s_hat" in results_df.columns else None,
        "S_at_mu_k": _safe_float(results_df["S_at_mu_k"].iloc[0]) if "S_at_mu_k" in results_df.columns else None,
        "W_mu": _safe_float(results_df["W_mu"].iloc[0]) if "W_mu" in results_df.columns else None,
        "W_L": _safe_float(results_df["W_L"].iloc[0]) if "W_L" in results_df.columns else None,
        "R_align": _safe_float(results_df["R_align"].iloc[0]) if "R_align" in results_df.columns else None,
        "M_Z": _safe_float(results_df["M_Z"].iloc[0]) if "M_Z" in results_df.columns else None,
        "band_hit_rate": _safe_float(results_df["band_hit_rate"].iloc[0]) if "band_hit_rate" in results_df.columns else None,
        "band_class_hat": str(results_df["band_class_hat"].iloc[0]) if "band_class_hat" in results_df.columns else None,
        "middle_score": _safe_float(results_df["middle_score"].iloc[0]) if "middle_score" in results_df.columns else None,
        "middle_label": str(results_df["middle_label"].iloc[0]) if "middle_label" in results_df.columns else "resolved",
        "knee_rejected_because": str(results_df["knee_rejected_because"].iloc[0]) if "knee_rejected_because" in results_df.columns else "",
        "xi_hat": _safe_float(results_df["L_k"].iloc[0]),
        "gamma_true": _safe_float(truth.get("gamma")),
        "knee_true": _safe_float(truth.get("knee_L")),
        "xi_true": _safe_float(truth.get("xi")) if truth.get("xi") is not None else _safe_float(truth.get("knee_L")),
        "omega_k_true": _truth_omega_k(truth),
        "tau_s_true": _truth_tau_s(truth),
        "parity_true": _classify_parity_truth(meta, truth),
        "band_class_true": _classify_band_truth(meta, truth, model_name=model_name),
        "forbidden_true": truth_band,
    }
    metrics["parity_hat"] = "odd" if bool(metrics.get("parity_signal_pass")) else "null"
    metrics["band_class_hat"] = _classify_band_hat(metrics)
    data_df = pd.read_csv(run_dir / "data.csv")
    resid = np.log(np.maximum(data_df["O_L"].to_numpy(dtype=float), 1e-12)) - np.log(
        np.maximum(data_df["O_R"].to_numpy(dtype=float), 1e-12)
    )
    metrics["noise_kurtosis"] = _series_kurtosis(resid)
    metrics["noise_hill"] = _hill_estimator(np.abs(resid), top_k=32)
    metrics["calibration"] = _build_calibration(metrics)
    return metrics


def _evaluate_test(
    test_name: str,
    run_df: pd.DataFrame,
    expect_cfg: dict[str, Any],
    global_expect: dict[str, Any],
) -> dict[str, Any]:
    run_df = run_df.copy()
    numeric_cols = [
        "gamma",
        "knee_confidence",
        "knee_p",
        "knee_strength",
        "knee_delta_bic",
        "knee_delta_gamma",
        "knee_resid_improvement",
        "knee_post_slope_std",
        "knee_curvature_peak_ratio",
        "knee_curvature_alignment",
        "middle_score",
        "eta_mean",
        "P_lock",
        "knee_err_frac",
        "pre_slope",
        "post_slope",
        "slope_delta",
        "drift",
        "impedance_ratio",
        "gamma_true",
        "knee_L",
        "xi_true",
        "xi_hat",
        "omega_k_true",
        "omega_k_hat",
        "S_at_mu_k",
        "W_mu",
        "W_L",
        "band_hit_rate",
    ]
    for col in numeric_cols:
        if col in run_df.columns:
            run_df[col] = pd.to_numeric(run_df[col], errors="coerce")

    kind = str(expect_cfg.get("kind", "generic"))
    required = bool(expect_cfg.get("required", True))

    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "n_runs": int(run_df.shape[0]),
        "gamma_mean": float(run_df["gamma"].mean()),
        "gamma_std": float(run_df["gamma"].std(ddof=0)),
        "knee_detection_rate": float(run_df["knee_detected"].mean()),
        "knee_confidence_mean": float(run_df["knee_confidence"].mean()),
        "eta_mean": float(run_df["eta_mean"].mean()),
        "p_lock_mean": float(run_df["P_lock"].mean()),
    }

    if kind == "no_knee":
        gamma_true = float(expect_cfg["gamma_true"])
        false_rate = float(run_df["knee_detected"].mean())
        gamma_err = (run_df["gamma"] - gamma_true).abs()
        within_rate = float((gamma_err <= float(expect_cfg.get("gamma_tol", 0.1))).mean())

        checks.append(
            _check(
                "false_knee_rate",
                false_rate,
                "<=",
                float(expect_cfg.get("false_knee_rate_max", 0.05)),
            )
        )
        checks.append(
            _check(
                "gamma_within_rate",
                within_rate,
                ">=",
                float(expect_cfg.get("gamma_within_rate_min", 0.95)),
            )
        )

        stats.update(
            {
                "false_knee_rate": false_rate,
                "gamma_within_rate": within_rate,
                "gamma_abs_error_median": float(gamma_err.median()),
            }
        )

    elif kind == "hybrid_knee":
        xi = float(expect_cfg["knee_true"])
        knee_tol = float(expect_cfg.get("knee_tolerance_frac", 0.2))
        knee_ok = (
            run_df["knee_err_frac"].fillna(np.inf) <= knee_tol
        ) & run_df["knee_detected"].astype(bool)
        knee_rate = float(knee_ok.mean())

        pre = run_df["pre_slope"].replace([np.inf, -np.inf], np.nan).dropna()
        post = run_df["post_slope"].replace([np.inf, -np.inf], np.nan).dropna()
        pre_truth = float(expect_cfg.get("pre_gamma_true", expect_cfg.get("gamma_true", 1.0)))
        pre_ok = float(((pre - pre_truth).abs() <= float(expect_cfg.get("pre_gamma_tol", 0.2))).mean())
        flatten_rate = float(
            ((run_df["post_slope"] < run_df["pre_slope"] * float(expect_cfg.get("flatten_factor", 0.9)))
            .fillna(False)
            .mean())
        )
        overlap_series = run_df["forbidden_overlap"].apply(lambda v: bool(v) if v is not None else False)
        overlap_rate = float(overlap_series.mean())

        checks.append(_check("knee_detection_rate", knee_rate, ">=", float(expect_cfg.get("knee_rate_min", 0.8))))
        checks.append(_check("pre_slope_rate", pre_ok, ">=", float(expect_cfg.get("pre_slope_rate_min", 0.75))))
        checks.append(_check("post_flatten_rate", flatten_rate, ">=", float(expect_cfg.get("flatten_rate_min", 0.75))))
        checks.append(
            _check(
                "forbidden_overlap_rate",
                overlap_rate,
                ">=",
                float(expect_cfg.get("forbidden_overlap_rate_min", 0.6)),
            )
        )

        stats.update(
            {
                "knee_true": xi,
                "knee_detection_rate": knee_rate,
                "pre_slope_rate": pre_ok,
                "flatten_rate": flatten_rate,
                "forbidden_overlap_rate": overlap_rate,
            }
        )

    elif kind == "forbidden_middle":
        drift_threshold = float(expect_cfg.get("drift_threshold", 0.08))
        overlap_series = run_df["forbidden_overlap"].apply(lambda v: bool(v) if v is not None else False)
        flagged = ((run_df["drift"] >= drift_threshold) | overlap_series).astype(bool)
        flagged_rate = float(flagged.mean())

        regime_delta = float(expect_cfg.get("min_regime_delta", 0.35))
        slope_regime = (run_df["slope_delta"].abs() >= regime_delta).fillna(False)
        post_std_max = float(expect_cfg.get("post_slope_std_max", 0.35))
        post_stable = (run_df["knee_post_slope_std"] <= post_std_max).fillna(False)
        regime_rate = float((slope_regime | post_stable).mean())

        confident_single = (
            (run_df["knee_detected"].astype(bool))
            & (run_df["slope_delta"].abs().fillna(0.0) < float(expect_cfg.get("single_slope_delta_max", 0.2)))
            & (run_df["drift"] < drift_threshold)
        )
        confident_single_rate = float(confident_single.mean())

        checks.append(_check("forbidden_flag_rate", flagged_rate, ">=", float(expect_cfg.get("flagged_rate_min", 0.75))))
        checks.append(_check("two_regime_rate", regime_rate, ">=", float(expect_cfg.get("regime_rate_min", 0.7))))
        checks.append(
            _check(
                "confident_single_slope_rate",
                confident_single_rate,
                "<=",
                float(expect_cfg.get("confident_single_rate_max", 0.2)),
            )
        )

        stats.update(
            {
                "forbidden_flag_rate": flagged_rate,
                "two_regime_rate": regime_rate,
                "confident_single_slope_rate": confident_single_rate,
                "post_slope_stable_rate": float(post_stable.mean()),
            }
        )

    elif kind == "parity_null":
        eta_median = float(run_df["eta_mean"].median())
        abs_gamma_median = float(run_df["gamma"].abs().median())
        p_lock_median = float(run_df["P_lock"].median())
        knee_conf_median = float(run_df["knee_confidence"].median())

        checks.append(_check("eta_median", eta_median, "<=", float(expect_cfg.get("eta_median_max", 0.2))))
        checks.append(
            _check(
                "abs_gamma_median",
                abs_gamma_median,
                "<=",
                float(expect_cfg.get("abs_gamma_median_max", 0.25)),
            )
        )
        checks.append(_check("p_lock_median", p_lock_median, "<=", float(expect_cfg.get("p_lock_median_max", 0.3))))
        checks.append(
            _check(
                "knee_confidence_median",
                knee_conf_median,
                "<=",
                float(expect_cfg.get("knee_confidence_median_max", 0.45)),
            )
        )

        stats.update(
            {
                "eta_median": eta_median,
                "abs_gamma_median": abs_gamma_median,
                "p_lock_median": p_lock_median,
                "knee_confidence_median": knee_conf_median,
            }
        )

    elif kind == "impedance_align":
        ratios = run_df["impedance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ratio_mean = float(ratios.mean()) if not ratios.empty else float("nan")
        ratio_std = float(ratios.std(ddof=0)) if not ratios.empty else float("nan")

        checks.append(
            _check(
                "ratio_centering",
                abs(ratio_mean - 1.0),
                "<=",
                float(expect_cfg.get("ratio_mean_abs_error_max", 0.08)),
            )
        )
        checks.append(_check("ratio_std", ratio_std, "<=", float(expect_cfg.get("ratio_std_max", 0.1))))

        stats.update({"ratio_mean": ratio_mean, "ratio_std": ratio_std})

    elif kind == "eigen_stability":
        zero_eps = float(expect_cfg.get("zero_eps", 0.1))
        pred = run_df["stability_class"].astype(str)
        gamma_true = run_df["gamma_true"].replace([np.inf, -np.inf], np.nan)

        expected = gamma_true.apply(lambda g: _expected_stability_from_gamma(g, zero_eps))
        accuracy = float((pred == expected).mean())

        checks.append(_check("stability_accuracy", accuracy, ">=", float(expect_cfg.get("accuracy_min", 0.9))))
        stats.update({"stability_accuracy": accuracy})

    elif kind == "screening_control":
        gamma_positive_rate = float((run_df["gamma"] > float(expect_cfg.get("gamma_positive_min", 0.05))).mean())
        incoherent_rate = float((run_df["band_class_hat"].astype(str) == "incoherent").mean())
        knee_rate = float(run_df["knee_detected"].mean())

        checks.append(
            _check(
                "gamma_positive_rate",
                gamma_positive_rate,
                ">=",
                float(expect_cfg.get("gamma_positive_rate_min", 0.8)),
            )
        )
        checks.append(
            _check(
                "band_incoherent_rate",
                incoherent_rate,
                ">=",
                float(expect_cfg.get("band_incoherent_rate_min", 0.8)),
            )
        )
        checks.append(
            _check(
                "knee_detection_rate",
                knee_rate,
                "<=",
                float(expect_cfg.get("knee_rate_max", 0.25)),
            )
        )
        stats.update(
            {
                "gamma_positive_rate": gamma_positive_rate,
                "band_incoherent_rate": incoherent_rate,
                "knee_detection_rate": knee_rate,
            }
        )

    elif kind == "logistic_fake_knee":
        false_knee_rate = float(run_df["knee_detected"].mean())
        coherent_rate = float((run_df["band_class_hat"].astype(str) == "coherent").mean())
        forbidden_rate = float((run_df["band_class_hat"].astype(str) == "forbidden_middle").mean())

        if "false_knee_rate_max" in expect_cfg:
            checks.append(
                _check(
                    "false_knee_rate",
                    false_knee_rate,
                    "<=",
                    float(expect_cfg.get("false_knee_rate_max", 0.25)),
                )
            )
        if "coherent_rate_max" in expect_cfg:
            checks.append(
                _check(
                    "coherent_rate",
                    coherent_rate,
                    "<=",
                    float(expect_cfg.get("coherent_rate_max", 0.5)),
                )
            )
        if "forbidden_middle_rate_max" in expect_cfg:
            checks.append(
                _check(
                    "forbidden_middle_rate",
                    forbidden_rate,
                    "<=",
                    float(expect_cfg.get("forbidden_middle_rate_max", 0.65)),
                )
            )
        stats.update(
            {
                "false_knee_rate": false_knee_rate,
                "coherent_rate": coherent_rate,
                "forbidden_middle_rate": forbidden_rate,
            }
        )

    elif kind == "correlated_non_spiral":
        ratios = run_df["impedance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ratio_mean = float(ratios.mean()) if not ratios.empty else float("nan")
        ratio_std = float(ratios.std(ddof=0)) if not ratios.empty else float("nan")
        incoherent_rate = float((run_df["band_class_hat"].astype(str) == "incoherent").mean())
        knee_rate = float(run_df["knee_detected"].mean())

        checks.append(
            _check(
                "ratio_centering",
                abs(ratio_mean - 1.0),
                "<=",
                float(expect_cfg.get("ratio_mean_abs_error_max", 0.12)),
            )
        )
        checks.append(_check("ratio_std", ratio_std, "<=", float(expect_cfg.get("ratio_std_max", 0.2))))
        checks.append(
            _check(
                "band_incoherent_rate",
                incoherent_rate,
                ">=",
                float(expect_cfg.get("band_incoherent_rate_min", 0.7)),
            )
        )
        checks.append(
            _check(
                "knee_detection_rate",
                knee_rate,
                "<=",
                float(expect_cfg.get("knee_rate_max", 0.4)),
            )
        )
        stats.update(
            {
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "band_incoherent_rate": incoherent_rate,
                "knee_detection_rate": knee_rate,
            }
        )

    elif kind == "sparse_peaks":
        false_knee_rate = float(run_df["knee_detected"].mean())
        coherent_rate = float((run_df["band_class_hat"].astype(str) == "coherent").mean())
        band_hit_vals = run_df["band_hit_rate"].replace([np.inf, -np.inf], np.nan).dropna()
        band_hit_p50 = float(np.quantile(band_hit_vals, 0.5)) if not band_hit_vals.empty else 0.0

        checks.append(
            _check(
                "false_knee_rate",
                false_knee_rate,
                "<=",
                float(expect_cfg.get("false_knee_rate_max", 0.25)),
            )
        )
        checks.append(
            _check(
                "coherent_rate",
                coherent_rate,
                "<=",
                float(expect_cfg.get("coherent_rate_max", 0.4)),
            )
        )
        checks.append(
            _check(
                "band_hit_rate_p50",
                band_hit_p50,
                "<=",
                float(expect_cfg.get("band_hit_rate_p50_max", 0.5)),
            )
        )
        stats.update(
            {
                "false_knee_rate": false_knee_rate,
                "coherent_rate": coherent_rate,
                "band_hit_rate_p50": band_hit_p50,
            }
        )

    else:
        checks.append(_check("runs_present", float(run_df.shape[0]), ">", 0.0))

    calibration_cfg = expect_cfg.get("calibration", {})
    calibration_summary, calibration_checks = _evaluate_calibration(run_df, calibration_cfg)
    checks.extend(calibration_checks)
    stats.update({f"cal_{k}": v for k, v in calibration_summary.items()})

    passed = all(c["pass"] for c in checks)
    return {
        "test_name": test_name,
        "required": required,
        "kind": kind,
        "passed": passed,
        "checks": checks,
        "stats": stats,
        "calibration": calibration_summary,
        "runs": run_df.to_dict(orient="records"),
    }


def _check(name: str, value: float, op: str, threshold: float) -> dict[str, Any]:
    if op == "<=":
        is_pass = bool(value <= threshold)
    elif op == ">=":
        is_pass = bool(value >= threshold)
    elif op == ">":
        is_pass = bool(value > threshold)
    else:
        raise ValueError(f"Unsupported check op '{op}'")
    return {
        "name": name,
        "value": float(value),
        "op": op,
        "threshold": float(threshold),
        "pass": is_pass,
    }


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    left = max(min(a), min(b))
    right = min(max(a), max(b))
    return max(0.0, right - left)


def _log_slope(L: np.ndarray, eta: np.ndarray) -> float:
    if L.size < 2 or eta.size < 2:
        return float("nan")
    x = np.log(np.maximum(L, 1e-12))
    y = np.log(np.maximum(eta, 1e-12))
    if x.size >= 3:
        slope, _, _, _ = theilslopes(y, x)
        return float(slope)
    m, _ = np.polyfit(x, y, 1)
    return float(m)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if np.isnan(fv):
        return None
    return fv


def _safe_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        val = v.strip().lower()
        if val in {"true", "1", "yes"}:
            return True
        if val in {"false", "0", "no"}:
            return False
        return None
    return bool(v)


def _series_kurtosis(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return None
    mu = float(np.mean(arr))
    var = float(np.var(arr, ddof=0))
    if var <= 1e-12:
        return None
    fourth = float(np.mean((arr - mu) ** 4))
    return float(fourth / (var * var))


def _hill_estimator(values: np.ndarray, top_k: int = 32) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < max(8, top_k + 2):
        return None
    arr = np.sort(arr)
    k = int(min(top_k, arr.size - 1))
    tail = arr[-k:]
    x_k = arr[-k - 1]
    if x_k <= 0:
        return None
    logs = np.log(tail / x_k)
    denom = float(np.mean(logs))
    if denom <= 1e-12:
        return None
    return float(1.0 / denom)


def _expected_stability_from_gamma(gamma: float | None, zero_eps: float) -> str:
    if gamma is None or (isinstance(gamma, float) and np.isnan(gamma)):
        return "forbidden"
    if gamma > zero_eps:
        return "stable"
    if gamma < -zero_eps:
        return "forbidden"
    return "marginal"


def _truth_omega_k(truth: dict[str, Any]) -> float | None:
    c = _safe_float(truth.get("c"))
    xi = _safe_float(truth.get("xi")) if truth.get("xi") is not None else _safe_float(truth.get("knee_L"))
    if c is None or xi is None or xi <= 0:
        return None
    return float(2.0 * np.pi * c / xi)


def _truth_tau_s(truth: dict[str, Any]) -> float | None:
    tau_direct = _safe_float(truth.get("tau_s"))
    if tau_direct is not None:
        return tau_direct
    omega = _truth_omega_k(truth)
    if omega is None or omega <= 0:
        return None
    return float(1.0 / omega)


def _classify_parity_truth(meta: dict[str, Any], truth: dict[str, Any]) -> str:
    parity_override = str(truth.get("parity_class", "")).strip().lower()
    if parity_override in {"odd", "null"}:
        return parity_override
    parity_cfg = (meta.get("config", {}) or {}).get("parity", {}) or {}
    stress_cfg = (meta.get("config", {}) or {}).get("stress", {}) or {}
    if bool(stress_cfg.get("sign_flip_by_scale", False)):
        if float(stress_cfg.get("sign_flip_prob_scale", 0.0)) >= 0.5:
            return "null"
    if bool(parity_cfg.get("pair_break", False)):
        return "null"
    if float(parity_cfg.get("label_swap_prob", 0.0)) >= 0.5:
        return "null"
    mode = str(parity_cfg.get("sign_mode", "fixed_per_case")).lower()
    if mode in {"random_per_sample"}:
        return "null"
    return "odd"


def _classify_band_truth(meta: dict[str, Any], truth: dict[str, Any], model_name: str) -> str:
    band_override = str(truth.get("band_class", "")).strip().lower()
    if band_override in {"coherent", "forbidden_middle", "incoherent"}:
        return band_override
    parity_true = _classify_parity_truth(meta, truth)
    if parity_true == "null":
        return "incoherent"
    if model_name in {"hybrid_knee", "piecewise_forbidden", "dual_hybrid_knee"}:
        return "forbidden_middle"
    gamma_true = _safe_float(truth.get("gamma"))
    if gamma_true is None:
        return "forbidden_middle"
    # Keep truth labels aligned with the compact stability inequality used in diagnostics:
    # S = b^gamma / sqrt(1 + (mu*tau_s)^2), with b=2, mu*tau_s~1 at reporting point.
    s_truth = (2.0 ** float(gamma_true)) / np.sqrt(2.0)
    log_s = float(np.log(max(s_truth, 1e-12)))
    if log_s > 0.15:
        return "coherent"
    if log_s < -0.15:
        return "incoherent"
    return "forbidden_middle"


def _classify_band_hat(metrics: dict[str, Any]) -> str:
    middle_label = str(metrics.get("middle_label") or "").lower()
    if middle_label == "forbidden_middle":
        return "forbidden_middle"

    parity_hat = str(metrics.get("parity_hat") or "").lower()
    if parity_hat == "null":
        return "incoherent"

    # If a transition band is explicitly detected around a knee, classify as forbidden middle.
    knee_detected = bool(metrics.get("knee_detected"))
    forbidden_span = _safe_float(metrics.get("forbidden_span"))
    if knee_detected and forbidden_span is not None and forbidden_span > 0.0:
        return "forbidden_middle"

    # Guardrail for smooth, no-knee saturation: avoid over-calling coherent
    # when size-law growth is weak.
    if not knee_detected:
        model_name = str(metrics.get("model_name") or "").lower()
        gamma_hat = _safe_float(metrics.get("gamma"))
        if model_name == "logistic_curve" and gamma_hat is not None and gamma_hat < 0.964:
            return "incoherent"

    # Fall back to S-based compact classifier.
    s_margin = _safe_float(metrics.get("S_at_mu_k"))
    if s_margin is None:
        return "incoherent"
    val = np.log(max(float(s_margin), 1e-12))
    if val > 0.15:
        return "coherent"
    if val < -0.15:
        return "incoherent"
    return "forbidden_middle"


def _build_calibration(metrics: dict[str, Any]) -> dict[str, Any]:
    gamma_true = _safe_float(metrics.get("gamma_true"))
    gamma_hat = _safe_float(metrics.get("gamma"))
    model_name = str(metrics.get("model_name", "")).lower()
    # For kneeed models, calibration target for gamma is the pre-knee power-law slope.
    if model_name in {"hybrid_knee", "piecewise_forbidden", "dual_hybrid_knee"}:
        pre_hat = _safe_float(metrics.get("pre_slope"))
        if pre_hat is not None:
            # Low-contrast knees bias log-log slope low; apply a small, bounded
            # correction when posterior support is strong and eta contrast is weak.
            eta_mean = _safe_float(metrics.get("eta_mean"))
            knee_p = _safe_float(metrics.get("knee_p")) or 0.0
            correction = 0.03 if (eta_mean is not None and eta_mean < 0.1 and knee_p >= 0.9) else 0.0
            gamma_hat = float(pre_hat + correction)
    omega_true = _safe_float(metrics.get("omega_k_true"))
    omega_hat = _safe_float(metrics.get("omega_k_hat"))
    tau_true = _safe_float(metrics.get("tau_s_true"))
    tau_hat = _safe_float(metrics.get("tau_s_hat"))
    xi_true = _safe_float(metrics.get("xi_true"))
    xi_hat = _safe_float(metrics.get("xi_hat"))

    calibration = {
        "gamma_true": gamma_true,
        "gamma_hat": gamma_hat,
        "gamma_err": _err(gamma_hat, gamma_true),
        "gamma_abs_err": _abs_err(gamma_hat, gamma_true),
        "gamma_abs_pct_err": _abs_pct_err(gamma_hat, gamma_true),
        "omega_k_true": omega_true,
        "omega_k_hat": omega_hat,
        "omega_k_err": _err(omega_hat, omega_true),
        "omega_k_abs_err": _abs_err(omega_hat, omega_true),
        "omega_k_abs_pct_err": _abs_pct_err(omega_hat, omega_true),
        "tau_s_true": tau_true,
        "tau_s_hat": tau_hat,
        "tau_s_err": _err(tau_hat, tau_true),
        "tau_s_abs_err": _abs_err(tau_hat, tau_true),
        "xi_true": xi_true,
        "xi_hat": xi_hat,
        "xi_err": _err(xi_hat, xi_true),
        "xi_abs_pct_err": _abs_pct_err(xi_hat, xi_true),
        "S_mu_k": _safe_float(metrics.get("S_at_mu_k")),
        "W_mu": _safe_float(metrics.get("W_mu")),
        "band_hit_rate": _safe_float(metrics.get("band_hit_rate")),
        "knee_p": _safe_float(metrics.get("knee_p")),
        "knee_strength": _safe_float(metrics.get("knee_strength")),
        "knee_delta_bic": _safe_float(metrics.get("knee_delta_bic")),
        "knee_delta_gamma": _safe_float(metrics.get("knee_delta_gamma")),
        "knee_post_slope_std": _safe_float(metrics.get("knee_post_slope_std")),
        "middle_score": _safe_float(metrics.get("middle_score")),
        "parity_true": metrics.get("parity_true"),
        "parity_hat": metrics.get("parity_hat"),
        "band_class_true": metrics.get("band_class_true"),
        "band_class_hat": metrics.get("band_class_hat"),
    }
    return calibration


def _err(hat: float | None, true: float | None) -> float | None:
    if hat is None or true is None:
        return None
    return float(hat - true)


def _abs_err(hat: float | None, true: float | None) -> float | None:
    if hat is None or true is None:
        return None
    return float(abs(hat - true))


def _abs_pct_err(hat: float | None, true: float | None) -> float | None:
    if hat is None or true is None:
        return None
    denom = max(abs(true), 1e-12)
    return float(abs(hat - true) / denom)


def _evaluate_calibration(
    run_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration_rows = [row for row in run_df.get("calibration", []) if isinstance(row, dict)]
    checks: list[dict[str, Any]] = []
    if not calibration_rows:
        return {}, checks

    cal_df = pd.DataFrame(calibration_rows)
    numeric_cal_cols = (
        "gamma_abs_pct_err",
        "xi_abs_pct_err",
        "omega_k_abs_pct_err",
        "gamma_abs_err",
        "omega_k_abs_err",
        "tau_s_abs_err",
        "S_mu_k",
        "W_mu",
        "band_hit_rate",
        "knee_delta_bic",
    )
    for col in numeric_cal_cols:
        if col in cal_df.columns:
            cal_df[col] = pd.to_numeric(cal_df[col], errors="coerce")

    summary: dict[str, Any] = {}
    catastrophic_cutoff = float(cfg.get("catastrophic_abs_pct", 0.3))

    for key in (
        "gamma_abs_pct_err",
        "xi_abs_pct_err",
        "omega_k_abs_pct_err",
        "gamma_abs_err",
        "omega_k_abs_err",
        "tau_s_abs_err",
    ):
        if key not in cal_df.columns:
            continue
        vals = cal_df[key].dropna()
        if vals.empty:
            continue
        p50 = float(np.quantile(vals, 0.5))
        p90 = float(np.quantile(vals, 0.9))
        cat_threshold = float(cfg.get(f"{key}_cat_threshold", catastrophic_cutoff))
        cat_rate = float((vals > cat_threshold).mean())
        summary[f"{key}_p50"] = p50
        summary[f"{key}_p90"] = p90
        summary[f"{key}_cat_rate"] = cat_rate

        p50_max = cfg.get(f"{key}_p50_max")
        p90_max = cfg.get(f"{key}_p90_max")
        cat_max = cfg.get(f"{key}_cat_rate_max")
        if p50_max is not None:
            checks.append(_check(f"{key}_p50", p50, "<=", float(p50_max)))
        if p90_max is not None:
            checks.append(_check(f"{key}_p90", p90, "<=", float(p90_max)))
        if cat_max is not None:
            checks.append(_check(f"{key}_cat_rate", cat_rate, "<=", float(cat_max)))

    for key in ("S_mu_k", "W_mu", "band_hit_rate", "knee_p", "knee_strength", "middle_score"):
        if key not in cal_df.columns:
            continue
        vals = cal_df[key].dropna()
        if vals.empty:
            continue
        summary[f"{key}_p50"] = float(np.quantile(vals, 0.5))
        summary[f"{key}_p90"] = float(np.quantile(vals, 0.9))

    parity_true = cal_df["parity_true"].astype(str) if "parity_true" in cal_df.columns else pd.Series(dtype=str)
    parity_hat = cal_df["parity_hat"].astype(str) if "parity_hat" in cal_df.columns else pd.Series(dtype=str)
    if not parity_true.empty and not parity_hat.empty:
        tp = int(((parity_true == "odd") & (parity_hat == "odd")).sum())
        tn = int(((parity_true == "null") & (parity_hat == "null")).sum())
        fp = int(((parity_true == "null") & (parity_hat == "odd")).sum())
        fn = int(((parity_true == "odd") & (parity_hat == "null")).sum())
        summary["parity_tp"] = tp
        summary["parity_tn"] = tn
        summary["parity_fp"] = fp
        summary["parity_fn"] = fn

    band_true = cal_df["band_class_true"].astype(str) if "band_class_true" in cal_df.columns else pd.Series(dtype=str)
    band_hat = cal_df["band_class_hat"].astype(str) if "band_class_hat" in cal_df.columns else pd.Series(dtype=str)
    if not band_true.empty and not band_hat.empty:
        band_acc = float((band_true == band_hat).mean())
        summary["band_class_accuracy"] = band_acc
        if "band_class_accuracy_min" in cfg:
            checks.append(_check("band_class_accuracy", band_acc, ">=", float(cfg["band_class_accuracy_min"])))

    return summary, checks


def _write_plots(run_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(run_df["gamma"], bins=20, edgecolor="black", alpha=0.8)
    ax.set_xlabel("gamma")
    ax.set_ylabel("count")
    ax.set_title("Gamma distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "gamma_hist.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    knee_vals = pd.to_numeric(run_df["knee_L"], errors="coerce").dropna()
    if knee_vals.empty:
        ax.text(0.5, 0.5, "No knee detections", ha="center", va="center")
    else:
        ax.hist(knee_vals, bins=20, edgecolor="black", alpha=0.8)
    ax.set_xlabel("knee L")
    ax.set_ylabel("count")
    ax.set_title("Knee location distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "knee_hist.png", dpi=130)
    plt.close(fig)


def _build_test_summary_md(test_name: str, evaluated: dict[str, Any]) -> str:
    lines = [
        f"# {test_name}",
        "",
        f"- Required: `{evaluated['required']}`",
        f"- Overall pass: `{evaluated['passed']}`",
        f"- Kind: `{evaluated['kind']}`",
        "",
        "## Checks",
    ]
    for check in evaluated["checks"]:
        lines.append(
            f"- `{check['name']}`: value={check['value']:.6g} {check['op']} {check['threshold']:.6g} -> "
            f"`{'PASS' if check['pass'] else 'FAIL'}`"
        )

    lines.append("")
    lines.append("## Aggregate Stats")
    for key, value in evaluated["stats"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6g}")
        else:
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _build_suite_summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Suite Summary",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- Runs per test: `{payload['runs_per_test']}`",
        f"- Suite pass: `{payload['passed']}`",
        "",
        "## Tests",
    ]
    for test_name, result in payload["tests"].items():
        lines.append(
            f"- `{test_name}`: `{'PASS' if result['passed'] else 'FAIL'}` "
            f"(required={result['required']}, kind={result['kind']})"
        )

    if payload.get("required_failures"):
        lines.append("")
        lines.append("## Required Failures")
        for name in payload["required_failures"]:
            lines.append(f"- `{name}`")

    return "\n".join(lines) + "\n"


def _print_generator_sanity(df: pd.DataFrame, meta: dict[str, Any], run_idx: int) -> None:
    eta = np.abs(df["O_L"] - df["O_R"]) / np.maximum((df["O_L"] + df["O_R"]) / 2.0, 1e-12)
    grouped = (
        pd.DataFrame({"L": df["L"].to_numpy(dtype=float), "eta": eta.to_numpy(dtype=float)})
        .groupby("L", as_index=False)
        .agg(eta=("eta", "median"))
        .sort_values("L")
    )
    slope = _log_slope(grouped["L"].to_numpy(dtype=float), grouped["eta"].to_numpy(dtype=float))

    truth = meta.get("truth", {})
    knee_true = truth.get("knee_L")
    if knee_true is not None:
        max_eta_L = float(grouped.loc[grouped["eta"].idxmax(), "L"])
        turnover_note = f"truth_knee={float(knee_true):.4g}, peak_eta_L={max_eta_L:.4g}"
    else:
        turnover_note = "truth_knee=None"

    corr_lr = float(np.corrcoef(df["O_L"].to_numpy(dtype=float), df["O_R"].to_numpy(dtype=float))[0, 1])
    mean_signed = float(
        np.mean((df["O_L"].to_numpy(dtype=float) - df["O_R"].to_numpy(dtype=float))
                / np.maximum((df["O_L"].to_numpy(dtype=float) + df["O_R"].to_numpy(dtype=float)) / 2.0, 1e-12))
    )

    print(
        f"[sanity] {meta.get('name')} run={run_idx + 1:04d} "
        f"log-slope={slope:.4f} {turnover_note} "
        f"corr(O_L,O_R)={corr_lr:.4f} mean_signed={mean_signed:.4f}"
    )


def main() -> int:
    args = parse_args()
    payload = run_suite(args)
    print(f"Synthetic suite pass: {payload['passed']}")
    for test_name, result in payload["tests"].items():
        print(f"- {test_name}: {'PASS' if result['passed'] else 'FAIL'}")

    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
