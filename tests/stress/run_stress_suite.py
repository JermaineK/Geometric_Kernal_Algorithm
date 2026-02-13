from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gka.calibrate.fit_thresholds import fit_thresholds_from_robustness, write_threshold_yaml
from gka.classify.blind_model import (
    InvariantBlindClassifier,
    expected_calibration_error,
    macro_ovr_auroc,
)

SYN_PATH = Path(__file__).resolve().parents[1] / "synthetic"
if str(SYN_PATH) not in sys.path:
    sys.path.insert(0, str(SYN_PATH))

from generate_synthetic import generate_dataset, load_config, write_generated_dataset
from run_synthetic_suite import (
    SuiteArgs,
    _collect_run_metrics,
    _execute_pipeline,
    _expand_configs,
    _write_canonical_dataset,
    run_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial stress suite for GKA")
    parser.add_argument("--configs", nargs="+", default=["tests/stress/configs/*.yaml"], help="Stress config paths or globs")
    parser.add_argument("--runs", type=int, default=200, help="Monte Carlo runs per stress config")
    parser.add_argument("--outroot", default="tests/stress/outputs", help="Output root for stress artifacts")
    parser.add_argument("--expectations", default="tests/stress/expected/expectations.yaml", help="Stress expectations YAML")
    parser.add_argument("--seed", type=int, default=24680, help="Suite random seed")
    parser.add_argument("--use-cli", action="store_true", default=True, help="Run pipeline through gka CLI (default true)")
    parser.add_argument("--no-cli", action="store_false", dest="use_cli", help="Run pipeline through internal API")
    parser.add_argument("--plots", action="store_true", help="Generate helper plots")
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness/ blind diagnostics report generation",
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=120,
        help="Number of sampled robustness sweeps for parameter stress envelope",
    )
    parser.add_argument(
        "--blind-n",
        type=int,
        default=100,
        help="Number of blinded synthetic datasets for regime classification audit",
    )
    parser.add_argument(
        "--robustness-report",
        default=None,
        help="Path for robustness_report.json (default: <outroot>/robustness_report.json)",
    )
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    config_paths = [Path(p) for p in _expand_configs(ns.configs)]
    if not config_paths:
        raise ValueError("No stress configs resolved from --configs")

    args = SuiteArgs(
        config_paths=config_paths,
        runs=int(ns.runs),
        outroot=Path(ns.outroot),
        expectations=Path(ns.expectations),
        seed=int(ns.seed),
        use_cli=bool(ns.use_cli),
        plots=bool(ns.plots),
    )
    payload = run_suite(args)
    print(f"Stress suite pass: {payload['passed']}")
    for test_name, result in payload["tests"].items():
        print(f"- {test_name}: {'PASS' if result['passed'] else 'FAIL'}")

    overall_pass = bool(payload["passed"])
    if not ns.skip_robustness:
        report_path = Path(ns.robustness_report) if ns.robustness_report else Path(ns.outroot) / "robustness_report.json"
        report = build_robustness_report(
            outroot=Path(ns.outroot),
            expectations_path=Path(ns.expectations),
            use_cli=bool(ns.use_cli),
            seed=int(ns.seed) + 1009,
            robustness_samples=int(ns.robustness_samples),
            blind_n=int(ns.blind_n),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Robustness report: {report_path}")
        print(f"- blind_accuracy: {report['blind_test']['accuracy']:.4f}")
        print(f"- blind_auroc: {report['blind_test'].get('auroc')}")
        print(f"- robustness_fp_rate: {report['parameter_robustness']['false_positive_rate']:.4f}")
        print(f"- robustness_fn_rate: {report['parameter_robustness']['false_negative_rate']:.4f}")
        robust_eval = evaluate_robustness_gates(
            report=report,
            expectations_path=Path(ns.expectations),
        )
        overall_pass = overall_pass and bool(robust_eval["passed"])
        robust_eval_path = Path(ns.outroot) / "robustness_gate.json"
        robust_eval_path.write_text(json.dumps(robust_eval, indent=2, sort_keys=True), encoding="utf-8")
        print(f"- robustness_gate_pass: {robust_eval['passed']}")
        for chk in robust_eval.get("checks", []):
            print(
                f"  - {chk['name']}: value={chk['value']:.6g} {chk['op']} {chk['threshold']:.6g} -> "
                f"{'PASS' if chk['pass'] else 'FAIL'}"
            )

    return 0 if overall_pass else 2


def build_robustness_report(
    outroot: Path,
    expectations_path: Path,
    use_cli: bool,
    seed: int,
    robustness_samples: int,
    blind_n: int,
) -> dict[str, Any]:
    outroot.mkdir(parents=True, exist_ok=True)
    global_expect = _load_global_expect(expectations_path)
    rng = np.random.default_rng(seed)

    robustness_dir = outroot / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)

    param_report = _run_parameter_robustness(
        out_dir=robustness_dir / "parameter_sweep",
        global_expect=global_expect,
        use_cli=use_cli,
        rng=rng,
        samples=max(1, robustness_samples),
    )
    blind_report = _run_blind_test(
        out_dir=robustness_dir / "blind_test",
        global_expect=global_expect,
        use_cli=use_cli,
        rng=rng,
        n=max(1, blind_n),
    )
    stability_map = _build_invariant_stability_map(epsilon=0.2)
    param_runs_path = Path(param_report["runs_path"])
    param_runs_df = pd.read_json(param_runs_path)
    threshold_payload = fit_thresholds_from_robustness(
        parameter_runs=param_runs_df,
        target_fp_max=float(global_expect.get("robustness", {}).get("false_positive_rate_max", 0.10)),
        objective_beta=1.0,
    )
    calibration_path = expectations_path.parent / "calibration.yaml"
    write_threshold_yaml(
        {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": str(param_runs_path),
            "thresholds": threshold_payload,
        },
        calibration_path,
    )

    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": int(seed),
        "parameter_robustness": param_report,
        "blind_test": blind_report,
        "invariant_stability_map": stability_map,
        "recommended_thresholds": threshold_payload,
        "recommended_thresholds_path": str(calibration_path),
    }


def _run_parameter_robustness(
    out_dir: Path,
    global_expect: dict[str, Any],
    use_cli: bool,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_hybrid = load_config("tests/stress/configs/stressB_hybrid_heavytail.yaml")
    base_no_knee = load_config("tests/stress/configs/stressA_nonstationary_no_knee.yaml")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(samples * 6, 24)

    while len(rows) < samples and attempts < max_attempts:
        idx = len(rows)
        has_knee = bool(rng.random() < 0.5)
        cfg = copy.deepcopy(base_hybrid if has_knee else base_no_knee)
        noise_amp = float(rng.uniform(0.0, 0.5))
        n_L = int(rng.integers(4, 21))
        gamma_val = float(rng.uniform(1.0, 2.0))
        xi_val = float(rng.uniform(20.0, 90.0))
        tau_s_val = float(rng.uniform(0.04, 0.4))

        _set_l_samples(cfg, n_L)
        _set_min_points(cfg, n_L)
        _set_noise_std(cfg, noise_amp)
        _set_gamma(cfg, gamma_val)
        _set_tau_and_impedance(cfg, xi_val=xi_val, tau_s=tau_s_val)

        if has_knee:
            cfg.setdefault("eta_model", {})["type"] = "hybrid_knee"
            cfg["eta_model"]["xi"] = xi_val
            cfg.setdefault("truth", {})["knee_L"] = xi_val
            cfg["truth"]["xi"] = xi_val
        else:
            cfg.setdefault("eta_model", {})["type"] = "power_law"
            cfg.setdefault("truth", {})["knee_L"] = None
            cfg["truth"]["xi"] = None
            for key in ("xi", "q", "decay_scale_frac"):
                cfg.get("eta_model", {}).pop(key, None)

        cfg["name"] = f"robustness_case_{idx:04d}"
        run_dir = out_dir / f"run_{idx + 1:04d}"
        attempts += 1
        try:
            metrics = _run_single_config(
                cfg,
                run_seed=int(rng.integers(0, 2**31 - 1)),
                run_dir=run_dir,
                use_cli=use_cli,
                global_expect=global_expect,
            )
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempts,
                    "intended_run": idx + 1,
                    "error": str(exc),
                }
            )
            continue

        row = {
            "run": idx + 1,
            "has_knee_true": has_knee,
            "noise_amp": noise_amp,
            "n_L": n_L,
            "gamma_true": gamma_val,
            "xi_true": xi_val if has_knee else None,
            "tau_s_true": tau_s_val,
            "knee_detected": bool(metrics.get("knee_detected", False)),
            "middle_label": str(metrics.get("middle_label", "")),
            "middle_score": _to_float(metrics.get("middle_score")),
            "knee_p": _to_float(metrics.get("knee_p")),
            "knee_strength": _to_float(metrics.get("knee_strength")),
            "knee_delta_bic": _to_float(metrics.get("knee_delta_bic")),
            "gamma_hat": _to_float(metrics.get("gamma")),
            "knee_hat": _to_float(metrics.get("knee_L")),
            "knee_err_frac": _to_float(metrics.get("knee_err_frac")),
            "band_class_hat": str(metrics.get("band_class_hat", "")),
            "S_at_mu_k": _to_float(metrics.get("S_at_mu_k")),
            "W_mu": _to_float(metrics.get("W_mu")),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("Parameter robustness sweep produced zero successful runs")

    df = pd.DataFrame(rows)
    df.to_json(out_dir / "parameter_runs.json", orient="records", indent=2)

    has_knee = df["has_knee_true"].astype(bool)
    knee_p_series = pd.to_numeric(df["knee_p"], errors="coerce").fillna(0.0)
    knee_strength_series = pd.to_numeric(df["knee_strength"], errors="coerce").fillna(0.0)
    knee_delta_bic_series = pd.to_numeric(df["knee_delta_bic"], errors="coerce").fillna(0.0)
    p_min = float(global_expect.get("knee_p_positive_min", 0.35))
    s_min = float(global_expect.get("knee_strength_positive_min", 0.0))
    b_min = float(global_expect.get("knee_delta_bic_positive_min", 0.0))
    soft_knee_positive = (knee_p_series >= p_min) & (
        (knee_strength_series >= s_min) | (knee_delta_bic_series >= b_min)
    )
    knee_positive = (
        df["knee_detected"].astype(bool)
        | (df["middle_label"].astype(str) == "forbidden_middle")
        | soft_knee_positive
    )
    false_positive_rate = float(knee_positive[~has_knee].mean()) if (~has_knee).any() else 0.0
    false_negative_rate = float((~knee_positive[has_knee]).mean()) if has_knee.any() else 0.0

    slope_bias = (pd.to_numeric(df["gamma_hat"], errors="coerce") - pd.to_numeric(df["gamma_true"], errors="coerce")).dropna()
    slope_bias_mean = float(slope_bias.mean()) if not slope_bias.empty else float("nan")
    slope_bias_p90_abs = float(np.quantile(np.abs(slope_bias), 0.9)) if not slope_bias.empty else float("nan")

    knee_err = pd.to_numeric(df.loc[has_knee, "knee_err_frac"], errors="coerce").dropna()
    knee_err_median = float(np.quantile(knee_err, 0.5)) if not knee_err.empty else float("nan")
    knee_err_p90 = float(np.quantile(knee_err, 0.9)) if not knee_err.empty else float("nan")

    catastrophic_rate = float((knee_err > 0.30).mean()) if not knee_err.empty else 0.0
    stability_envelopes = _build_stability_envelopes(df)

    return {
        "n_samples": int(samples),
        "n_success": int(df.shape[0]),
        "n_failures": int(len(failures)),
        "runs_path": str(out_dir / "parameter_runs.json"),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "slope_bias_mean": slope_bias_mean,
        "slope_bias_abs_p90": slope_bias_p90_abs,
        "knee_localization_err_median": knee_err_median,
        "knee_localization_err_p90": knee_err_p90,
        "catastrophic_miss_rate": catastrophic_rate,
        "stability_envelopes": stability_envelopes,
        "failures": failures,
    }


def _run_blind_test(
    out_dir: Path,
    global_expect: dict[str, Any],
    use_cli: bool,
    rng: np.random.Generator,
    n: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_to_cfgs: dict[str, list[str]] = {
        "null": ["tests/stress/configs/stressI_sign_flip_parity.yaml"],
        "hybrid": [
            "tests/stress/configs/stressB_hybrid_heavytail.yaml",
            "tests/stress/configs/stressL_low_contrast_real_knee.yaml",
            "tests/stress/configs/stressN_timewarp_knee.yaml",
        ],
        "forbidden_middle": [
            "tests/stress/configs/stressC_mixture_regimes.yaml",
            "tests/stress/configs/stressM_multiknee.yaml",
        ],
        "coherent": ["tests/stress/configs/stressA_nonstationary_no_knee.yaml"],
        "adversarial": [
            "tests/stress/configs/stressH_fake_knee_logistic.yaml",
            "tests/stress/configs/stressJ_correlated_non_spiral.yaml",
            "tests/stress/configs/stressK_sparse_spectral_peaks.yaml",
            "tests/stress/configs/stressG_heavytail_1f_no_knee.yaml",
        ],
    }
    loaded_cfgs: dict[str, list[dict[str, Any]]] = {
        label: [load_config(path) for path in paths] for label, paths in label_to_cfgs.items()
    }

    labels = list(label_to_cfgs.keys())
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(n * 6, 24)

    while len(rows) < n and attempts < max_attempts:
        idx = len(rows)
        true_label = str(labels[int(rng.integers(0, len(labels)))])
        options = loaded_cfgs[true_label]
        cfg = copy.deepcopy(options[int(rng.integers(0, len(options)))])
        cfg["name"] = f"blind_case_{idx:04d}"
        run_dir = out_dir / f"run_{idx + 1:04d}"
        attempts += 1
        try:
            metrics = _run_single_config(
                cfg,
                run_seed=int(rng.integers(0, 2**31 - 1)),
                run_dir=run_dir,
                use_cli=use_cli,
                global_expect=global_expect,
            )
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempts,
                    "intended_run": idx + 1,
                    "true_label": true_label,
                    "error": str(exc),
                }
            )
            continue
        pred_label = _predict_regime(metrics)
        invariant = _invariant_vector(metrics)
        rows.append(
            {
                "run": idx + 1,
                "true_label": true_label,
                "pred_label_rule": pred_label,
                "knee_detected": bool(metrics.get("knee_detected", False)),
                "knee_confidence": _to_float(metrics.get("knee_confidence")),
                "band_class_hat": str(metrics.get("band_class_hat", "")),
                "parity_hat": str(metrics.get("parity_hat", "")),
                "invariant": invariant,
            }
        )

    if not rows:
        raise RuntimeError("Blind test produced zero successful runs")

    df = pd.DataFrame(rows)
    X = np.vstack([np.asarray(v, dtype=float) for v in df["invariant"].tolist()])
    y = df["true_label"].astype(str).to_numpy()
    pred, prob, class_order = _cross_validated_blind_predictions(X=X, y=y, labels=np.asarray(labels), rng=rng, n_splits=5)
    df["pred_label"] = pred
    df["pred_conf"] = np.max(prob, axis=1)
    df.to_json(out_dir / "blind_predictions.json", orient="records", indent=2)

    accuracy = float(np.mean(df["true_label"].to_numpy() == df["pred_label"].to_numpy()))
    auroc = macro_ovr_auroc(y_true=y, y_prob=prob, classes=class_order)
    ece = expected_calibration_error(y_true=y, y_prob=prob, classes=class_order)
    confusion = (
        pd.crosstab(df["true_label"], df["pred_label"], dropna=False)
        .reindex(index=labels, columns=labels, fill_value=0)
        .astype(int)
    )

    return {
        "n_samples": int(n),
        "n_success": int(df.shape[0]),
        "n_failures": int(len(failures)),
        "accuracy": accuracy,
        "auroc": None if auroc is None else float(auroc),
        "ece": float(ece),
        "confusion_matrix": confusion.to_dict(orient="index"),
        "failures": failures,
    }


def _predict_regime(metrics: dict[str, Any]) -> str:
    parity_hat = str(metrics.get("parity_hat", "")).lower()
    knee_detected = bool(metrics.get("knee_detected", False))
    knee_conf = float(metrics.get("knee_confidence", 0.0) or 0.0)
    band_hat = str(metrics.get("band_class_hat", "")).lower()
    impedance_ratio = _to_float(metrics.get("impedance_ratio"))

    if parity_hat == "null" and knee_conf < 0.65:
        return "null"
    if knee_detected and band_hat == "forbidden_middle":
        return "forbidden_middle"
    if knee_detected and knee_conf >= 0.55 and band_hat in {"coherent", "forbidden_middle"}:
        return "hybrid"
    if not knee_detected and band_hat == "coherent":
        return "coherent"
    if impedance_ratio is not None and abs(impedance_ratio - 1.0) < 0.2 and band_hat == "incoherent":
        return "adversarial"
    return "adversarial"


def _invariant_vector(metrics: dict[str, Any]) -> list[float]:
    ci_lo = _to_float(metrics.get("knee_ci_lo"))
    ci_hi = _to_float(metrics.get("knee_ci_hi"))
    if ci_lo is None or ci_hi is None:
        ci_width = np.nan
    else:
        ci_width = float(max(ci_hi - ci_lo, 0.0))

    return [
        _to_float(metrics.get("knee_p")) or 0.0,
        _to_float(metrics.get("knee_confidence")) or 0.0,
        ci_width,
        _to_float(metrics.get("knee_strength")) or 0.0,
        _to_float(metrics.get("knee_delta_gamma")) or 0.0,
        _to_float(metrics.get("knee_delta_bic")) or 0.0,
        _to_float(metrics.get("knee_resid_improvement")) or 0.0,
        _to_float(metrics.get("knee_post_slope_std")) or 0.0,
        _to_float(metrics.get("knee_curvature_peak_ratio")) or 0.0,
        _to_float(metrics.get("knee_curvature_alignment")) or 0.0,
        _to_float(metrics.get("middle_score")) or 0.0,
        _to_float(metrics.get("P_lock")) or 0.0,
        _to_float(metrics.get("parity_p_perm")) or 0.0,
        _to_float(metrics.get("parity_p_dir")) or 0.0,
        _to_float(metrics.get("parity_obs_dir")) or 0.0,
        _to_float(metrics.get("S_at_mu_k")) or 0.0,
        _to_float(metrics.get("W_mu")) or 0.0,
        _to_float(metrics.get("band_hit_rate")) or 0.0,
        _to_float(metrics.get("R_align")) or 0.0,
        _to_float(metrics.get("M_Z")) or 0.0,
        _to_float(metrics.get("noise_kurtosis")) or 0.0,
        _to_float(metrics.get("noise_hill")) or 0.0,
        _to_float(metrics.get("gamma")) or 0.0,
        _to_float(metrics.get("drift")) or 0.0,
        _to_float(metrics.get("eta_mean")) or 0.0,
    ]


def _cross_validated_blind_predictions(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).astype(str)
    label_order = [str(v) for v in np.asarray(labels).tolist()]
    classes = np.array([lbl for lbl in label_order if np.any(y_arr == lbl)], dtype=str)
    if classes.size == 0:
        classes = np.unique(y_arr)
    fold_idx = np.zeros(y_arr.shape[0], dtype=int)
    for cls in classes:
        idx = np.where(y_arr == cls)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        for i, ridx in enumerate(shuffled):
            fold_idx[ridx] = i % max(n_splits, 2)

    pred = np.full(y_arr.shape[0], None, dtype=object)
    prob = np.zeros((y_arr.shape[0], classes.size), dtype=float)
    for fold in range(max(n_splits, 2)):
        test_mask = fold_idx == fold
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue
        y_train = y_arr[train_mask]
        if np.unique(y_train).size < 2:
            # Degenerate fold fallback.
            pred[test_mask] = y_train[0]
            prob[test_mask] = 0.0
            cidx = int(np.where(classes == y_train[0])[0][0])
            prob[test_mask, cidx] = 1.0
            continue

        model = InvariantBlindClassifier(
            max_iter=500,
            lr=0.07,
            reg=2e-3,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        model.fit(X_arr[train_mask], y_train)
        prob_fold = model.predict_proba(X_arr[test_mask])
        pred_fold = model.predict(X_arr[test_mask])
        pred[test_mask] = pred_fold
        # Align class axis.
        prob_aligned = np.zeros((prob_fold.shape[0], classes.size), dtype=float)
        for i, cls in enumerate(model.classes_):
            target_idx = int(np.where(classes == cls)[0][0])
            prob_aligned[:, target_idx] = prob_fold[:, i]
        prob[test_mask] = prob_aligned

    missing = np.asarray([v is None for v in pred], dtype=bool)
    if np.any(missing):
        # Final fallback model for any uncovered rows.
        model = InvariantBlindClassifier(max_iter=500, lr=0.07, reg=2e-3, random_state=int(rng.integers(0, 2**31 - 1)))
        model.fit(X_arr, y_arr)
        prob_fill = model.predict_proba(X_arr[missing])
        pred_fill = model.predict(X_arr[missing])
        pred[missing] = pred_fill
        prob_aligned = np.zeros((prob_fill.shape[0], classes.size), dtype=float)
        for i, cls in enumerate(model.classes_):
            target_idx = int(np.where(classes == cls)[0][0])
            prob_aligned[:, target_idx] = prob_fill[:, i]
        prob[missing] = prob_aligned

    return np.asarray(pred, dtype=str), prob, classes


def _build_invariant_stability_map(epsilon: float = 0.2) -> dict[str, Any]:
    gamma_values = np.linspace(1.0, 2.0, 11)
    xi_over_L_values = np.linspace(0.3, 2.0, 10)
    tau_s_values = np.geomspace(0.03, 0.4, 8)
    impedance_ratio_values = np.linspace(0.6, 1.4, 9)

    rows: list[dict[str, Any]] = []
    for gamma in gamma_values:
        delta_b = 2.0 - float(gamma)
        for xi_over_L in xi_over_L_values:
            for tau_s in tau_s_values:
                for imp_ratio in impedance_ratio_values:
                    mu = float(imp_ratio / max(xi_over_L, 1e-12))
                    s_score = float((2.0 ** gamma) / math.sqrt(1.0 + (mu * tau_s) ** 2))
                    log_s = float(math.log(max(s_score, 1e-12)))
                    if log_s > epsilon:
                        band = "coherent"
                    elif log_s < -epsilon:
                        band = "incoherent"
                    else:
                        band = "forbidden_middle"
                    rows.append(
                        {
                            "gamma": round(float(gamma), 6),
                            "Delta_b": round(delta_b, 6),
                            "xi_over_L": round(float(xi_over_L), 6),
                            "tau_s": round(float(tau_s), 8),
                            "impedance_ratio": round(float(imp_ratio), 6),
                            "S": round(s_score, 8),
                            "log_S": round(log_s, 8),
                            "band_class": band,
                        }
                    )

    map_df = pd.DataFrame(rows)
    class_rates = map_df["band_class"].value_counts(normalize=True).to_dict()
    by_gamma = (
        map_df.groupby("gamma")["log_S"]
        .agg(logS_mean="mean", logS_std="std")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "epsilon": float(epsilon),
        "n_points": int(map_df.shape[0]),
        "class_rates": {str(k): float(v) for k, v in class_rates.items()},
        "delta_b_variance": float(np.var(map_df["Delta_b"].to_numpy(dtype=float))),
        "logS_variance": float(np.var(map_df["log_S"].to_numpy(dtype=float))),
        "by_gamma": by_gamma,
        "grid": rows,
    }


def _run_single_config(
    config: dict[str, Any],
    run_seed: int,
    run_dir: Path,
    use_cli: bool,
    global_expect: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_dataset(config, seed=run_seed, run_index=0)
    write_generated_dataset(run_dir, generated)
    dataset_dir = run_dir / "dataset"
    _write_canonical_dataset(run_dir / "data.csv", run_dir / "meta.json", config, dataset_dir)
    gka_out = run_dir / "gka_output"
    run_result = _execute_pipeline(
        dataset_dir=dataset_dir,
        gka_out=gka_out,
        run_seed=run_seed,
        config=config,
        use_cli=use_cli,
    )
    metrics = _collect_run_metrics(
        run_result=run_result,
        run_dir=run_dir,
        test_name=str(config.get("name", "stress_case")),
        global_expect=global_expect,
    )
    (run_dir / "results.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def evaluate_robustness_gates(
    report: dict[str, Any],
    expectations_path: Path,
) -> dict[str, Any]:
    robust_cfg = _load_global_expect(expectations_path).get("robustness", {})
    if not isinstance(robust_cfg, dict) or not robust_cfg:
        return {"enabled": False, "passed": True, "checks": []}

    param = report.get("parameter_robustness", {})
    blind = report.get("blind_test", {})

    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "false_positive_rate",
            float(param.get("false_positive_rate", 1.0)),
            "<=",
            float(robust_cfg.get("false_positive_rate_max", 0.10)),
        )
    )
    checks.append(
        _check(
            "false_negative_rate",
            float(param.get("false_negative_rate", 1.0)),
            "<=",
            float(robust_cfg.get("false_negative_rate_max", 0.35)),
        )
    )
    checks.append(
        _check(
            "knee_localization_err_p90",
            float(param.get("knee_localization_err_p90", float("inf"))),
            "<=",
            float(robust_cfg.get("knee_localization_err_p90_max", 0.08)),
        )
    )

    acc = float(blind.get("accuracy", 0.0))
    auroc_val = blind.get("auroc")
    auroc = None if auroc_val is None else float(auroc_val)
    acc_min = float(robust_cfg.get("blind_accuracy_min", 0.80))
    auroc_min = float(robust_cfg.get("blind_auroc_min", 0.85))
    blind_pass = bool(acc >= acc_min or (auroc is not None and auroc >= auroc_min))
    checks.append(
        {
            "name": "blind_accuracy_or_auroc",
            "value": float(acc if auroc is None else max(acc, auroc)),
            "op": ">=",
            "threshold": float(min(acc_min, auroc_min)),
            "pass": blind_pass,
            "details": {"accuracy": acc, "accuracy_min": acc_min, "auroc": auroc, "auroc_min": auroc_min},
        }
    )

    passed = all(bool(c.get("pass", False)) for c in checks)
    return {"enabled": True, "passed": passed, "checks": checks}


def _load_global_expect(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return {}
    section = payload.get("global", {})
    return section if isinstance(section, dict) else {}


def _set_l_samples(cfg: dict[str, Any], n: int) -> None:
    axis = cfg.get("L_values")
    if isinstance(axis, dict):
        axis["n"] = int(n)
    elif isinstance(axis, list):
        if len(axis) >= 2:
            cfg["L_values"] = np.geomspace(float(min(axis)), float(max(axis)), int(n)).tolist()


def _set_noise_std(cfg: dict[str, Any], std: float) -> None:
    cfg.setdefault("noise", {})["std"] = float(max(0.0, std))


def _set_min_points(cfg: dict[str, Any], n_l: int) -> None:
    run_cfg = cfg.setdefault("run", {}).setdefault("gka_config", {})
    knee_cfg = run_cfg.setdefault("knee", {})
    scaling_cfg = run_cfg.setdefault("scaling", {})
    # Keep the sweep within executable bounds while still covering low sample counts.
    min_points = int(max(3, min(6, n_l - 1)))
    knee_cfg["min_points"] = min_points
    scaling_cfg["min_sizes"] = int(max(3, min(4, n_l - 1)))


def _set_gamma(cfg: dict[str, Any], gamma_val: float) -> None:
    cfg.setdefault("eta_model", {})["gamma"] = float(gamma_val)
    cfg.setdefault("truth", {})["gamma"] = float(gamma_val)


def _set_tau_and_impedance(cfg: dict[str, Any], xi_val: float, tau_s: float) -> None:
    c_val = float(max(xi_val, 1e-6) / (2.0 * math.pi * max(tau_s, 1e-9)))
    imp = cfg.setdefault("impedance", {})
    imp["enabled"] = True
    imp["c"] = c_val
    imp["jitter_std"] = 0.03
    cfg.setdefault("truth", {})["tau_s"] = float(tau_s)
    cfg["truth"]["c"] = c_val
    run_cfg = cfg.setdefault("run", {}).setdefault("gka_config", {})
    run_cfg.setdefault("impedance", {})
    run_cfg["impedance"]["enabled"] = True
    run_cfg["impedance"].setdefault("tolerance", 0.16)


def _build_stability_envelopes(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    numeric_cols = ("noise_amp", "n_L", "gamma_true", "tau_s_true")
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        q = np.quantile(vals, [0.1, 0.5, 0.9])
        out[col] = {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}

    gamma_err = (pd.to_numeric(df["gamma_hat"], errors="coerce") - pd.to_numeric(df["gamma_true"], errors="coerce")).abs()
    out["gamma_abs_err"] = {
        "p50": float(np.nanquantile(gamma_err.to_numpy(dtype=float), 0.5)),
        "p90": float(np.nanquantile(gamma_err.to_numpy(dtype=float), 0.9)),
    }
    knee_err = pd.to_numeric(df["knee_err_frac"], errors="coerce")
    if knee_err.notna().any():
        out["knee_err_frac"] = {
            "p50": float(np.nanquantile(knee_err.to_numpy(dtype=float), 0.5)),
            "p90": float(np.nanquantile(knee_err.to_numpy(dtype=float), 0.9)),
        }
    return out


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _check(name: str, value: float, op: str, threshold: float) -> dict[str, Any]:
    if op == "<=":
        is_pass = bool(value <= threshold)
    elif op == ">=":
        is_pass = bool(value >= threshold)
    else:
        raise ValueError(f"Unsupported op {op}")
    return {
        "name": name,
        "value": float(value),
        "op": op,
        "threshold": float(threshold),
        "pass": is_pass,
    }


if __name__ == "__main__":
    raise SystemExit(main())
