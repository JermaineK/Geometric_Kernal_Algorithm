"""Domain-agnostic orchestration for the GKA pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gka.core.config import dump_yaml, resolve_config
from gka.core.registry import get_adapter
from gka.core.types import DatasetBundle, PipelineResult
from gka.core.versioning import git_commit_hash, invocation_string, package_versions
from gka.data.io import load_dataset_spec, write_json
from gka.metrics.eigen_stability import estimate_eigen_stability
from gka.data.validators import report_to_dict, validate_dataset
from gka.metrics.forbidden_middle import compute_forbidden_middle_score
from gka.ops.coherence import compute_coherence_metrics
from gka.ops.diagnostics import (
    alignment_residual,
    band_hit_rate,
    compute_stability_diagnostics,
    predict_bands,
)
from gka.ops.geometry import compute_geometry
from gka.ops.impedance import impedance_alignment
from gka.ops.knee import detect_knee
from gka.ops.scaling import fit_scaling
from gka.ops.spectrum import extract_ticks
from gka.ops.stability import classify_stability
from gka.stats.null_models import apply_null_model, parity_significance_pvalues
from gka.utils.hash import dataset_hash, mapping_sha256
from gka.utils.safe_math import safe_log
from gka.utils.time import utc_now_iso


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete."""


def run_pipeline(
    dataset_path: str,
    domain: str | None,
    out_dir: str,
    config_path: str | None,
    null_n: int | None,
    allow_missing: bool,
    seed: int | None,
    dump_intermediates: bool = False,
    argv: list[str] | None = None,
) -> PipelineResult:
    dataset_root = Path(dataset_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset(dataset_root, allow_missing=allow_missing)
    if not validation.valid:
        failures = [f"[{i.code}] {i.message}" for i in validation.issues if i.level == "error"]
        raise PipelineError(
            "Dataset validation failed. Run 'gka validate <dataset>' for details. "
            + " | ".join(failures)
        )

    dataset_spec = load_dataset_spec(dataset_root)
    cfg_overrides: dict[str, Any] = {}
    if null_n is not None:
        cfg_overrides["nulls"] = {"enabled": null_n > 0, "n": int(null_n)}
    if seed is not None:
        cfg_overrides["seed"] = int(seed)

    resolved = resolve_config(dataset_spec=dataset_spec, config_path=config_path, overrides=cfg_overrides)
    dump_yaml(resolved, out_root / "config_resolved.yaml")

    rng = np.random.default_rng(int(resolved.get("seed", 42)))
    domain_name = (domain or dataset_spec.get("domain", "")).strip().lower()
    if not domain_name:
        raise PipelineError("Domain was not provided and dataset.yaml is missing domain")

    adapter = get_adapter(domain_name)
    bundle = adapter.load(str(dataset_root))

    results_df, stage_context, intermediates = _run_core(bundle, adapter=adapter, cfg=resolved, rng=rng)
    results_path = out_root / "results.parquet"
    results_df.to_parquet(results_path, index=False)

    if dump_intermediates:
        _write_intermediates(out_root / "intermediates", intermediates)

    null_summary_df = None
    if resolved.get("nulls", {}).get("enabled", False):
        n_rep = int(resolved.get("nulls", {}).get("n", 0))
        models = list(resolved.get("nulls", {}).get("models", []))
        if n_rep > 0 and models:
            null_summary_df = _run_nulls(
                bundle=bundle,
                adapter=adapter,
                cfg=resolved,
                n_rep=n_rep,
                models=models,
                rng=rng,
            )
            null_dir = out_root / "nulls"
            null_dir.mkdir(exist_ok=True)
            null_summary_df.to_parquet(null_dir / "null_distributions.parquet", index=False)

    metadata = {
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit_hash(Path.cwd()),
        "package_versions": package_versions(),
        "dataset_hash": dataset_hash(dataset_root),
        "config_hash": mapping_sha256(resolved),
        "random_seed": int(resolved.get("seed", 42)),
        "cli_invocation": invocation_string(argv or []),
        "dataset_path": str(dataset_root.resolve()),
        "results_path": str(results_path.resolve()),
        "domain": domain_name,
        "dataset_id": dataset_spec.get("id"),
        "null_n": int(resolved.get("nulls", {}).get("n", 0))
        if resolved.get("nulls", {}).get("enabled", False)
        else 0,
        "validation": report_to_dict(validation),
        "stage_context": stage_context,
        "dump_intermediates": bool(dump_intermediates),
    }
    write_json(metadata, out_root / "run_metadata.json")

    return PipelineResult(summary=results_df, null_summary=null_summary_df, metadata=metadata)


def _run_core(
    bundle: DatasetBundle,
    adapter,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    obs = adapter.observable(bundle)
    X = np.asarray(obs.X, dtype=float)
    pair_df = _build_pair_table(bundle, X)

    geometry = compute_geometry(X)
    ticks = extract_ticks(X, method=cfg["ticks"]["method"])
    per_size = _aggregate_eta_by_size(pair_df)
    knee = detect_knee(
        eta_series=per_size["eta"].to_numpy(),
        L_series=per_size["L"].to_numpy(),
        method=cfg["knee"]["method"],
        rho=float(cfg["knee"]["rho"]),
        min_points=int(cfg["knee"].get("min_points", 6)),
        bic_delta_min=float(cfg["knee"].get("bic_delta_min", 10.0)),
        knee_p_min=float(cfg["knee"].get("knee_p_min", 0.35)),
        knee_strength_min=float(cfg["knee"].get("knee_strength_min", 0.5)),
        max_candidates=int(cfg["knee"].get("max_candidates", 6)),
        bootstrap_n=int(cfg["knee"].get("bootstrap_n", 200)),
        bootstrap_cov_max=float(cfg["knee"].get("bootstrap_cov_max", 0.25)),
        edge_buffer_frac=float(cfg["knee"].get("edge_buffer_frac", 0.15)),
        min_points_each_side=int(cfg["knee"].get("min_points_each_side", 3)),
        min_frac_each_side=float(cfg["knee"].get("min_frac_each_side", 0.2)),
        slope_abs_max=float(cfg["knee"].get("slope_abs_max", 6.0)),
        post_slope_max=float(cfg["knee"].get("post_slope_max", 6.0)),
        resid_improvement_min=float(cfg["knee"].get("resid_improvement_min", 0.0)),
        curvature_peak_ratio_min=float(cfg["knee"].get("curvature_peak_ratio_min", 0.0)),
        curvature_alignment_min=float(cfg["knee"].get("curvature_alignment_min", 0.0)),
        instability_window=int(cfg["knee"].get("instability_window", 7)),
        instability_zscore=float(cfg["knee"].get("instability_zscore", 2.0)),
        middle_score_threshold=float(cfg["knee"].get("middle_score_threshold", 0.6)),
        rng=rng,
    )

    exclude_band: tuple[float, float] | None = None
    if cfg["scaling"].get("exclude_forbidden", True):
        exclude_band = _coerce_band(knee.forbidden_band)

    scaling = fit_scaling(
        eta=per_size["eta"].to_numpy(),
        L=per_size["L"].to_numpy(),
        exclude_band=exclude_band,
        weights=per_size["n"].to_numpy(dtype=float),
        method=cfg["scaling"]["method"],
        robust=bool(cfg["scaling"].get("robust", True)),
        huber_delta=float(cfg["scaling"].get("huber_delta", 1.5)),
        min_sizes=int(cfg["scaling"].get("min_sizes", 4)),
        bootstrap_n=int(cfg["scaling"].get("bootstrap_n", 1000)),
        rng=rng,
    )
    eigen = estimate_eigen_stability(
        gamma=scaling.gamma,
        b=float(cfg["stability"]["b"]),
        margin_eps=float(cfg["stability"].get("marginal_eps", 0.1)),
        gamma_ci=scaling.ci,
        evidence="size_law_gamma",
    )

    stability = classify_stability(
        gamma=scaling.gamma,
        drift=scaling.drift,
        b=float(cfg["stability"]["b"]),
        gamma_min=float(cfg["stability"].get("gamma_min", 0.0)),
        marginal_eps=float(cfg["stability"].get("marginal_eps", 0.1)),
    )

    imp_in = adapter.impedance_proxy(bundle)
    l_for_imp = knee.L_k if knee.L_k is not None else float(np.nanmedian(per_size["L"].to_numpy()))
    omega_k = _median_or_none(imp_in.omega_k)
    if omega_k is None and ticks.Omega_candidates.size > 0:
        omega_k = float(ticks.Omega_candidates[0])
    imp = impedance_alignment(
        omega_k=omega_k,
        L=l_for_imp,
        cm_or_v=imp_in.cm_or_v,
        a=imp_in.a_or_L,
        tolerance=float(cfg["impedance"].get("tolerance", 0.1)),
    )
    c_m_hat = _median_or_none(imp_in.cm_or_v)
    if c_m_hat is None and omega_k is not None and l_for_imp is not None and l_for_imp > 0:
        c_m_hat = float((omega_k * l_for_imp) / (2.0 * np.pi))
    tau_s_hat = None if omega_k is None or omega_k <= 0 else float(1.0 / omega_k)

    stability_cfg = cfg.get("stability_score", {})
    mu_grid = _positive_finite(np.concatenate([ticks.Omega_candidates, _array_or_empty(imp_in.omega_k)]))
    sdiag = compute_stability_diagnostics(
        gamma=scaling.gamma,
        b=float(cfg["stability"]["b"]),
        mu_k_hat=omega_k,
        tau_s_hat=tau_s_hat,
        mu_grid=mu_grid if mu_grid.size > 0 else None,
        eps_log=float(stability_cfg.get("eps_log", 0.15)),
        L_values=per_size["L"].to_numpy(dtype=float),
        c_m_hat=c_m_hat,
    )

    middle = compute_forbidden_middle_score(
        knee_probability=float(knee.knee_p),
        knee_ci=knee.knee_ci,
        L_bounds=(
            float(np.min(per_size["L"].to_numpy(dtype=float))),
            float(np.max(per_size["L"].to_numpy(dtype=float))),
        ),
        knee_strength=float(knee.knee_strength),
        delta_gamma=knee.delta_gamma,
        slope_drift=float(scaling.drift),
        ridge_strength=float(ticks.ridge_strength),
        bootstrap_cov=knee.bootstrap_cov,
        threshold=float(cfg["knee"].get("middle_score_threshold", 0.6)),
    )

    bands_cfg = cfg.get("bands", {})
    l_ref = float(l_for_imp)
    if c_m_hat is not None and l_ref > 0:
        predicted = predict_bands(
            L=l_ref,
            c_m=float(c_m_hat),
            n_max=int(bands_cfg.get("n_max", 3)),
        )
        hit_rate = band_hit_rate(
            observed_peaks=ticks.Omega_candidates,
            predicted_peaks=predicted,
            rel_tol=float(bands_cfg.get("rel_tol", 0.15)),
        )
    else:
        predicted = np.array([], dtype=float)
        hit_rate = 0.0
    r_align, m_z = alignment_residual(
        omega_k=omega_k,
        L=l_ref,
        c_m=c_m_hat,
    )

    post_mask = np.ones(pair_df.shape[0], dtype=bool)
    band = _coerce_band(knee.forbidden_band)
    if band is not None:
        _, band_hi = band
        post_mask = pair_df["L"].to_numpy(dtype=float) > band_hi
        if not np.any(post_mask):
            post_mask = np.ones(pair_df.shape[0], dtype=bool)

    signed_for_lock = pair_df.loc[post_mask, "signed_parity"].to_numpy(dtype=float)
    segment_labels = _segment_labels(pair_df.loc[post_mask, "L"].to_numpy(dtype=float))
    coherence = compute_coherence_metrics(
        pair_df["X_plus"].to_numpy(dtype=float),
        pair_df["X_minus"].to_numpy(dtype=float),
        signed_parity=signed_for_lock,
        segment_labels=segment_labels,
    )

    parity_sig_cfg = cfg.get("parity_significance", {})
    if parity_sig_cfg.get("enabled", True):
        parity_sig = parity_significance_pvalues(
            pair_df=pair_df[["L", "O_L", "O_R"]].copy(),
            n_perm=int(parity_sig_cfg.get("n_perm", 200)),
            rng=rng,
            alpha=float(parity_sig_cfg.get("alpha", 0.05)),
            eta_min=float(parity_sig_cfg.get("eta_min", 0.03)),
            direction_min=float(parity_sig_cfg.get("direction_min", 0.30)),
        )
    else:
        signed = pair_df["signed_parity"].to_numpy(dtype=float)
        obs_dir = float(np.abs(np.mean(np.sign(signed)))) if signed.size else 0.0
        parity_sig = {
            "mirror_stat": float(np.median(pair_df["eta"])),
            "obs_dir": obs_dir,
            "p_perm": 0.0,
            "p_dir": 0.0,
            "mirror_pass": True,
            "direction_pass": True,
            "signal_pass": True,
        }

    knee_confidence = float(knee.confidence if parity_sig["signal_pass"] else 0.0)
    knee_detected = bool(knee.has_knee and knee_confidence > 0 and middle.label != "forbidden_middle")
    band_class_hat = "forbidden_middle" if middle.label == "forbidden_middle" else sdiag.band_class_hat
    forbidden_lo, forbidden_hi = _band_values(knee.forbidden_band)

    out = per_size.copy()
    out["E_plus"] = float(np.mean(np.square(pair_df["X_plus"].to_numpy(dtype=float))))
    out["E_minus"] = float(np.mean(np.square(pair_df["X_minus"].to_numpy(dtype=float))))
    out["L_k"] = _nan_or_value(knee.L_k)
    out["knee_lo"] = _nan_or_value(knee.knee_window[0])
    out["knee_hi"] = _nan_or_value(knee.knee_window[1])
    out["forbidden_lo"] = forbidden_lo
    out["forbidden_hi"] = forbidden_hi
    out["knee_confidence"] = knee_confidence
    out["knee_detected"] = knee_detected
    out["knee_p"] = knee.knee_p
    out["knee_ci_lo"] = _nan_or_value(knee.knee_ci[0])
    out["knee_ci_hi"] = _nan_or_value(knee.knee_ci[1])
    out["knee_strength"] = knee.knee_strength
    out["knee_delta_gamma"] = _nan_or_value(knee.delta_gamma)
    out["knee_resid_improvement"] = _nan_or_value(knee.resid_improvement)
    out["knee_post_slope_std"] = _nan_or_value(knee.post_slope_std)
    out["knee_curvature_peak_ratio"] = _nan_or_value(knee.curvature_peak_ratio)
    out["knee_curvature_alignment"] = _nan_or_value(knee.curvature_alignment)
    out["knee_candidate_count_proposed"] = int(knee.candidate_count_proposed)
    out["knee_candidate_count_evaluated"] = int(knee.candidate_count_evaluated)
    out["knee_candidate_count_sanity_pass"] = int(knee.candidate_count_sanity_pass)
    out["middle_score"] = _nan_or_value(middle.score)
    out["middle_label"] = middle.label
    out["forbidden_middle_width"] = _nan_or_value(middle.width)
    out["forbidden_middle_center"] = _nan_or_value(middle.center)
    out["forbidden_middle_reason_codes"] = ";".join(middle.reason_codes)
    out["knee_delta_bic"] = knee.delta_bic
    out["knee_bic_no_knee"] = knee.bic_no_knee
    out["knee_bic_knee"] = knee.bic_knee
    out["knee_bootstrap_cov"] = _nan_or_value(knee.bootstrap_cov)
    out["knee_bootstrap_count"] = knee.bootstrap_count
    out["forbidden_detected"] = knee.forbidden_detected
    out["knee_slope_left"] = _nan_or_value(knee.slope_left)
    out["knee_slope_right"] = _nan_or_value(knee.slope_right)
    out["knee_rejected_because"] = ";".join(knee.rejection_reasons)
    out["gamma"] = scaling.gamma
    out["Delta_hat"] = scaling.Delta_hat
    out["gamma_ci_lo"] = scaling.ci[0]
    out["gamma_ci_hi"] = scaling.ci[1]
    out["drift"] = scaling.drift
    out["stability_class"] = stability.stability_class
    out["lambda_eff"] = stability.lambda_eff
    out["ineq_pass"] = stability.ineq_pass
    out["eigen_band"] = eigen.eigen_band
    out["stability_margin"] = eigen.stability_margin
    out["eigen_confidence"] = eigen.confidence
    out["eigen_evidence"] = eigen.eigen_evidence
    out["impedance_ratio"] = imp.ratio
    out["impedance_pass"] = imp.passed
    out["A"] = coherence.A
    out["F"] = coherence.F
    out["P_lock"] = coherence.P_lock
    out["parity_mirror_stat"] = float(parity_sig["mirror_stat"])
    out["parity_p_perm"] = float(parity_sig["p_perm"])
    out["parity_p_dir"] = float(parity_sig["p_dir"])
    out["parity_obs_dir"] = float(parity_sig["obs_dir"])
    out["parity_signal_pass"] = bool(parity_sig["signal_pass"])
    out["R_Omega"] = ticks.R_Omega
    out["ridge_strength"] = ticks.ridge_strength
    out["omega_k_hat"] = _nan_or_value(omega_k)
    out["tau_s_hat"] = _nan_or_value(sdiag.tau_s_hat)
    out["S_at_mu_k"] = _nan_or_value(sdiag.S_at_mu_k)
    out["S_margin_log"] = _nan_or_value(sdiag.S_margin_log)
    out["W_mu"] = _nan_or_value(sdiag.W_mu)
    out["W_L"] = _nan_or_value(sdiag.W_L)
    out["band_class_hat"] = band_class_hat
    out["c_m_hat"] = _nan_or_value(c_m_hat)
    out["R_align"] = _nan_or_value(r_align)
    out["M_Z"] = _nan_or_value(m_z)
    out["band_hit_rate"] = float(hit_rate)
    out["geometry_kappa_mean"] = float(np.mean(geometry.kappa))
    out["geometry_tau_mean"] = float(np.mean(geometry.tau))

    stage_context = {
        "tick_candidates": ticks.Omega_candidates.tolist(),
        "omega_band": ticks.omega_band,
        "scaling_points": scaling.n_points,
        "knee_detected": knee_detected,
        "knee_delta_bic": knee.delta_bic,
        "knee_p": knee.knee_p,
        "knee_strength": knee.knee_strength,
        "knee_post_slope_std": knee.post_slope_std,
        "knee_curvature_peak_ratio": knee.curvature_peak_ratio,
        "knee_curvature_alignment": knee.curvature_alignment,
        "knee_candidate_count_proposed": int(knee.candidate_count_proposed),
        "knee_candidate_count_evaluated": int(knee.candidate_count_evaluated),
        "knee_candidate_count_sanity_pass": int(knee.candidate_count_sanity_pass),
        "middle_score": middle.score,
        "middle_label": middle.label,
        "forbidden_middle_width": middle.width,
        "forbidden_middle_center": middle.center,
        "forbidden_middle_reason_codes": middle.reason_codes,
        "parity_p_perm": float(parity_sig["p_perm"]),
        "parity_p_dir": float(parity_sig["p_dir"]),
        "parity_obs_dir": float(parity_sig["obs_dir"]),
        "parity_signal_pass": bool(parity_sig["signal_pass"]),
        "knee_rejection_reasons": knee.rejection_reasons,
        "S_curve_mu": sdiag.S_curve_mu.tolist(),
        "S_curve_values": sdiag.S_curve_values.tolist(),
        "predicted_bands": predicted.tolist(),
        "band_hit_rate": float(hit_rate),
        "eigen_band": eigen.eigen_band,
        "stability_margin": eigen.stability_margin,
    }
    scaling_points = _build_scaling_points(
        per_size=per_size,
        exclude_band=exclude_band,
    )
    stability_curve = pd.DataFrame(
        {
            "mu": sdiag.S_curve_mu,
            "S": sdiag.S_curve_values,
        }
    )
    intermediates = {
        "windowed_series": per_size,
        "scaling_points": scaling_points,
        "stability_curve": stability_curve,
        "knee_candidates": {
            "L_k": knee.L_k,
            "knee_window": list(knee.knee_window),
            "forbidden_band": list(knee.forbidden_band),
            "confidence": knee.confidence,
            "delta_bic": knee.delta_bic,
            "bic_no_knee": knee.bic_no_knee,
            "bic_knee": knee.bic_knee,
            "bootstrap_cov": knee.bootstrap_cov,
            "bootstrap_count": knee.bootstrap_count,
            "knee_p": knee.knee_p,
            "knee_ci": list(knee.knee_ci),
            "knee_strength": knee.knee_strength,
            "delta_gamma": knee.delta_gamma,
            "resid_improvement": knee.resid_improvement,
            "rejection_reasons": knee.rejection_reasons,
            "candidate_count_proposed": int(knee.candidate_count_proposed),
            "candidate_count_evaluated": int(knee.candidate_count_evaluated),
            "candidate_count_sanity_pass": int(knee.candidate_count_sanity_pass),
        },
        "spectrum_candidates": {
            "Omega_candidates": ticks.Omega_candidates.tolist(),
            "omega_band": ticks.omega_band,
            "R_Omega": ticks.R_Omega,
            "ridge_strength": ticks.ridge_strength,
        },
        "parity_diagnostics": {
            "mirror_stat": float(parity_sig["mirror_stat"]),
            "p_perm": float(parity_sig["p_perm"]),
            "p_dir": float(parity_sig["p_dir"]),
            "signal_pass": bool(parity_sig["signal_pass"]),
            "P_lock": float(coherence.P_lock),
        },
        "forbidden_middle": {
            "score": middle.score,
            "label": middle.label,
            "components": middle.components,
            "reason_codes": middle.reason_codes,
            "width": middle.width,
            "center": middle.center,
        },
    }
    return out, stage_context, intermediates


def _aggregate_eta_by_size(pair_df: pd.DataFrame) -> pd.DataFrame:
    df = pair_df[["L", "eta", "signed_parity"]].copy()
    grouped = (
        df.groupby("L", as_index=False)
        .agg(
            eta=("eta", "median"),
            eta_mean=("eta", "mean"),
            eta_std=("eta", "std"),
            signed_mean=("signed_parity", "mean"),
            n=("eta", "size"),
        )
        .sort_values("L")
    )
    grouped["eta_std"] = grouped["eta_std"].fillna(0.0)
    return grouped.reset_index(drop=True)


def _build_pair_table(bundle: DatasetBundle, observable_values: np.ndarray) -> pd.DataFrame:
    spec = bundle.dataset_spec
    cols = spec.get("columns", {})
    group_col = cols.get("group", "case_id")
    time_col = cols.get("time", "t")
    size_col = cols.get("size", "L")
    hand_col = cols.get("handedness", "hand")
    observable_candidates = cols.get("observable", ["O"])

    samples = bundle.samples.reset_index(drop=True).copy()
    value_col = _resolve_observable_column(samples, observable_candidates, observable_values)
    key_cols = [group_col, time_col, size_col]
    if "omega" in samples.columns:
        key_cols.append("omega")

    required_cols = key_cols + [hand_col, value_col]
    missing = [c for c in required_cols if c not in samples.columns]
    if missing:
        raise PipelineError(f"Cannot build parity pairs: missing columns {missing}")

    dup_counts = samples.groupby(key_cols + [hand_col], dropna=False).size()
    if (dup_counts > 1).any():
        raise PipelineError(
            "Found duplicate rows for identical pairing keys. Expected one row per hand for each "
            f"{key_cols}. Please deduplicate before running."
        )

    pivot = (
        samples[key_cols + [hand_col, value_col]]
        .pivot_table(index=key_cols, columns=hand_col, values=value_col, aggfunc="first")
        .rename(columns=lambda h: f"O_{h}")
        .reset_index()
    )
    if "O_L" not in pivot.columns or "O_R" not in pivot.columns:
        raise PipelineError(
            "Pair table construction failed: each pair key must contain hand='L' and hand='R'. "
            "Run `gka validate` to inspect pairing completeness."
        )

    out = pd.DataFrame()
    out["L"] = pivot[size_col].astype(float)
    out["omega"] = pivot["omega"].astype(float) if "omega" in pivot.columns else 0.0
    out["O_L"] = pivot["O_L"].astype(float)
    out["O_R"] = pivot["O_R"].astype(float)
    denom = np.maximum((out["O_L"] + out["O_R"]) / 2.0, 1e-12)
    out["eta"] = np.abs(out["O_L"] - out["O_R"]) / denom
    out["signed_parity"] = (out["O_L"] - out["O_R"]) / denom
    out["X_plus"] = (out["O_L"] + out["O_R"]) / 2.0
    out["X_minus"] = (out["O_L"] - out["O_R"]) / 2.0
    return out.sort_values(["L", "omega"]).reset_index(drop=True)


def _resolve_observable_column(
    samples: pd.DataFrame,
    observable_candidates: list[str],
    observable_values: np.ndarray,
) -> str:
    for col in observable_candidates:
        if col in samples.columns:
            return col
    fallback = "__obs__"
    if fallback in samples.columns:
        return fallback
    if observable_values.size != samples.shape[0]:
        raise PipelineError(
            "Adapter observable length does not match samples rows; cannot derive fallback observable column"
        )
    samples[fallback] = observable_values
    return fallback


def _coerce_band(band: tuple[float | None, float | None]) -> tuple[float, float] | None:
    lo, hi = band
    if lo is None or hi is None:
        return None
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        return None
    return lo_f, hi_f


def _band_values(band: tuple[float | None, float | None]) -> tuple[float, float]:
    b = _coerce_band(band)
    if b is None:
        return float("nan"), float("nan")
    return b


def _nan_or_value(v: float | None) -> float:
    if v is None:
        return float("nan")
    return float(v)


def _segment_labels(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=int)
    if np.unique(values).size < 3:
        return np.zeros(values.size, dtype=int)
    qs = np.quantile(values, [0.33, 0.66])
    return np.digitize(values, bins=qs, right=False)


def _run_nulls(
    bundle: DatasetBundle,
    adapter,
    cfg: dict[str, Any],
    n_rep: int,
    models: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    spec = bundle.dataset_spec
    cols = spec.get("columns", {})
    gcol = cols.get("group", "case_id")
    hcol = cols.get("handedness", "hand")

    base_samples = bundle.samples.copy()
    obs = adapter.observable(bundle)
    value_col = cols.get("observable", ["O"])[0]
    if value_col not in base_samples.columns:
        value_col = "__O__"
        base_samples[value_col] = np.asarray(obs.X, dtype=float)

    records: list[dict[str, Any]] = []
    for rep in range(n_rep):
        model = models[rep % len(models)]
        null_samples = apply_null_model(
            model,
            df=base_samples,
            value_col=value_col,
            group_col=gcol,
            hand_col=hcol,
            rng=rng,
        )
        null_spec = deepcopy(spec)
        null_spec.setdefault("columns", {})
        null_spec["columns"]["observable"] = [value_col]

        null_bundle = DatasetBundle(
            dataset_path=bundle.dataset_path,
            samples=null_samples,
            dataset_spec=null_spec,
            arrays=bundle.arrays,
            metadata={"null_model": model, "rep": rep},
        )

        try:
            result_df, _, _ = _run_core(null_bundle, adapter=adapter, cfg=cfg, rng=rng)
            gamma = float(result_df["gamma"].iloc[0])
            L_k = float(result_df["L_k"].iloc[0])
            eta_mean = float(result_df["eta"].mean())
        except Exception:
            gamma = float("nan")
            L_k = float("nan")
            eta_mean = float("nan")

        records.append({"rep": rep, "model": model, "gamma": gamma, "L_k": L_k, "eta_mean": eta_mean})

    return pd.DataFrame(records)


def _median_or_none(value: np.ndarray | float | None) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(np.nanmedian(arr))


def _array_or_empty(value: np.ndarray | float | None) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _positive_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return np.unique(arr[np.isfinite(arr) & (arr > 0)])


def _build_scaling_points(
    per_size: pd.DataFrame,
    exclude_band: tuple[float, float] | None,
) -> pd.DataFrame:
    df = per_size.copy()
    include = np.ones(df.shape[0], dtype=bool)
    if exclude_band is not None:
        lo, hi = exclude_band
        include &= ~((df["L"].to_numpy(dtype=float) >= lo) & (df["L"].to_numpy(dtype=float) <= hi))
    df["included"] = include
    df["log_L"] = safe_log(df["L"].to_numpy(dtype=float))
    df["log_eta"] = safe_log(np.abs(df["eta"].to_numpy(dtype=float)) + 1e-12)
    return df


def _write_intermediates(out_dir: Path, intermediates: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in intermediates.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(out_dir / f"{key}.parquet", index=False)
        elif isinstance(value, dict):
            write_json(value, out_dir / f"{key}.json")
        elif isinstance(value, (list, tuple)):
            write_json({"values": list(value)}, out_dir / f"{key}.json")
