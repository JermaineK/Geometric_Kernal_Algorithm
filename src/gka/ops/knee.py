"""Knee detection over eta(L) with posterior and two-stage scoring."""

from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes

from gka.core.types import KneeOutputs
from gka.knee.posterior import estimate_knee_posterior
from gka.utils.safe_math import safe_log


def detect_knee(
    eta_series: np.ndarray,
    L_series: np.ndarray,
    method: str,
    rho: float = 1.5,
    min_points: int = 6,
    bic_delta_min: float = 10.0,
    knee_p_min: float = 0.35,
    knee_strength_min: float = 0.5,
    max_candidates: int = 6,
    bootstrap_n: int = 200,
    bootstrap_cov_max: float = 0.25,
    edge_buffer_frac: float = 0.15,
    min_points_each_side: int = 3,
    min_frac_each_side: float = 0.2,
    slope_abs_max: float = 6.0,
    post_slope_max: float = 6.0,
    resid_improvement_min: float = 0.0,
    curvature_peak_ratio_min: float = 0.0,
    curvature_alignment_min: float = 0.0,
    instability_window: int = 7,
    instability_zscore: float = 2.0,
    middle_score_threshold: float = 0.6,
    rng: np.random.Generator | None = None,
) -> KneeOutputs:
    eta = np.asarray(eta_series, dtype=float)
    L = np.asarray(L_series, dtype=float)
    if eta.size != L.size:
        raise ValueError("eta_series and L_series must have equal length")
    if eta.size < min_points:
        raise ValueError(f"Knee detection requires at least {min_points} points")
    if method not in {"segmented", "log_curvature", "cpt"}:
        raise ValueError(f"Unsupported knee detection method '{method}'. Use segmented|log_curvature|cpt")

    order = np.argsort(L)
    Ls = L[order]
    ys = np.maximum(np.abs(eta[order]), 1e-12)
    x = safe_log(Ls)
    y = safe_log(ys)
    n = x.size
    rng = rng or np.random.default_rng(0)

    rss0 = _linear_rss(x, y)
    bic0 = _bic(rss0, n=n, k=2)
    forbidden = _detect_forbidden_band(
        Ls,
        ys,
        window=instability_window,
        zscore=instability_zscore,
    )

    posterior = estimate_knee_posterior(
        L=Ls,
        x=x,
        y=y,
        method=method,
        bic_delta_min=bic_delta_min,
        knee_strength_min=knee_strength_min,
        min_points_each_side=min_points_each_side,
        min_frac_each_side=min_frac_each_side,
        slope_abs_max=slope_abs_max,
        edge_buffer_frac=edge_buffer_frac,
        post_slope_max=post_slope_max,
        resid_improvement_min=resid_improvement_min,
        curvature_peak_ratio_min=curvature_peak_ratio_min,
        curvature_alignment_min=curvature_alignment_min,
        max_candidates=max_candidates,
        n_boot=bootstrap_n,
        rng=rng,
    )

    candidate = posterior.best_candidate
    knee_p = float(np.clip(posterior.knee_probability, 0.0, 1.0))
    knee_ci = posterior.knee_ci
    knee_samples = np.asarray(posterior.knee_samples, dtype=float)
    post_slope_samples = np.asarray(posterior.post_slope_samples, dtype=float)
    knee_cov = _cv(knee_samples)
    post_slope_std = _std(post_slope_samples)

    if candidate is None:
        return _no_knee_output(
            bic0=bic0,
            forbidden=forbidden,
            rejection_reasons=["no_candidate"],
            knee_p=knee_p,
            knee_ci=knee_ci,
            knee_cov=knee_cov,
            post_slope_std=post_slope_std,
        )

    reasons: list[str] = list(candidate.reasons)
    if knee_p < knee_p_min:
        reasons.append("knee_p_low")
    if candidate.delta_bic < bic_delta_min:
        reasons.append("bic_weak")
    if candidate.strength < knee_strength_min:
        reasons.append("knee_strength_weak")
    if knee_cov is not None and knee_cov > bootstrap_cov_max and knee_p < 0.8:
        reasons.append("bootstrap_unstable")

    if knee_samples.size > 0:
        l_k = float(np.median(knee_samples))
    else:
        l_k = float(candidate.L_k)
    if knee_ci[0] is None or knee_ci[1] is None:
        knee_window = (l_k, l_k)
    else:
        knee_window = (float(knee_ci[0]), float(knee_ci[1]))

    if forbidden is None:
        forbidden_band = (float(l_k / rho), float(l_k * rho))
        forbidden_detected = False
    else:
        forbidden_band = (float(forbidden[0]), float(forbidden[1]))
        forbidden_detected = True

    middle_score = _middle_score_from_posterior(
        knee_p=knee_p,
        knee_ci=knee_window,
        L_bounds=(float(np.min(Ls)), float(np.max(Ls))),
        knee_strength=float(candidate.strength),
        delta_gamma=float(candidate.delta_gamma),
    )
    middle_label = "forbidden_middle" if middle_score >= middle_score_threshold else "resolved"

    has_knee = len(reasons) == 0 and middle_label != "forbidden_middle"
    bic_strength = float(np.clip((candidate.delta_bic - bic_delta_min) / max(5.0, bic_delta_min), 0.0, 1.0))
    strength_norm = float(1.0 / (1.0 + np.exp(-(candidate.strength - knee_strength_min))))
    cov_term = 1.0 if knee_cov is None else float(np.clip(1.0 - knee_cov / max(bootstrap_cov_max, 1e-12), 0.0, 1.0))
    confidence = float(np.clip(0.40 * knee_p + 0.25 * bic_strength + 0.25 * strength_norm + 0.10 * cov_term, 0.0, 1.0))
    if not has_knee:
        confidence = float(confidence * 0.5)

    return KneeOutputs(
        L_k=l_k if has_knee else None,
        knee_window=knee_window if has_knee else (None, None),
        forbidden_band=forbidden_band,
        confidence=confidence if has_knee else 0.0,
        has_knee=has_knee,
        delta_bic=float(candidate.delta_bic),
        bic_no_knee=float(bic0),
        bic_knee=float(candidate.bic_knee),
        bootstrap_cov=knee_cov,
        bootstrap_count=int(knee_samples.size),
        forbidden_detected=forbidden_detected,
        slope_left=float(candidate.slope_left),
        slope_right=float(candidate.slope_right),
        rejection_reasons=[] if has_knee else reasons,
        knee_p=knee_p,
        knee_ci=knee_window,
        knee_strength=float(candidate.strength),
        delta_gamma=float(candidate.delta_gamma),
        resid_improvement=float(candidate.resid_improvement),
        middle_score=float(middle_score),
        middle_label=middle_label,
        post_slope_std=post_slope_std,
        curvature_peak_ratio=float(candidate.curvature_peak_ratio),
        curvature_alignment=float(candidate.curvature_alignment),
    )


def _linear_rss(x: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return float(np.sum((y - pred) ** 2))


def _bic(rss: float, n: int, k: int) -> float:
    rss_clamped = max(float(rss), 1e-12)
    return float(n * np.log(rss_clamped / max(n, 1)) + k * np.log(max(n, 2)))


def _middle_score_from_posterior(
    knee_p: float,
    knee_ci: tuple[float | None, float | None],
    L_bounds: tuple[float, float],
    knee_strength: float,
    delta_gamma: float,
) -> float:
    lo, hi = knee_ci
    l_min, l_max = L_bounds
    span = max(l_max - l_min, 1e-12)
    if lo is None or hi is None:
        ci_width_frac = 1.0
    else:
        ci_width_frac = float(np.clip((hi - lo) / span, 0.0, 1.0))

    p = float(np.clip(knee_p, 0.0, 1.0))
    p_ambiguity = 1.0 - abs(2.0 * p - 1.0)
    strength_ambiguity = float(1.0 / (1.0 + np.exp(max(knee_strength, -6.0))))
    dg = float(np.clip(abs(delta_gamma) / 0.35, 0.0, 1.0))
    score = float(np.clip(0.40 * p_ambiguity + 0.30 * ci_width_frac + 0.20 * strength_ambiguity + 0.10 * dg, 0.0, 1.0))
    return score


def _cv(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    med = float(np.median(arr))
    if abs(med) <= 1e-12:
        return None
    return float(np.std(arr, ddof=0) / abs(med))


def _std(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    return float(np.std(arr, ddof=0))


def _detect_forbidden_band(
    L: np.ndarray,
    eta: np.ndarray,
    window: int,
    zscore: float,
) -> tuple[float, float] | None:
    n = L.size
    if n < 6:
        return None

    w = max(5, int(window))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return None

    x = safe_log(L)
    y = safe_log(np.maximum(eta, 1e-12))
    half = w // 2

    slopes = np.full(n, np.nan, dtype=float)
    residuals = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        if end - start < 4:
            continue
        xw = x[start:end]
        yw = y[start:end]
        slope, intercept, _, _ = theilslopes(yw, xw)
        pred = slope * xw + intercept
        slopes[i] = float(slope)
        residuals[i] = float(np.sqrt(np.mean((yw - pred) ** 2)))

    slope_var = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        vals = slopes[start:end]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            slope_var[i] = float(np.var(vals))

    slope_thr = _robust_threshold(slope_var, zscore)
    resid_thr = _robust_threshold(residuals, zscore)
    if slope_thr is None or resid_thr is None:
        return None

    mask = (slope_var > slope_thr) & (residuals > resid_thr)
    runs = _contiguous_true_runs(mask)
    if not runs:
        return None
    start_idx, end_idx = max(runs, key=lambda r: r[1] - r[0])
    if end_idx - start_idx < 1:
        return None
    return float(L[start_idx]), float(L[end_idx])


def _robust_threshold(values: np.ndarray, zscore: float) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    return med + zscore * max(scale, 1e-12)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _no_knee_output(
    bic0: float,
    forbidden: tuple[float, float] | None,
    rejection_reasons: list[str],
    knee_p: float,
    knee_ci: tuple[float | None, float | None],
    knee_cov: float | None,
    post_slope_std: float | None,
) -> KneeOutputs:
    band: tuple[float | None, float | None]
    if forbidden is None:
        band = (None, None)
        forbidden_detected = False
    else:
        band = (float(forbidden[0]), float(forbidden[1]))
        forbidden_detected = True
    return KneeOutputs(
        L_k=None,
        knee_window=(None, None),
        forbidden_band=band,
        confidence=0.0,
        has_knee=False,
        delta_bic=0.0,
        bic_no_knee=float(bic0),
        bic_knee=float("inf"),
        bootstrap_cov=knee_cov,
        bootstrap_count=0,
        forbidden_detected=forbidden_detected,
        slope_left=None,
        slope_right=None,
        rejection_reasons=list(rejection_reasons),
        knee_p=float(knee_p),
        knee_ci=knee_ci,
        knee_strength=0.0,
        delta_gamma=None,
        resid_improvement=None,
        middle_score=None,
        middle_label="resolved",
        post_slope_std=post_slope_std,
        curvature_peak_ratio=None,
        curvature_alignment=None,
    )
