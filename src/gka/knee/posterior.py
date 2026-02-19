"""Bootstrap posterior estimation for knee diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gka.knee.two_stage import KneeCandidate, evaluate_candidates, propose_candidates


@dataclass(frozen=True)
class KneePosterior:
    best_candidate: KneeCandidate | None
    knee_probability: float
    knee_ci: tuple[float | None, float | None]
    knee_samples: np.ndarray
    post_slope_samples: np.ndarray
    accepted_fraction: float
    candidate_count_proposed: int
    candidate_count_evaluated: int
    candidate_count_sanity_pass: int


def estimate_knee_posterior(
    L: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    bic_delta_min: float,
    knee_strength_min: float,
    min_points_each_side: int,
    min_frac_each_side: float,
    slope_abs_max: float,
    edge_buffer_frac: float,
    post_slope_max: float,
    resid_improvement_min: float,
    curvature_peak_ratio_min: float,
    curvature_alignment_min: float,
    max_candidates: int,
    n_boot: int,
    rng: np.random.Generator,
) -> KneePosterior:
    n = x.size
    if n < 4:
        return KneePosterior(
            best_candidate=None,
            knee_probability=0.0,
            knee_ci=(None, None),
            knee_samples=np.array([], dtype=float),
            post_slope_samples=np.array([], dtype=float),
            accepted_fraction=0.0,
            candidate_count_proposed=0,
            candidate_count_evaluated=0,
            candidate_count_sanity_pass=0,
        )

    rss0 = _linear_rss(x, y)
    bic0 = _bic(rss0, n=n, k=2)
    proposed = propose_candidates(
        x=x,
        y=y,
        method=method,
        min_points_each_side=min_points_each_side,
        min_frac_each_side=min_frac_each_side,
        max_candidates=max_candidates,
    )
    evaluated = evaluate_candidates(
        x=x,
        y=y,
        L=L,
        candidate_indices=proposed,
        bic_no_knee=bic0,
        slope_abs_max=slope_abs_max,
        edge_buffer_frac=edge_buffer_frac,
        post_slope_max=post_slope_max,
        resid_improvement_min=resid_improvement_min,
        curvature_peak_ratio_min=curvature_peak_ratio_min,
        curvature_alignment_min=curvature_alignment_min,
    )
    best_candidate = _choose_best(evaluated, bic_delta_min=bic_delta_min, knee_strength_min=knee_strength_min)
    sanity_pass = int(sum(1 for c in evaluated if c.sanity_ok))

    boot_samples: list[float] = []
    boot_post_slopes: list[float] = []
    boot_valid = 0
    for _ in range(max(int(n_boot), 0)):
        idx = rng.integers(0, n, size=n)
        Lb = L[idx]
        xb = x[idx]
        yb = y[idx]
        order = np.argsort(Lb)
        Lb = Lb[order]
        xb = xb[order]
        yb = yb[order]
        if np.unique(Lb).size < max(4, min_points_each_side + 1):
            continue
        rss0_b = _linear_rss(xb, yb)
        bic0_b = _bic(rss0_b, n=xb.size, k=2)
        cand_b = propose_candidates(
            x=xb,
            y=yb,
            method=method,
            min_points_each_side=min_points_each_side,
            min_frac_each_side=min_frac_each_side,
            max_candidates=max_candidates,
        )
        if not cand_b:
            continue
        eval_b = evaluate_candidates(
            x=xb,
            y=yb,
            L=Lb,
            candidate_indices=cand_b,
            bic_no_knee=bic0_b,
            slope_abs_max=slope_abs_max,
            edge_buffer_frac=edge_buffer_frac,
            post_slope_max=post_slope_max,
            resid_improvement_min=resid_improvement_min,
            curvature_peak_ratio_min=curvature_peak_ratio_min,
            curvature_alignment_min=curvature_alignment_min,
        )
        if not eval_b:
            continue
        boot_valid += 1
        # Slightly relaxed acceptance in bootstrap to avoid over-conservative posterior.
        chosen = _choose_best(
            eval_b,
            bic_delta_min=max(0.0, 0.5 * bic_delta_min),
            knee_strength_min=max(-0.5, 0.5 * knee_strength_min),
        )
        if chosen is not None:
            boot_samples.append(float(chosen.L_k))
            boot_post_slopes.append(float(chosen.slope_right))

    samples = np.asarray(boot_samples, dtype=float)
    if boot_valid <= 0:
        return KneePosterior(
            best_candidate=best_candidate,
            knee_probability=0.0,
            knee_ci=(None, None),
            knee_samples=np.array([], dtype=float),
            post_slope_samples=np.array([], dtype=float),
            accepted_fraction=0.0,
            candidate_count_proposed=int(len(proposed)),
            candidate_count_evaluated=int(len(evaluated)),
            candidate_count_sanity_pass=int(sanity_pass),
        )

    knee_probability = float(samples.size / boot_valid)
    if samples.size >= 2:
        knee_ci = (float(np.quantile(samples, 0.1)), float(np.quantile(samples, 0.9)))
    elif samples.size == 1:
        knee_ci = (float(samples[0]), float(samples[0]))
    else:
        knee_ci = (None, None)

    return KneePosterior(
        best_candidate=best_candidate,
        knee_probability=knee_probability,
        knee_ci=knee_ci,
        knee_samples=samples,
        post_slope_samples=np.asarray(boot_post_slopes, dtype=float),
        accepted_fraction=float(samples.size / max(int(n_boot), 1)),
        candidate_count_proposed=int(len(proposed)),
        candidate_count_evaluated=int(len(evaluated)),
        candidate_count_sanity_pass=int(sanity_pass),
    )


def _choose_best(
    candidates: list[KneeCandidate],
    bic_delta_min: float,
    knee_strength_min: float,
) -> KneeCandidate | None:
    if not candidates:
        return None
    for cand in candidates:
        if not cand.sanity_ok:
            continue
        if cand.delta_bic < bic_delta_min:
            continue
        if cand.strength < knee_strength_min:
            continue
        return cand
    return None


def _linear_rss(x: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return float(np.sum((y - pred) ** 2))


def _bic(rss: float, n: int, k: int) -> float:
    rss_clamped = max(float(rss), 1e-12)
    return float(n * np.log(rss_clamped / max(n, 1)) + k * np.log(max(n, 2)))
