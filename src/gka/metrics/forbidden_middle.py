"""Forbidden-middle scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForbiddenMiddleResult:
    score: float
    label: str
    components: dict[str, float]
    reason_codes: list[str]
    width: float | None
    center: float | None


def compute_forbidden_middle_score(
    knee_probability: float,
    knee_ci: tuple[float | None, float | None],
    L_bounds: tuple[float, float],
    knee_strength: float,
    delta_gamma: float | None,
    slope_drift: float,
    ridge_strength: float,
    bootstrap_cov: float | None = None,
    threshold: float = 0.6,
    posterior_ci_width_min: float = 0.25,
    posterior_ambiguity_min: float = 0.45,
    posterior_cov_min: float = 0.35,
    slope_drift_min: float = 0.10,
) -> ForbiddenMiddleResult:
    lo, hi = knee_ci
    l_min, l_max = L_bounds
    span = max(float(l_max - l_min), 1e-12)
    if lo is None or hi is None:
        ci_width_frac = 1.0
    else:
        ci_width_frac = float(np.clip((float(hi) - float(lo)) / span, 0.0, 1.0))

    p = float(np.clip(knee_probability, 0.0, 1.0))
    p_ambiguity = 1.0 - abs(2.0 * p - 1.0)  # high when posterior is indecisive

    k_strength = float(max(knee_strength, -4.0))
    strength_ambiguity = float(1.0 / (1.0 + np.exp(k_strength)))  # high for weak effects

    dg = 0.0 if delta_gamma is None or not np.isfinite(delta_gamma) else float(abs(delta_gamma))
    slope_instability = float(np.clip(dg / 0.35, 0.0, 1.0))
    drift_score = float(np.clip(float(max(slope_drift, 0.0)) / 0.2, 0.0, 1.0))
    ridge_broad = float(1.0 / (1.0 + max(float(ridge_strength), 0.0)))
    cov_term = (
        0.0
        if bootstrap_cov is None or not np.isfinite(bootstrap_cov)
        else float(np.clip(float(bootstrap_cov) / 0.5, 0.0, 1.0))
    )

    score = float(
        0.22 * p_ambiguity
        + 0.22 * ci_width_frac
        + 0.15 * strength_ambiguity
        + 0.13 * slope_instability
        + 0.10 * drift_score
        + 0.10 * ridge_broad
        + 0.08 * cov_term
    )
    score = float(np.clip(score, 0.0, 1.0))

    posterior_diffuse = bool(
        ci_width_frac >= posterior_ci_width_min and p_ambiguity >= posterior_ambiguity_min
    )
    posterior_multimodal = bool(
        bootstrap_cov is not None
        and np.isfinite(bootstrap_cov)
        and float(bootstrap_cov) >= posterior_cov_min
    )
    slope_drift_high = bool(max(float(slope_drift), 0.0) >= slope_drift_min)
    posterior_gate = posterior_diffuse or posterior_multimodal

    reason_codes: list[str] = []
    if posterior_diffuse:
        reason_codes.append("posterior_diffuse")
    if posterior_multimodal:
        reason_codes.append("posterior_multimodal")
    if slope_drift_high:
        reason_codes.append("slope_drift_high")
    if score >= threshold:
        reason_codes.append("score_above_threshold")

    # Deterministic decision: forbidden only when posterior ambiguity and slope drift co-occur.
    label = "forbidden_middle" if (posterior_gate and slope_drift_high) else "resolved"
    if label == "forbidden_middle":
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            width = float(hi - lo)
            center = float((hi + lo) / 2.0)
        else:
            width = float(ci_width_frac * span)
            center = None
    else:
        width = 0.0
        center = None

    return ForbiddenMiddleResult(
        score=score,
        label=label,
        components={
            "p_ambiguity": p_ambiguity,
            "ci_width_frac": ci_width_frac,
            "strength_ambiguity": strength_ambiguity,
            "slope_instability": slope_instability,
            "drift_score": drift_score,
            "ridge_broad": ridge_broad,
            "cov_term": cov_term,
        },
        reason_codes=reason_codes,
        width=width,
        center=center,
    )
