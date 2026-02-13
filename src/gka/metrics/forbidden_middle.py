"""Forbidden-middle scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForbiddenMiddleResult:
    score: float
    label: str
    components: dict[str, float]


def compute_forbidden_middle_score(
    knee_probability: float,
    knee_ci: tuple[float | None, float | None],
    L_bounds: tuple[float, float],
    knee_strength: float,
    delta_gamma: float | None,
    slope_drift: float,
    ridge_strength: float,
    threshold: float = 0.6,
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

    score = float(
        0.25 * p_ambiguity
        + 0.25 * ci_width_frac
        + 0.15 * strength_ambiguity
        + 0.15 * slope_instability
        + 0.10 * drift_score
        + 0.10 * ridge_broad
    )
    score = float(np.clip(score, 0.0, 1.0))
    label = "forbidden_middle" if score >= threshold else "resolved"
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
        },
    )
