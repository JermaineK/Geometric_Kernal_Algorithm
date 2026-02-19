"""Eigen-stability summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenStabilityResult:
    eigen_band: str
    stability_margin: float
    lambda_eff: float
    confidence: float
    eigen_evidence: str


def estimate_eigen_stability(
    gamma: float,
    *,
    b: float = 2.0,
    margin_eps: float = 0.05,
    gamma_ci: tuple[float, float] | None = None,
    evidence: str = "size_law",
) -> EigenStabilityResult:
    if b <= 0:
        raise ValueError("b must be positive")
    lambda_eff = float(b**gamma)
    margin = float(np.log(max(abs(lambda_eff), 1e-12)))

    if margin > margin_eps:
        band = "stable"
    elif margin < -margin_eps:
        band = "unstable"
    else:
        band = "marginal"

    confidence = _confidence_from_ci(gamma_ci=gamma_ci, eps=max(margin_eps, 1e-8))
    return EigenStabilityResult(
        eigen_band=band,
        stability_margin=margin,
        lambda_eff=lambda_eff,
        confidence=confidence,
        eigen_evidence=evidence,
    )


def _confidence_from_ci(gamma_ci: tuple[float, float] | None, eps: float) -> float:
    if gamma_ci is None:
        return 0.5
    lo, hi = gamma_ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.5
    width = max(float(hi - lo), 0.0)
    # Confidence decreases as CI spans several decision-width units.
    return float(np.clip(1.0 - width / (4.0 * eps), 0.0, 1.0))
