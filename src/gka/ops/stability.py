"""Stability classification stage."""

from __future__ import annotations

from gka.core.types import StabilityOutputs


def classify_stability(
    gamma: float,
    drift: float,
    b: float = 2.0,
    gamma_min: float = 0.0,
    marginal_eps: float = 0.1,
) -> StabilityOutputs:
    del drift  # Stability class is defined directly in gamma-space.
    lambda_eff = float(b**gamma)
    ineq_pass = bool(gamma > marginal_eps)

    if gamma > marginal_eps:
        cls = "stable"
    elif gamma < -marginal_eps:
        cls = "forbidden"
    else:
        cls = "marginal"

    return StabilityOutputs(stability_class=cls, lambda_eff=lambda_eff, ineq_pass=ineq_pass)
