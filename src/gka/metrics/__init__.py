"""Higher-level diagnostic metrics."""

from .eigen_stability import EigenStabilityResult, estimate_eigen_stability
from .forbidden_middle import ForbiddenMiddleResult, compute_forbidden_middle_score

__all__ = [
    "EigenStabilityResult",
    "estimate_eigen_stability",
    "ForbiddenMiddleResult",
    "compute_forbidden_middle_score",
]
