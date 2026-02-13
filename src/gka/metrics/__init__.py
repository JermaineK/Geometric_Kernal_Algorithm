"""Higher-level diagnostic metrics."""

from .forbidden_middle import ForbiddenMiddleResult, compute_forbidden_middle_score

__all__ = [
    "ForbiddenMiddleResult",
    "compute_forbidden_middle_score",
]
