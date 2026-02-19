"""Adapter protocol for external, non-canonical datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class AdapterOutput:
    """Standardized adapter payload consumed by preparation utilities."""

    X: np.ndarray
    mirror_op: dict[str, Any]
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class DataAdapter(Protocol):
    """Protocol for converting domain-specific raw data to standardized tensors."""

    name: str

    def prepare(self, source: str, **kwargs: Any) -> AdapterOutput:
        """Load source data and emit standardized tensors and metadata."""
