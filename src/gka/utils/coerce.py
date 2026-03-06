"""Type coercion utilities for safe data conversion."""

from __future__ import annotations

from typing import Any

import pandas as pd


def safe_float(v: Any) -> float | None:
    """Safely convert value to float, returning None if conversion fails."""
    try:
        if v is None:
            return None
        fv = float(v)
    except (ValueError, TypeError):
        return None
    if pd.isna(fv):
        return None
    return fv


def as_float(row: pd.Series, key: str, default: float = float("nan")) -> float:
    """Extract float value from Series, returning default on failure."""
    try:
        v = float(row.get(key, default))
        return v
    except (ValueError, TypeError):
        return default


def as_bool(row: pd.Series, key: str, default: bool = False) -> bool:
    """Extract boolean value from Series, returning default on failure."""
    v = row.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    try:
        return bool(int(v))
    except (ValueError, TypeError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int, returning default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
