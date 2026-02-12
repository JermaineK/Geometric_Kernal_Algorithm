"""Memoization wrappers."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def memoize(maxsize: int = 128):
    def decorator(func: F) -> F:
        return lru_cache(maxsize=maxsize)(func)  # type: ignore[return-value]

    return decorator
