"""Synthetic benchmark domain adapter."""

from __future__ import annotations

from dataclasses import dataclass

from gka.domains.base import TabularAdapterBase


@dataclass
class SyntheticAdapter(TabularAdapterBase):
    name: str = "synthetic"
