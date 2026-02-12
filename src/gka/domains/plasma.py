"""Plasma domain adapter."""

from __future__ import annotations

from dataclasses import dataclass

from gka.domains.base import TabularAdapterBase


@dataclass
class PlasmaAdapter(TabularAdapterBase):
    name: str = "plasma"
