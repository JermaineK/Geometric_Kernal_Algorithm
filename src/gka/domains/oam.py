"""OAM image domain adapter."""

from __future__ import annotations

from dataclasses import dataclass

from gka.domains.base import TabularAdapterBase


@dataclass
class OAMAdapter(TabularAdapterBase):
    name: str = "oam"
