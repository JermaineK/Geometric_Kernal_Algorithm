"""EM resonator domain adapter."""

from __future__ import annotations

from dataclasses import dataclass

from gka.domains.base import TabularAdapterBase


@dataclass
class EMResonatorAdapter(TabularAdapterBase):
    name: str = "em_resonator"
