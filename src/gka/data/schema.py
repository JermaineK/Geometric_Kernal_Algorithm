"""Dataset schema constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass

REQUIRED_DATASET_FIELDS = [
    "schema_version",
    "domain",
    "id",
    "description",
    "units",
    "mirror",
    "columns",
    "analysis",
]

REQUIRED_COLUMNS_MAP_FIELDS = ["time", "size", "handedness", "group", "observable"]
ALLOWED_DOMAINS = {
    "weather",
    "josephson",
    "oam",
    "em_resonator",
    "plasma",
    "synthetic",
    "custom",
}
ALLOWED_HANDS = {"L", "R"}


@dataclass(frozen=True)
class SchemaCheck:
    ok: bool
    message: str
