"""Implementation of `gka schema`."""

from __future__ import annotations

import argparse

import yaml


SCHEMA_TEMPLATE = {
    "schema_version": 1,
    "domain": "weather",
    "id": "example_dataset",
    "description": "Example dataset manifest",
    "units": {"time": "hours since 1970-01-01", "L": "km", "omega": "rad/s"},
    "mirror": {"type": "label_swap", "details": {"left": "L", "right": "R"}},
    "columns": {
        "time": "t",
        "size": "L",
        "handedness": "hand",
        "group": "case_id",
        "observable": ["O"],
    },
    "analysis": {
        "knee": {"method": "segmented", "rho": 1.5},
        "scaling": {"method": "wls", "min_points": 4, "exclude_forbidden": True},
        "stability": {"b": 2.0},
        "impedance": {"enabled": True, "tolerance": 0.1},
    },
}


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("schema", help="Print canonical dataset.yaml template")
    parser.set_defaults(func=cmd_schema)


def cmd_schema(args: argparse.Namespace) -> int:
    del args
    print(yaml.safe_dump(SCHEMA_TEMPLATE, sort_keys=False))
    return 0
