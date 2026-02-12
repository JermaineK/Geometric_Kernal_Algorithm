from pathlib import Path

import pandas as pd
import yaml

from gka.data.validators import validate_dataset


def test_schema_validation_pass(tmp_path: Path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    spec = {
        "schema_version": 1,
        "domain": "weather",
        "id": "toy",
        "description": "toy",
        "units": {"time": "index", "L": "km", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {}},
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
    with (ds / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    rows = [
        {"case_id": "a", "t": 0, "L": 10.0, "hand": "L", "O": 1.0},
        {"case_id": "a", "t": 0, "L": 10.0, "hand": "R", "O": 0.8},
    ]
    pd.DataFrame(rows).to_parquet(ds / "samples.parquet", index=False)

    report = validate_dataset(ds)
    assert report.valid


def test_schema_validation_missing_pair(tmp_path: Path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    spec = {
        "schema_version": 1,
        "domain": "weather",
        "id": "toy",
        "description": "toy",
        "units": {"time": "index", "L": "km", "omega": "rad/s"},
        "mirror": {"type": "label_swap", "details": {}},
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
    with (ds / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    pd.DataFrame([{"case_id": "a", "t": 0, "L": 10.0, "hand": "L", "O": 1.0}]).to_parquet(
        ds / "samples.parquet", index=False
    )

    report = validate_dataset(ds)
    assert not report.valid
