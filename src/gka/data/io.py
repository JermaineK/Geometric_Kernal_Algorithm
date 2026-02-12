"""I/O helpers for canonical GKA datasets and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class DatasetIOError(FileNotFoundError):
    """Raised when expected dataset files are missing."""


def ensure_dataset_paths(dataset_path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(dataset_path)
    dataset_yaml = root / "dataset.yaml"
    samples_parquet = root / "samples.parquet"
    if not root.exists():
        raise DatasetIOError(f"Dataset folder does not exist: {root}")
    if not dataset_yaml.exists():
        raise DatasetIOError(f"Missing dataset manifest: {dataset_yaml}")
    if not samples_parquet.exists():
        raise DatasetIOError(f"Missing sample table: {samples_parquet}")
    return root, dataset_yaml, samples_parquet


def load_dataset_spec(dataset_path: str | Path) -> dict[str, Any]:
    _, dataset_yaml, _ = ensure_dataset_paths(dataset_path)
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"dataset.yaml must be a mapping: {dataset_yaml}")
    return data


def load_samples(dataset_path: str | Path) -> pd.DataFrame:
    _, _, samples_parquet = ensure_dataset_paths(dataset_path)
    return pd.read_parquet(samples_parquet)


def write_dataset_spec(spec: dict[str, Any], dataset_path: str | Path) -> None:
    root = Path(dataset_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def write_samples(samples: pd.DataFrame, dataset_path: str | Path) -> None:
    root = Path(dataset_path)
    root.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(root / "samples.parquet", index=False)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def prepare_dataset_layout(dataset_path: str | Path) -> None:
    root = Path(dataset_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "arrays").mkdir(exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
