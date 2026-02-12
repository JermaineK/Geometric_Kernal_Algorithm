"""Validation logic for dataset schema and pairing completeness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gka.core.types import ValidationIssue, ValidationReport
from gka.data.io import load_dataset_spec, load_samples
from gka.data.schema import (
    ALLOWED_DOMAINS,
    ALLOWED_HANDS,
    REQUIRED_COLUMNS_MAP_FIELDS,
    REQUIRED_DATASET_FIELDS,
)


def validate_dataset(dataset_path: str | Path, allow_missing: bool = False) -> ValidationReport:
    issues: list[ValidationIssue] = []

    try:
        spec = load_dataset_spec(dataset_path)
        samples = load_samples(dataset_path)
    except Exception as exc:
        return ValidationReport(
            valid=False,
            issues=[ValidationIssue(level="error", code="io_error", message=str(exc), context={})],
            pairing_completeness=0.0,
        )

    issues.extend(_validate_spec(spec))
    issues.extend(_validate_samples(spec, samples, allow_missing=allow_missing))

    completeness = _pairing_completeness(spec, samples)
    if completeness < 0.99:
        level = "warning" if allow_missing else "error"
        issues.append(
            ValidationIssue(
                level=level,
                code="pairing_completeness",
                message=(
                    f"Pairing completeness is {completeness:.3f}; expected >= 0.990. "
                    "Pass --allow-missing to continue with warning."
                ),
                context={"pairing_completeness": completeness},
            )
        )

    has_error = any(issue.level == "error" for issue in issues)
    return ValidationReport(valid=not has_error, issues=issues, pairing_completeness=completeness)


def _validate_spec(spec: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for field in REQUIRED_DATASET_FIELDS:
        if field not in spec:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_dataset_field",
                    message=f"Missing required dataset.yaml field '{field}'",
                    context={"field": field},
                )
            )

    domain = spec.get("domain")
    if domain is not None and domain not in ALLOWED_DOMAINS:
        issues.append(
            ValidationIssue(
                level="error",
                code="invalid_domain",
                message=f"dataset.yaml domain '{domain}' is unsupported",
                context={"allowed": sorted(ALLOWED_DOMAINS)},
            )
        )

    columns = spec.get("columns", {})
    if not isinstance(columns, dict):
        issues.append(
            ValidationIssue(
                level="error",
                code="invalid_columns_mapping",
                message="dataset.yaml columns must be a mapping",
                context={},
            )
        )
        return issues

    for field in REQUIRED_COLUMNS_MAP_FIELDS:
        if field not in columns:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_columns_field",
                    message=f"dataset.yaml columns missing '{field}'",
                    context={"field": field},
                )
            )

    observable = columns.get("observable")
    if observable is not None and not isinstance(observable, list):
        issues.append(
            ValidationIssue(
                level="error",
                code="invalid_observable_mapping",
                message="dataset.yaml columns.observable must be a list",
                context={"value": observable},
            )
        )

    return issues


def _validate_samples(spec: dict[str, Any], samples: pd.DataFrame, allow_missing: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    columns_map = spec.get("columns", {})

    group_col = columns_map.get("group", "case_id")
    time_col = columns_map.get("time", "t")
    size_col = columns_map.get("size", "L")
    hand_col = columns_map.get("handedness", "hand")
    observable_cols = columns_map.get("observable", ["O"])

    expected_columns = [group_col, time_col, size_col, hand_col]
    for col in expected_columns:
        if col not in samples.columns:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_samples_column",
                    message=f"samples.parquet missing required column '{col}'",
                    context={"column": col},
                )
            )

    has_observable = any(col in samples.columns for col in observable_cols) or "O_path" in samples.columns
    if not has_observable:
        issues.append(
            ValidationIssue(
                level="error",
                code="missing_observable",
                message=(
                    "samples.parquet must contain at least one observable column listed in "
                    f"dataset.yaml columns.observable={observable_cols} or O_path"
                ),
                context={},
            )
        )

    if hand_col in samples.columns:
        hand_values = set(samples[hand_col].dropna().astype(str).unique().tolist())
        invalid_hands = sorted(hand_values - ALLOWED_HANDS)
        if invalid_hands:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="invalid_hand_values",
                    message=f"Invalid hand values found: {invalid_hands}; expected only ['L', 'R']",
                    context={"invalid_values": invalid_hands},
                )
            )

    if size_col in samples.columns:
        non_numeric = samples[size_col].isna().sum() + (~np.isfinite(pd.to_numeric(samples[size_col], errors="coerce"))).sum()
        if int(non_numeric) > 0:
            level = "warning" if allow_missing else "error"
            issues.append(
                ValidationIssue(
                    level=level,
                    code="invalid_size_values",
                    message=f"Column '{size_col}' has {int(non_numeric)} non-finite values",
                    context={"column": size_col},
                )
            )

    return issues


def _pairing_completeness(spec: dict[str, Any], samples: pd.DataFrame) -> float:
    columns_map = spec.get("columns", {})
    group_col = columns_map.get("group", "case_id")
    time_col = columns_map.get("time", "t")
    size_col = columns_map.get("size", "L")
    hand_col = columns_map.get("handedness", "hand")

    needed = [group_col, time_col, size_col, hand_col]
    if any(col not in samples.columns for col in needed) or samples.empty:
        return 0.0

    grouped = samples.groupby([group_col, time_col, size_col], dropna=False)[hand_col].nunique()
    if grouped.empty:
        return 0.0
    complete = (grouped == 2).sum()
    return float(complete / grouped.size)


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "valid": report.valid,
        "pairing_completeness": report.pairing_completeness,
        "issues": [
            {
                "level": issue.level,
                "code": issue.code,
                "message": issue.message,
                "context": dict(issue.context),
            }
            for issue in report.issues
        ],
    }
