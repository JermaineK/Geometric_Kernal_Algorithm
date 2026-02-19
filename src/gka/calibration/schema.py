"""Schema validators for calibration and robustness gate payloads."""

from __future__ import annotations

from typing import Any, Mapping


CALIBRATION_SCHEMA_VERSION = 1
ROBUSTNESS_GATE_SCHEMA_VERSION = 1


class CalibrationSchemaError(ValueError):
    """Raised when a calibration or scoring payload violates schema."""


def validate_calibration_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CalibrationSchemaError("Calibration payload must be a mapping")

    schema_version = _as_int(payload.get("schema_version"), "schema_version")
    if schema_version != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationSchemaError(
            f"Unsupported calibration schema_version={schema_version}; "
            f"expected {CALIBRATION_SCHEMA_VERSION}"
        )

    generated_at = payload.get("generated_at_utc")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise CalibrationSchemaError("Calibration payload missing non-empty generated_at_utc")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise CalibrationSchemaError("Calibration payload missing mapping field: source")

    objective = payload.get("objective")
    if not isinstance(objective, Mapping):
        raise CalibrationSchemaError("Calibration payload missing mapping field: objective")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise CalibrationSchemaError("Calibration payload missing mapping field: thresholds")

    knee_p_min = _as_float(thresholds.get("knee_p_min"), "thresholds.knee_p_min")
    knee_strength_min = _as_float(
        thresholds.get("knee_strength_min"), "thresholds.knee_strength_min"
    )
    knee_delta_bic_min = _as_float(
        thresholds.get("knee_delta_bic_min"), "thresholds.knee_delta_bic_min"
    )

    middle_score_band = thresholds.get("middle_score_band")
    if not isinstance(middle_score_band, Mapping):
        raise CalibrationSchemaError("thresholds.middle_score_band must be a mapping")
    forbidden_min = _as_float(
        middle_score_band.get("forbidden_min"), "thresholds.middle_score_band.forbidden_min"
    )

    if not 0.0 <= knee_p_min <= 1.0:
        raise CalibrationSchemaError("thresholds.knee_p_min must be within [0, 1]")
    if not 0.0 <= forbidden_min <= 1.0:
        raise CalibrationSchemaError("thresholds.middle_score_band.forbidden_min must be within [0, 1]")

    return {
        "schema_version": schema_version,
        "generated_at_utc": generated_at,
        "source": dict(source),
        "objective": dict(objective),
        "thresholds": {
            "knee_p_min": knee_p_min,
            "knee_strength_min": knee_strength_min,
            "knee_delta_bic_min": knee_delta_bic_min,
            "middle_score_band": {"forbidden_min": forbidden_min},
        },
    }


def validate_robustness_gate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CalibrationSchemaError("robustness_gate payload must be a mapping")

    schema_version = _as_int(payload.get("schema_version", ROBUSTNESS_GATE_SCHEMA_VERSION), "schema_version")
    if schema_version != ROBUSTNESS_GATE_SCHEMA_VERSION:
        raise CalibrationSchemaError(
            f"Unsupported robustness_gate schema_version={schema_version}; "
            f"expected {ROBUSTNESS_GATE_SCHEMA_VERSION}"
        )

    passed = payload.get("passed")
    if not isinstance(passed, bool):
        raise CalibrationSchemaError("robustness_gate.passed must be a boolean")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise CalibrationSchemaError("robustness_gate.enabled must be a boolean")

    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise CalibrationSchemaError("robustness_gate.checks must be a non-empty list")

    normalized_checks: list[dict[str, Any]] = []
    for i, check in enumerate(checks):
        if not isinstance(check, Mapping):
            raise CalibrationSchemaError(f"robustness_gate.checks[{i}] must be a mapping")
        name = check.get("name")
        if not isinstance(name, str) or not name:
            raise CalibrationSchemaError(f"robustness_gate.checks[{i}].name must be a non-empty string")
        op = check.get("op")
        if op not in {"<=", ">="}:
            raise CalibrationSchemaError(
                f"robustness_gate.checks[{i}].op must be one of <=, >="
            )
        value = _as_float(check.get("value"), f"robustness_gate.checks[{i}].value")
        threshold = _as_float(
            check.get("threshold"), f"robustness_gate.checks[{i}].threshold"
        )
        chk_pass = check.get("pass")
        if not isinstance(chk_pass, bool):
            raise CalibrationSchemaError(
                f"robustness_gate.checks[{i}].pass must be boolean"
            )
        normalized = {
            "name": name,
            "op": op,
            "value": value,
            "threshold": threshold,
            "pass": chk_pass,
        }
        if "details" in check:
            normalized["details"] = check["details"]
        normalized_checks.append(normalized)

    return {
        "schema_version": schema_version,
        "enabled": enabled,
        "passed": passed,
        "checks": normalized_checks,
    }


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise CalibrationSchemaError(f"{field} must be numeric") from exc


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise CalibrationSchemaError(f"{field} must be integer") from exc
