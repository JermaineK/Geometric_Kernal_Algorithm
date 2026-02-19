"""Core package types used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Canonical in-memory dataset object produced by domain adapters."""

    dataset_path: Path
    samples: pd.DataFrame
    dataset_spec: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MirrorSpec:
    """Mirror involution mapping definition for parity split."""

    mirror_type: str
    pair_index: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observable:
    """Primary observable with optional aux fields."""

    X: np.ndarray
    names: list[str] = field(default_factory=lambda: ["O"])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SizeSeries:
    """Size proxy series aligned to samples."""

    L: np.ndarray


@dataclass(frozen=True)
class FrequencySeries:
    """Frequency axis series (optional by domain)."""

    omega: np.ndarray | None


@dataclass(frozen=True)
class ImpedanceInputs:
    """Inputs required for impedance alignment check."""

    omega_k: np.ndarray | float | None
    L: np.ndarray | float | None
    cm_or_v: np.ndarray | float | None
    a_or_L: np.ndarray | float | None = None


@dataclass(frozen=True)
class GeometryOutputs:
    kappa: np.ndarray
    tau: np.ndarray
    R: np.ndarray
    R_cov: np.ndarray


@dataclass(frozen=True)
class TickOutputs:
    Omega_candidates: np.ndarray
    omega_band: tuple[float, float] | None
    R_Omega: float
    ridge_strength: float


@dataclass(frozen=True)
class ParityOutputs:
    X_plus: np.ndarray
    X_minus: np.ndarray
    E_plus: float
    E_minus: float
    eta: np.ndarray


@dataclass(frozen=True)
class KneeOutputs:
    L_k: float | None
    knee_window: tuple[float | None, float | None]
    forbidden_band: tuple[float | None, float | None]
    confidence: float
    has_knee: bool
    delta_bic: float
    bic_no_knee: float
    bic_knee: float
    bootstrap_cov: float | None
    bootstrap_count: int
    forbidden_detected: bool
    slope_left: float | None
    slope_right: float | None
    rejection_reasons: list[str]
    knee_p: float = 0.0
    knee_ci: tuple[float | None, float | None] = (None, None)
    knee_strength: float = 0.0
    delta_gamma: float | None = None
    resid_improvement: float | None = None
    middle_score: float | None = None
    middle_label: str = "resolved"
    post_slope_std: float | None = None
    curvature_peak_ratio: float | None = None
    curvature_alignment: float | None = None
    candidate_count_proposed: int = 0
    candidate_count_evaluated: int = 0
    candidate_count_sanity_pass: int = 0


@dataclass(frozen=True)
class ScalingOutputs:
    gamma: float
    Delta_hat: float
    ci: tuple[float, float]
    drift: float
    n_points: int


@dataclass(frozen=True)
class StabilityOutputs:
    stability_class: str
    lambda_eff: float
    ineq_pass: bool


@dataclass(frozen=True)
class ImpedanceOutputs:
    ratio: float | None
    passed: bool | None
    tolerance: float | None


@dataclass(frozen=True)
class CoherenceOutputs:
    A: float
    F: float
    P_lock: float


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: Sequence[ValidationIssue]
    pairing_completeness: float


@dataclass(frozen=True)
class PipelineResult:
    summary: pd.DataFrame
    null_summary: pd.DataFrame | None
    metadata: dict[str, Any]
