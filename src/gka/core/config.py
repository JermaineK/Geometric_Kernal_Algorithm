"""Configuration loading and resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


DEFAULT_CONFIG: dict[str, Any] = {
    "knee": {
        "method": "segmented",
        "rho": 1.5,
        "min_points": 6,
        "bic_delta_min": 10.0,
        "knee_p_min": 0.35,
        "knee_strength_min": 0.5,
        "max_candidates": 6,
        "bootstrap_n": 200,
        "bootstrap_cov_max": 0.25,
        "edge_buffer_frac": 0.15,
        "min_points_each_side": 3,
        "min_frac_each_side": 0.2,
        "slope_abs_max": 6.0,
        "post_slope_max": 6.0,
        "resid_improvement_min": 0.0,
        "curvature_peak_ratio_min": 0.0,
        "curvature_alignment_min": 0.0,
        "instability_window": 7,
        "instability_zscore": 2.0,
        "middle_score_threshold": 0.6,
    },
    "scaling": {
        "method": "wls",
        "robust": True,
        "huber_delta": 1.5,
        "min_sizes": 4,
        "exclude_forbidden": True,
        "ci": "bootstrap",
        "bootstrap_n": 2000,
    },
    "stability": {
        "b": 2.0,
        "gamma_min": 0.0,
        "marginal_eps": 0.1,
    },
    "parity_significance": {
        "enabled": True,
        "n_perm": 200,
        "alpha": 0.05,
        "eta_min": 0.03,
        "direction_min": 0.30,
    },
    "stability_score": {
        "eps_log": 0.15,
    },
    "bands": {
        "n_max": 3,
        "rel_tol": 0.15,
    },
    "impedance": {
        "enabled": True,
        "tolerance": 0.10,
    },
    "ticks": {
        "method": "welch",
        "ridge_min_snr": 3.0,
    },
    "nulls": {
        "enabled": True,
        "n": 200,
        "models": ["time_shuffle", "mirror_swap", "phase_scramble", "block_bootstrap"],
    },
    "seed": 42,
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries and return a new dictionary."""

    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping at root: {p}")
    return data


def resolve_config(
    dataset_spec: dict[str, Any],
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve run configuration from defaults, dataset spec, and optional user file."""

    resolved = deepcopy(DEFAULT_CONFIG)
    analysis = dataset_spec.get("analysis", {})
    if analysis:
        resolved = deep_merge(resolved, analysis)
    if config_path is not None:
        resolved = deep_merge(resolved, load_yaml(config_path))
    if overrides:
        resolved = deep_merge(resolved, overrides)
    _validate_methods(resolved)
    return resolved


def dump_yaml(data: dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _validate_methods(cfg: dict[str, Any]) -> None:
    knee_method = cfg.get("knee", {}).get("method")
    if knee_method not in {"segmented", "log_curvature", "cpt"}:
        raise ConfigError(
            f"Unsupported knee.method '{knee_method}'. Supported: segmented|log_curvature|cpt"
        )

    scaling_method = cfg.get("scaling", {}).get("method")
    if scaling_method not in {"wls", "theil_sen"}:
        raise ConfigError(
            f"Unsupported scaling.method '{scaling_method}'. Supported: wls|theil_sen"
        )

    tick_method = cfg.get("ticks", {}).get("method")
    if tick_method not in {"welch", "cwt"}:
        raise ConfigError(f"Unsupported ticks.method '{tick_method}'. Supported: welch|cwt")
