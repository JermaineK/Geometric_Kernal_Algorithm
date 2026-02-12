from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class GeneratedSynthetic:
    data: pd.DataFrame
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for GKA validation")
    parser.add_argument("--config", required=True, help="Synthetic config YAML")
    parser.add_argument("--outdir", required=True, help="Output folder")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--run-index", type=int, default=0, help="Run index for scheduled configs")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return data


def generate_dataset(config: dict[str, Any], seed: int, run_index: int = 0) -> GeneratedSynthetic:
    rng = np.random.default_rng(seed)

    name = str(config.get("name", "synthetic_test"))
    domain = str(config.get("domain", "synthetic"))
    n_cases = int(config.get("n_cases", 32))
    L_values = _resolve_axis(config.get("L_values"), axis_name="L")
    omega_values = _resolve_axis(config.get("omega_values", 1.0), axis_name="omega")

    baseline_cfg = config.get("baseline_B", {"kind": "constant", "value": 1.0})
    eta_cfg = config.get("eta_model", {"type": "power_law", "A": 0.02, "gamma": 1.0})
    noise_cfg = config.get("noise", {"distribution": "gaussian", "std": 0.01, "relative": True})
    parity_cfg = config.get("parity", {"sign_mode": "fixed_per_case"})

    truth: dict[str, Any] = _truth_from_model(eta_cfg, run_index=run_index)
    if "truth" in config and isinstance(config["truth"], dict):
        truth.update(config["truth"])

    records: list[dict[str, Any]] = []
    signs_by_case: dict[int, int] = {}

    for case_idx in range(n_cases):
        if parity_cfg.get("sign_mode", "fixed_per_case") == "fixed_per_case":
            signs_by_case[case_idx] = int(rng.choice([-1, 1]))

        for L in L_values:
            for omega in omega_values:
                eta_true = float(_eta_value(L, omega, eta_cfg, rng=rng, run_index=run_index))
                eta_true = float(np.clip(eta_true, 0.0, float(config.get("eta_clip", 1.8))))

                baseline = float(_baseline_value(L, omega, baseline_cfg))
                sign = _select_sign(parity_cfg, case_idx=case_idx, rng=rng, signs_by_case=signs_by_case)

                o_left = baseline * (1.0 + sign * eta_true / 2.0)
                o_right = baseline * (1.0 - sign * eta_true / 2.0)

                o_left = _apply_noise(o_left, noise_cfg, rng)
                o_right = _apply_noise(o_right, noise_cfg, rng)

                row: dict[str, Any] = {
                    "case_id": f"case_{case_idx:04d}",
                    "L": float(L),
                    "omega": float(omega),
                    "O_L": float(max(o_left, 1e-8)),
                    "O_R": float(max(o_right, 1e-8)),
                    "domain": domain,
                    "test_name": name,
                    "eta_true": eta_true,
                }

                impedance_cfg = config.get("impedance", {})
                if isinstance(impedance_cfg, dict) and impedance_cfg.get("enabled", False):
                    c_val = float(impedance_cfg.get("c", 1.0))
                    jitter = float(impedance_cfg.get("jitter_std", 0.0))
                    omega_k = (2.0 * np.pi * c_val / float(L)) * (1.0 + rng.normal(0.0, jitter))
                    row["omega_k"] = float(omega_k)
                    row["c_true"] = c_val
                    truth["c"] = c_val

                records.append(row)

    df = pd.DataFrame.from_records(records)
    df = _apply_parity_variants(df, parity_cfg, rng)
    df = df.sort_values(["case_id", "L", "omega"]).reset_index(drop=True)

    eta_obs = np.abs(df["O_L"] - df["O_R"]) / np.maximum((df["O_L"] + df["O_R"]) / 2.0, 1e-12)
    truth["eta_observed_mean"] = float(np.mean(eta_obs))

    meta = {
        "name": name,
        "domain": domain,
        "seed": seed,
        "run_index": run_index,
        "n_rows": int(df.shape[0]),
        "n_cases": n_cases,
        "L_values": [float(v) for v in L_values],
        "omega_values": [float(v) for v in omega_values],
        "truth": truth,
        "config": {
            "baseline_B": baseline_cfg,
            "eta_model": eta_cfg,
            "noise": noise_cfg,
            "parity": parity_cfg,
            "impedance": config.get("impedance", {}),
        },
    }

    return GeneratedSynthetic(data=df, meta=meta)


def write_generated_dataset(outdir: str | Path, generated: GeneratedSynthetic) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    generated.data.to_csv(out / "data.csv", index=False)
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(generated.meta, f, indent=2, sort_keys=True)


def _resolve_axis(value: Any, axis_name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.array([float(value)], dtype=float)

    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{axis_name}_values list cannot be empty")
        return arr

    if isinstance(value, dict):
        vmin = float(value["min"])
        vmax = float(value["max"])
        n = int(value["n"])
        if n < 2:
            raise ValueError(f"{axis_name}_values n must be >=2")
        spacing = str(value.get("spacing", "linear")).lower()
        if spacing == "linear":
            return np.linspace(vmin, vmax, n)
        if spacing == "log":
            if vmin <= 0 or vmax <= 0:
                raise ValueError(f"{axis_name}_values log spacing requires positive min/max")
            return np.geomspace(vmin, vmax, n)
        raise ValueError(f"Unsupported spacing '{spacing}' for {axis_name}_values")

    raise ValueError(
        f"Invalid {axis_name}_values format: expected scalar/list/mapping, got {type(value).__name__}"
    )


def _baseline_value(L: float, omega: float, cfg: dict[str, Any]) -> float:
    kind = str(cfg.get("kind", "constant")).lower()
    if kind == "constant":
        return float(cfg.get("value", 1.0))
    if kind == "power_law":
        amp = float(cfg.get("amplitude", cfg.get("A", 1.0)))
        exp = float(cfg.get("exponent", 0.0))
        return amp * (float(L) ** exp)
    if kind == "omega_power":
        amp = float(cfg.get("amplitude", 1.0))
        exp_l = float(cfg.get("L_exponent", 0.0))
        exp_o = float(cfg.get("omega_exponent", 0.0))
        return amp * (float(L) ** exp_l) * (float(omega) ** exp_o)
    raise ValueError(f"Unsupported baseline_B kind '{kind}'")


def _eta_value(
    L: float,
    omega: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    run_index: int,
) -> float:
    model = str(cfg.get("type", "power_law")).lower()

    if model == "power_law":
        A = float(cfg.get("A", 0.01))
        gamma = float(cfg.get("gamma", 1.0))
        L0 = float(cfg.get("L0", 1.0))
        return A * ((float(L) / L0) ** gamma)

    if model == "hybrid_knee":
        A = float(cfg.get("A", 0.01))
        gamma = float(cfg.get("gamma", 1.0))
        xi = float(cfg.get("xi", 40.0))
        q = float(cfg.get("q", 2.0))
        L0 = float(cfg.get("L0", 1.0))
        decay_scale_frac = float(cfg.get("decay_scale_frac", 0.5))
        if float(L) <= xi:
            return A * ((float(L) / L0) ** gamma)
        scale = max(decay_scale_frac * xi, 1e-12)
        peak = A * ((xi / L0) ** gamma)
        return peak * np.exp(-(((float(L) - xi) / scale) ** q))

    if model == "piecewise_forbidden":
        A = float(cfg.get("A", 0.01))
        L0 = float(cfg.get("L0", 1.0))
        L_k = float(cfg.get("L_k", 30.0))
        L_k2 = float(cfg.get("L_k2", 70.0))
        gamma_hi = float(cfg.get("gamma_hi", 1.6))
        gamma_lo = float(cfg.get("gamma_lo", 0.7))
        unstable_sigma = float(cfg.get("unstable_sigma", 0.35))

        if L < L_k:
            return A * ((float(L) / L0) ** gamma_hi)
        if L > L_k2:
            bridge = A * ((L_k / L0) ** gamma_hi)
            return bridge * ((float(L) / L_k2) ** gamma_lo)

        base = A * ((L_k / L0) ** gamma_hi)
        drift = np.exp(rng.normal(0.0, unstable_sigma))
        slope = rng.uniform(min(gamma_lo, gamma_hi) - 0.5, max(gamma_lo, gamma_hi) + 0.5)
        return base * ((float(L) / L_k) ** slope) * drift

    if model == "eigen_stability":
        A = float(cfg.get("A", 0.01))
        L0 = float(cfg.get("L0", 1.0))
        gammas = cfg.get("gamma_values", [-0.5, 0.2, 1.0, 1.8])
        if not isinstance(gammas, list) or len(gammas) == 0:
            raise ValueError("eigen_stability gamma_values must be a non-empty list")
        gamma = float(gammas[run_index % len(gammas)])
        return A * ((float(L) / L0) ** gamma)

    raise ValueError(f"Unsupported eta_model type '{model}'")


def _truth_from_model(cfg: dict[str, Any], run_index: int) -> dict[str, Any]:
    model = str(cfg.get("type", "power_law")).lower()
    truth: dict[str, Any] = {"eta_model": model}

    if model == "power_law":
        truth["gamma"] = float(cfg.get("gamma", 1.0))
        truth["knee_L"] = None
    elif model == "hybrid_knee":
        truth["gamma"] = float(cfg.get("gamma", 1.0))
        truth["xi"] = float(cfg.get("xi", 40.0))
        truth["q"] = float(cfg.get("q", 2.0))
        truth["knee_L"] = float(cfg.get("xi", 40.0))
    elif model == "piecewise_forbidden":
        truth["gamma_hi"] = float(cfg.get("gamma_hi", 1.6))
        truth["gamma_lo"] = float(cfg.get("gamma_lo", 0.7))
        truth["knee_L"] = float(cfg.get("L_k", 30.0))
        truth["knee_L2"] = float(cfg.get("L_k2", 70.0))
        truth["forbidden_band"] = [truth["knee_L"], truth["knee_L2"]]
    elif model == "eigen_stability":
        gammas = cfg.get("gamma_values", [-0.5, 0.2, 1.0, 1.8])
        gamma = float(gammas[run_index % len(gammas)])
        truth["gamma"] = gamma
        truth["knee_L"] = None
    else:
        truth["knee_L"] = None

    return truth


def _select_sign(
    parity_cfg: dict[str, Any],
    case_idx: int,
    rng: np.random.Generator,
    signs_by_case: dict[int, int],
) -> int:
    mode = str(parity_cfg.get("sign_mode", "fixed_per_case")).lower()

    if mode == "fixed_positive":
        return 1
    if mode == "fixed_negative":
        return -1
    if mode == "fixed_per_case":
        return signs_by_case.get(case_idx, 1)
    if mode in {"random_per_case", "random"}:
        return int(rng.choice([-1, 1]))
    if mode == "random_per_sample":
        return int(rng.choice([-1, 1]))

    raise ValueError(
        f"Unsupported parity.sign_mode '{mode}'. "
        "Use fixed_per_case|random_per_case|random_per_sample|fixed_positive|fixed_negative"
    )


def _apply_noise(value: float, noise_cfg: dict[str, Any], rng: np.random.Generator) -> float:
    dist = str(noise_cfg.get("distribution", "gaussian")).lower()
    std = float(noise_cfg.get("std", 0.01))
    relative = bool(noise_cfg.get("relative", True))

    if std <= 0:
        return float(value)

    if dist == "gaussian":
        noise = rng.normal(0.0, std)
        return float(value * (1.0 + noise) if relative else value + noise)

    if dist == "lognormal":
        scale = rng.lognormal(mean=0.0, sigma=std)
        return float(value * scale)

    raise ValueError(f"Unsupported noise distribution '{dist}'. Use gaussian|lognormal")


def _apply_parity_variants(
    df: pd.DataFrame,
    parity_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    swap_prob = float(parity_cfg.get("label_swap_prob", 0.0))
    if swap_prob > 0:
        mask = rng.random(out.shape[0]) < swap_prob
        tmp = out.loc[mask, "O_L"].to_numpy(copy=True)
        out.loc[mask, "O_L"] = out.loc[mask, "O_R"].to_numpy()
        out.loc[mask, "O_R"] = tmp

    if bool(parity_cfg.get("pair_break", False)):
        scope = str(parity_cfg.get("pair_break_scope", "by_L")).lower()
        if scope == "global":
            shuffled = out["O_R"].to_numpy(copy=True)
            rng.shuffle(shuffled)
            out["O_R"] = shuffled
        elif scope == "by_l":
            for _, idx in out.groupby("L").groups.items():
                vals = out.loc[idx, "O_R"].to_numpy(copy=True)
                rng.shuffle(vals)
                out.loc[idx, "O_R"] = vals
        else:
            raise ValueError(
                f"Unsupported pair_break_scope '{scope}'. Use global|by_L"
            )

    return out


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    generated = generate_dataset(config, seed=args.seed, run_index=args.run_index)
    write_generated_dataset(args.outdir, generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
