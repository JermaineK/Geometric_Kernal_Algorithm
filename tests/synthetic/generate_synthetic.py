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
        t_counter = 0

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
                    "t": t_counter,
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
                t_counter += 1

    df = pd.DataFrame.from_records(records)
    df = _apply_parity_variants(df, parity_cfg, rng)
    stress_cfg = config.get("stress", {})
    if isinstance(stress_cfg, dict) and stress_cfg:
        df = _apply_stress_transforms(df, stress_cfg, rng)
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
            "stress": stress_cfg,
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

    if model == "dual_hybrid_knee":
        A1 = float(cfg.get("A1", cfg.get("A", 0.01)))
        g1 = float(cfg.get("gamma1", 1.4))
        xi1 = float(cfg.get("xi1", 24.0))
        q1 = float(cfg.get("q1", 2.0))
        A2 = float(cfg.get("A2", A1 * 0.8))
        g2 = float(cfg.get("gamma2", 1.0))
        xi2 = float(cfg.get("xi2", 70.0))
        q2 = float(cfg.get("q2", 2.0))
        L0 = float(cfg.get("L0", 1.0))
        term1 = A1 * ((float(L) / L0) ** g1) * np.exp(-((float(L) / max(xi1, 1e-12)) ** q1))
        term2 = A2 * ((float(L) / L0) ** g2) * np.exp(-((float(L) / max(xi2, 1e-12)) ** q2))
        return term1 + term2

    if model == "screened_power_law":
        A = float(cfg.get("A", 0.02))
        gamma = float(cfg.get("gamma", 1.0))
        L0 = float(cfg.get("L0", 1.0))
        xi = float(cfg.get("xi_screen", 30.0))
        q = float(cfg.get("q_screen", 2.5))
        base = A * ((float(L) / L0) ** gamma)
        return base * np.exp(-((float(L) / max(xi, 1e-12)) ** q))

    if model == "logistic_curve":
        eta_min = float(cfg.get("eta_min", 0.01))
        eta_max = float(cfg.get("eta_max", 0.35))
        L0 = float(cfg.get("L0", 40.0))
        width = max(float(cfg.get("width", 8.0)), 1e-6)
        z = (float(L) - L0) / width
        return eta_min + (eta_max - eta_min) / (1.0 + np.exp(-z))

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
    elif model == "dual_hybrid_knee":
        truth["gamma"] = float(cfg.get("gamma1", 1.4))
        truth["knee_L"] = float(cfg.get("xi1", 24.0))
        truth["knee_L2"] = float(cfg.get("xi2", 70.0))
        truth["forbidden_band"] = [truth["knee_L"], truth["knee_L2"]]
    elif model == "screened_power_law":
        truth["gamma"] = float(cfg.get("gamma", 1.0))
        truth["knee_L"] = float(cfg.get("xi_screen", 30.0))
        truth["xi"] = float(cfg.get("xi_screen", 30.0))
        # Optional explicit truth for stress controls where screening dominates.
        truth["tau_s"] = _safe_float(cfg.get("tau_s"))
    elif model == "logistic_curve":
        truth["knee_L"] = None
        truth["L0"] = float(cfg.get("L0", 40.0))
        truth["width"] = float(cfg.get("width", 8.0))
        truth["eta_min"] = float(cfg.get("eta_min", 0.01))
        truth["eta_max"] = float(cfg.get("eta_max", 0.35))
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
        out = float(value * scale)
    elif dist == "laplace":
        noise = rng.laplace(0.0, std)
        out = float(value * (1.0 + noise) if relative else value + noise)
    elif dist in {"student_t", "student-t", "t"}:
        dof = float(noise_cfg.get("df", 3.0))
        noise = rng.standard_t(max(dof, 1.1)) * std
        out = float(value * (1.0 + noise) if relative else value + noise)
    else:
        raise ValueError(f"Unsupported noise distribution '{dist}'. Use gaussian|lognormal|laplace|student_t")

    burst_prob = float(noise_cfg.get("burst_prob", 0.0))
    burst_scale = float(noise_cfg.get("burst_scale", 0.0))
    if burst_prob > 0 and rng.random() < burst_prob:
        burst = rng.normal(0.0, burst_scale)
        out = float(out * (1.0 + burst) if relative else out + burst)
    return out


def _apply_stress_transforms(
    df: pd.DataFrame,
    stress_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce").fillna(0.0).astype(float)

    if bool(stress_cfg.get("nonstationary_baseline", False)):
        drift = float(stress_cfg.get("drift_strength", 0.25))
        vol = float(stress_cfg.get("variance_strength", 0.1))
        for _, idx in out.groupby("case_id").groups.items():
            n = len(idx)
            if n <= 1:
                continue
            t_norm = np.linspace(0.0, 1.0, n)
            trend = np.exp(drift * t_norm + vol * rng.normal(0.0, 1.0, size=n) * np.sqrt(np.maximum(t_norm, 1e-8)))
            out.loc[idx, "O_L"] *= trend
            out.loc[idx, "O_R"] *= trend

    if bool(stress_cfg.get("temporal_warp", False)):
        warp_jitter = float(stress_cfg.get("warp_jitter", 0.35))
        for _, idx in out.groupby("case_id").groups.items():
            n = len(idx)
            if n <= 1:
                continue
            increments = np.maximum(1e-3, 1.0 + rng.normal(0.0, warp_jitter, size=n))
            warped = np.cumsum(increments)
            out.loc[idx, "t"] = warped

    if bool(stress_cfg.get("one_over_f_noise", False)):
        alpha = float(stress_cfg.get("one_over_f_alpha", 1.0))
        scale = float(stress_cfg.get("one_over_f_scale", 0.12))
        for _, idx in out.sort_values(["case_id", "t", "L"]).groupby("case_id").groups.items():
            n = len(idx)
            if n <= 3:
                continue
            colored = _colored_noise(n=n, alpha=alpha, rng=rng)
            out.loc[idx, "O_L"] *= np.maximum(1e-3, 1.0 + scale * colored)
            out.loc[idx, "O_R"] *= np.maximum(1e-3, 1.0 + scale * colored)

    if bool(stress_cfg.get("heavy_tail", False)):
        scale = float(stress_cfg.get("heavy_tail_scale", 0.08))
        dof = float(stress_cfg.get("heavy_tail_df", 3.0))
        noise = rng.standard_t(max(dof, 1.1), size=len(out)) * scale
        out["O_L"] *= np.maximum(1e-3, 1.0 + noise)
        out["O_R"] *= np.maximum(1e-3, 1.0 + noise)

    if bool(stress_cfg.get("confound_even", False)):
        strength = float(stress_cfg.get("confound_strength", 0.6))
        center = float(stress_cfg.get("confound_center_L", np.nanmedian(out["L"].to_numpy(dtype=float))))
        width = float(stress_cfg.get("confound_width_L", max(1.0, np.nanstd(out["L"].to_numpy(dtype=float)))))
        z = (out["L"].to_numpy(dtype=float) - center) / max(width, 1e-6)
        event = 1.0 / (1.0 + np.exp(-z))
        even_term = np.maximum(0.0, strength * event)
        out["O_L"] *= (1.0 + even_term)
        out["O_R"] *= (1.0 + even_term)

    if bool(stress_cfg.get("spatial_aliasing", False)):
        jitter = float(stress_cfg.get("alias_jitter", 0.1))
        blur = float(stress_cfg.get("alias_blur", 0.05))
        scale = np.maximum(1e-3, 1.0 + rng.normal(0.0, jitter, size=len(out)))
        out["O_L"] *= scale
        out["O_R"] *= scale
        out["O_L"] = out["O_L"].rolling(3, min_periods=1, center=True).mean() * (1.0 - blur) + out["O_L"] * blur
        out["O_R"] = out["O_R"].rolling(3, min_periods=1, center=True).mean() * (1.0 - blur) + out["O_R"] * blur

    if bool(stress_cfg.get("missing_blocks", False)):
        block_frac = float(stress_cfg.get("missing_block_frac", 0.08))
        random_frac = float(stress_cfg.get("missing_random_frac", 0.03))
        keep_mask = np.ones(len(out), dtype=bool)
        # Drop contiguous blocks in each case after sorting by t.
        for _, idx in out.sort_values(["case_id", "t", "L"]).groupby("case_id").groups.items():
            n = len(idx)
            if n < 8:
                continue
            block = int(max(1, round(n * block_frac)))
            if block >= n:
                continue
            start = int(rng.integers(0, max(1, n - block)))
            drop_idx = np.asarray(idx)[start : start + block]
            keep_mask[drop_idx] = False
        # Additional random drop.
        if random_frac > 0:
            rand_drop = rng.random(len(out)) < random_frac
            keep_mask &= ~rand_drop
        out = out.loc[keep_mask].copy()

    if bool(stress_cfg.get("duplicate_timestamps", False)):
        frac = float(stress_cfg.get("duplicate_timestamp_frac", 0.15))
        if frac > 0:
            quant = max(1, int(round(1.0 / max(frac, 1e-6))))
            t_vals = pd.to_numeric(out["t"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out["t"] = np.floor(t_vals / quant) * quant

    if bool(stress_cfg.get("sign_flip_by_scale", False)):
        flip_prob = float(stress_cfg.get("sign_flip_prob_scale", 0.5))
        for _, idx in out.groupby("L").groups.items():
            if rng.random() < flip_prob:
                tmp = out.loc[idx, "O_L"].to_numpy(copy=True)
                out.loc[idx, "O_L"] = out.loc[idx, "O_R"].to_numpy()
                out.loc[idx, "O_R"] = tmp

    if bool(stress_cfg.get("sparse_random_peaks", False)):
        peak_prob = float(stress_cfg.get("sparse_peak_prob", 0.05))
        peak_scale = float(stress_cfg.get("sparse_peak_scale", 0.8))
        peak_width = max(float(stress_cfg.get("sparse_peak_width", 1.0)), 1e-6)
        mask = rng.random(len(out)) < peak_prob
        if np.any(mask):
            peak_gain = np.exp(rng.normal(loc=np.log1p(max(peak_scale, 0.0)), scale=peak_width, size=int(mask.sum())))
            side = rng.random(int(mask.sum())) < 0.5
            masked_index = np.flatnonzero(mask)
            left_idx = masked_index[side]
            right_idx = masked_index[~side]
            if left_idx.size:
                out.loc[left_idx, "O_L"] *= peak_gain[side]
            if right_idx.size:
                out.loc[right_idx, "O_R"] *= peak_gain[~side]

    out["O_L"] = np.maximum(out["O_L"].to_numpy(dtype=float), 1e-8)
    out["O_R"] = np.maximum(out["O_R"].to_numpy(dtype=float), 1e-8)
    return out.reset_index(drop=True)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _colored_noise(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)
    alpha = max(alpha, 0.0)
    freqs = np.fft.rfftfreq(n)
    amp = np.ones_like(freqs)
    nonzero = freqs > 0
    amp[nonzero] = np.power(freqs[nonzero], -alpha / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape[0])
    spectrum = amp * (np.cos(phase) + 1j * np.sin(phase))
    noise = np.fft.irfft(spectrum, n=n)
    noise = noise - np.mean(noise)
    std = float(np.std(noise, ddof=0))
    if std <= 1e-12:
        return np.zeros(n, dtype=float)
    return noise / std


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
