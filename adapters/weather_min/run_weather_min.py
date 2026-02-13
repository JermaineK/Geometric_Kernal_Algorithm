from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gka.ops.diagnostics import compute_stability_diagnostics
from gka.ops.knee import detect_knee
from gka.ops.scaling import fit_scaling
from gka.ops.spectrum import extract_ticks

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise ImportError(
        "weather_min requires xarray. Install with `pip install -e .[weather]`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal weather adapter for single-tile GKA diagnostics")
    parser.add_argument("--in", dest="input_path", required=True, help="Input NetCDF/Zarr dataset path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--scales", nargs="*", type=int, default=[3, 5, 9, 15, 21], help="Window scales used as L surrogates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _first_time_dim(ds: xr.Dataset) -> str:
    for dim in ds.dims:
        if "time" in dim.lower():
            return dim
    return next(iter(ds.dims))


def _series_from_dataset(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    # Return (base amplitude proxy, odd-channel proxy)
    vars_lower = {v.lower(): v for v in ds.data_vars}

    if "u10" in vars_lower and "v10" in vars_lower:
        u = ds[vars_lower["u10"]]
        v = ds[vars_lower["v10"]]
        time_dim = _first_time_dim(ds)
        spatial_dims = [d for d in u.dims if d != time_dim]
        u_mean = u.mean(dim=spatial_dims).to_numpy(dtype=float)
        v_mean = v.mean(dim=spatial_dims).to_numpy(dtype=float)
        amp = np.sqrt(np.square(u_mean) + np.square(v_mean))
        odd = np.gradient(v_mean) - np.gradient(u_mean)
        return amp, odd

    if "vorticity" in vars_lower:
        vort = ds[vars_lower["vorticity"]]
        time_dim = _first_time_dim(ds)
        spatial_dims = [d for d in vort.dims if d != time_dim]
        vort_mean = vort.mean(dim=spatial_dims).to_numpy(dtype=float)
        return np.abs(vort_mean), vort_mean

    # Fallback: first variable reduced over non-time dims.
    var_name = next(iter(ds.data_vars))
    da = ds[var_name]
    time_dim = _first_time_dim(ds)
    spatial_dims = [d for d in da.dims if d != time_dim]
    s = da.mean(dim=spatial_dims).to_numpy(dtype=float)
    return np.abs(s), np.gradient(s)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window=window, center=True, min_periods=window).mean().to_numpy(dtype=float)


def _build_pairs(
    amp: np.ndarray,
    odd: np.ndarray,
    scales: Iterable[int],
) -> pd.DataFrame:
    odd_std = float(np.nanstd(odd))
    odd_norm = odd / max(odd_std, 1e-8)

    rows: list[dict[str, float | int | str]] = []
    for scale in scales:
        if scale < 3:
            continue
        a_sm = _rolling_mean(amp, int(scale))
        o_sm = _rolling_mean(odd_norm, int(scale))
        valid = np.isfinite(a_sm) & np.isfinite(o_sm)
        if valid.sum() < 12:
            continue
        idx = np.where(valid)[0]
        for t_idx in idx:
            base = max(float(a_sm[t_idx]), 1e-8)
            odd_term = float(0.35 * o_sm[t_idx])
            o_l = max(base * (1.0 + odd_term), 1e-8)
            o_r = max(base * (1.0 - odd_term), 1e-8)
            denom = max((o_l + o_r) / 2.0, 1e-12)
            rows.append(
                {
                    "case_id": "tile_000",
                    "t": int(t_idx),
                    "L": float(scale),
                    "O_L": o_l,
                    "O_R": o_r,
                    "eta": abs(o_l - o_r) / denom,
                }
            )
    return pd.DataFrame(rows)


def run_weather_min(input_path: Path, out_path: Path, scales: list[int], seed: int) -> dict[str, object]:
    _ = np.random.default_rng(seed)
    ds = xr.open_dataset(input_path)
    amp, odd = _series_from_dataset(ds)
    pairs = _build_pairs(amp=amp, odd=odd, scales=scales)
    if pairs.empty:
        raise ValueError("No valid multi-scale pairs could be built from the dataset")

    per_size = (
        pairs.groupby("L", as_index=False)
        .agg(eta=("eta", "median"), n=("eta", "size"))
        .sort_values("L")
        .reset_index(drop=True)
    )

    ticks = extract_ticks(odd, method="welch")
    knee = detect_knee(
        eta_series=per_size["eta"].to_numpy(dtype=float),
        L_series=per_size["L"].to_numpy(dtype=float),
        method="segmented",
        rho=1.5,
        min_points=max(4, min(6, len(per_size))),
    )

    exclude_band = None
    if knee.forbidden_band[0] is not None and knee.forbidden_band[1] is not None:
        exclude_band = (float(knee.forbidden_band[0]), float(knee.forbidden_band[1]))

    scaling = fit_scaling(
        eta=per_size["eta"].to_numpy(dtype=float),
        L=per_size["L"].to_numpy(dtype=float),
        exclude_band=exclude_band,
        weights=per_size["n"].to_numpy(dtype=float),
        method="theil_sen",
        min_sizes=max(3, min(4, len(per_size))),
        bootstrap_n=400,
    )

    omega_hat = float(ticks.Omega_candidates[0]) if ticks.Omega_candidates.size > 0 else 1.0
    tau_s = 1.0 / max(omega_hat, 1e-12)
    sdiag = compute_stability_diagnostics(
        gamma=scaling.gamma,
        b=2.0,
        mu_k_hat=omega_hat,
        tau_s_hat=tau_s,
        mu_grid=ticks.Omega_candidates if ticks.Omega_candidates.size > 0 else None,
        eps_log=0.15,
        L_values=per_size["L"].to_numpy(dtype=float),
        c_m_hat=1.0,
    )

    payload: dict[str, object] = {
        "adapter": "weather_min",
        "input_path": str(input_path),
        "n_points": int(len(amp)),
        "n_pairs": int(len(pairs)),
        "scales": [int(s) for s in scales],
        "Omega_hat": omega_hat,
        "knee_L_hat": knee.L_k,
        "knee_confidence": knee.confidence,
        "knee_rejected_because": knee.rejection_reasons,
        "gamma_hat": scaling.gamma,
        "Delta_b": scaling.Delta_hat,
        "gamma_ci": [float(scaling.ci[0]), float(scaling.ci[1])],
        "tau_s_hat": sdiag.tau_s_hat,
        "S_at_mu_k": sdiag.S_at_mu_k,
        "W_mu": sdiag.W_mu,
        "band_label": sdiag.band_class_hat,
        "ridge_strength": float(ticks.ridge_strength),
        "R_Omega": float(ticks.R_Omega),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    payload = run_weather_min(
        input_path=Path(args.input_path),
        out_path=Path(args.out),
        scales=[int(v) for v in args.scales],
        seed=int(args.seed),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
