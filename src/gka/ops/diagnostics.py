"""Operational diagnostics for stability, bands, and alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityDiagnostics:
    tau_s_hat: float | None
    mu_k_hat: float | None
    S_at_mu_k: float | None
    S_margin_log: float | None
    band_class_hat: str
    W_mu: float | None
    W_L: float | None
    S_curve_mu: np.ndarray
    S_curve_values: np.ndarray


def compute_stability_score(
    gamma: float,
    mu: np.ndarray | float,
    tau_s: float,
    b: float = 2.0,
) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float)
    return (b**gamma) / np.sqrt(1.0 + np.square(mu_arr * tau_s))


def compute_stability_diagnostics(
    gamma: float,
    b: float,
    mu_k_hat: float | None,
    tau_s_hat: float | None,
    mu_grid: np.ndarray | None = None,
    eps_log: float = 0.15,
    L_values: np.ndarray | None = None,
    c_m_hat: float | None = None,
) -> StabilityDiagnostics:
    if mu_k_hat is None or tau_s_hat is None or mu_k_hat <= 0 or tau_s_hat <= 0:
        return StabilityDiagnostics(
            tau_s_hat=tau_s_hat,
            mu_k_hat=mu_k_hat,
            S_at_mu_k=None,
            S_margin_log=None,
            band_class_hat="incoherent",
            W_mu=None,
            W_L=None,
            S_curve_mu=np.array([], dtype=float),
            S_curve_values=np.array([], dtype=float),
        )

    mu0 = float(mu_k_hat)
    tau = float(tau_s_hat)

    if mu_grid is None or np.asarray(mu_grid).size < 5:
        mu_grid = np.geomspace(mu0 / 20.0, mu0 * 20.0, 128)
    else:
        mu_grid = np.asarray(mu_grid, dtype=float)
        mu_grid = mu_grid[np.isfinite(mu_grid) & (mu_grid > 0)]
        if mu_grid.size < 5:
            mu_grid = np.geomspace(mu0 / 20.0, mu0 * 20.0, 128)
        else:
            mu_grid = np.sort(mu_grid)

    S_curve = compute_stability_score(gamma=gamma, mu=mu_grid, tau_s=tau, b=b)
    S_at_mu_k = float(compute_stability_score(gamma=gamma, mu=mu0, tau_s=tau, b=b).reshape(-1)[0])
    margin = float(np.log(max(S_at_mu_k, 1e-12)))

    if margin > eps_log:
        band_class = "coherent"
    elif margin < -eps_log:
        band_class = "incoherent"
    else:
        band_class = "forbidden_middle"

    W_mu = _width_where_logS_near_one(mu_grid, S_curve, eps_log=eps_log)
    W_L = _width_in_L(gamma=gamma, b=b, tau=tau, eps_log=eps_log, L_values=L_values, c_m_hat=c_m_hat)

    return StabilityDiagnostics(
        tau_s_hat=tau,
        mu_k_hat=mu0,
        S_at_mu_k=S_at_mu_k,
        S_margin_log=margin,
        band_class_hat=band_class,
        W_mu=W_mu,
        W_L=W_L,
        S_curve_mu=mu_grid,
        S_curve_values=S_curve,
    )


def predict_bands(L: float, c_m: float, n_max: int) -> np.ndarray:
    if L <= 0:
        raise ValueError("L must be positive for band prediction")
    if c_m <= 0:
        raise ValueError("c_m must be positive for band prediction")
    n = np.arange(1, n_max + 1, dtype=float)
    return n * (2.0 * np.pi * c_m / L)


def band_hit_rate(
    observed_peaks: np.ndarray,
    predicted_peaks: np.ndarray,
    rel_tol: float = 0.15,
) -> float:
    obs = np.asarray(observed_peaks, dtype=float)
    pred = np.asarray(predicted_peaks, dtype=float)
    obs = obs[np.isfinite(obs) & (obs > 0)]
    pred = pred[np.isfinite(pred) & (pred > 0)]
    if obs.size == 0 or pred.size == 0:
        return 0.0
    hits = 0
    for p in pred:
        if np.any(np.abs(obs - p) / p <= rel_tol):
            hits += 1
    return float(hits / pred.size)


def alignment_residual(omega_k: float | None, L: float | None, c_m: float | None) -> tuple[float | None, float | None]:
    if omega_k is None or L is None or c_m is None:
        return None, None
    if omega_k <= 0 or L <= 0 or c_m <= 0:
        return None, None
    omega_ref = 2.0 * np.pi * c_m / L
    ratio = omega_k / omega_ref
    if ratio <= 0:
        return None, None
    r = float(np.log(ratio))
    return r, float(np.abs(r))


def _width_where_logS_near_one(mu_grid: np.ndarray, s_curve: np.ndarray, eps_log: float) -> float:
    mask = np.abs(np.log(np.maximum(s_curve, 1e-12))) <= eps_log
    if not np.any(mask):
        return 0.0
    log_mu = np.log(mu_grid)
    return float(log_mu[mask].max() - log_mu[mask].min())


def _width_in_L(
    gamma: float,
    b: float,
    tau: float,
    eps_log: float,
    L_values: np.ndarray | None,
    c_m_hat: float | None,
) -> float | None:
    if L_values is None or c_m_hat is None:
        return None
    L = np.asarray(L_values, dtype=float)
    L = L[np.isfinite(L) & (L > 0)]
    if L.size < 3:
        return None
    mu_l = 2.0 * np.pi * c_m_hat / L
    s_l = compute_stability_score(gamma=gamma, mu=mu_l, tau_s=tau, b=b)
    mask = np.abs(np.log(np.maximum(s_l, 1e-12))) <= eps_log
    if not np.any(mask):
        return 0.0
    log_l = np.log(np.sort(L))
    mask_sorted = mask[np.argsort(L)]
    return float(log_l[mask_sorted].max() - log_l[mask_sorted].min())
