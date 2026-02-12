"""Knee detection over eta(L) series with explicit no-knee selection."""

from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes

from gka.core.types import KneeOutputs
from gka.utils.safe_math import safe_log


def detect_knee(
    eta_series: np.ndarray,
    L_series: np.ndarray,
    method: str,
    rho: float = 1.5,
    min_points: int = 6,
    bic_delta_min: float = 10.0,
    bootstrap_n: int = 200,
    bootstrap_cov_max: float = 0.25,
    edge_buffer_frac: float = 0.15,
    min_points_each_side: int = 3,
    min_frac_each_side: float = 0.2,
    slope_abs_max: float = 6.0,
    instability_window: int = 7,
    instability_zscore: float = 2.0,
    rng: np.random.Generator | None = None,
) -> KneeOutputs:
    eta = np.asarray(eta_series, dtype=float)
    L = np.asarray(L_series, dtype=float)
    if eta.size != L.size:
        raise ValueError("eta_series and L_series must have equal length")
    if eta.size < min_points:
        raise ValueError(f"Knee detection requires at least {min_points} points")
    if method not in {"segmented", "log_curvature", "cpt"}:
        raise ValueError(f"Unsupported knee detection method '{method}'. Use segmented|log_curvature|cpt")

    order = np.argsort(L)
    Ls = L[order]
    ys = np.maximum(np.abs(eta[order]), 1e-12)
    x = safe_log(Ls)
    y = safe_log(ys)
    n = x.size

    rss0 = _linear_rss(x, y)
    bic0 = _bic(rss0, n=n, k=2)

    rejection_reasons: list[str] = []
    best_idx, rss1, slope_left, slope_right = _best_piecewise_break(
        x=x,
        y=y,
        method=method,
        min_points_each_side=min_points_each_side,
        min_frac_each_side=min_frac_each_side,
    )
    if best_idx is None:
        rejection_reasons.append("insufficient_side_points")
        return _no_knee_output(
            bic0=bic0,
            bic1=np.inf,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            rejection_reasons=rejection_reasons,
        )

    bic1 = _bic(rss1, n=n, k=3)
    delta_bic = float(bic0 - bic1)
    if delta_bic < bic_delta_min:
        rejection_reasons.append("bic_weak")
        return _no_knee_output(
            bic0=bic0,
            bic1=bic1,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            rejection_reasons=rejection_reasons,
            slope_left=slope_left,
            slope_right=slope_right,
        )

    if slope_left is None or slope_right is None:
        rejection_reasons.append("slope_fit_failed")
        return _no_knee_output(
            bic0=bic0,
            bic1=bic1,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            rejection_reasons=rejection_reasons,
        )

    if not (np.isfinite(slope_left) and np.isfinite(slope_right)):
        rejection_reasons.append("slope_non_finite")
    elif abs(slope_left) > slope_abs_max or abs(slope_right) > slope_abs_max:
        rejection_reasons.append("slope_unphysical")
    elif abs(slope_left - slope_right) < 0.05:
        rejection_reasons.append("slope_contrast_weak")

    if rejection_reasons:
        return _no_knee_output(
            bic0=bic0,
            bic1=bic1,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            rejection_reasons=rejection_reasons,
            slope_left=slope_left,
            slope_right=slope_right,
            delta_bic=delta_bic,
        )

    rng = rng or np.random.default_rng(0)
    boot_knees = _bootstrap_knees(
        L=Ls,
        eta=ys,
        method=method,
        n_boot=bootstrap_n,
        min_points_each_side=min_points_each_side,
        min_frac_each_side=min_frac_each_side,
        rng=rng,
    )

    if boot_knees.size == 0:
        rejection_reasons.append("bootstrap_empty")
        return _no_knee_output(
            bic0=bic0,
            bic1=bic1,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            rejection_reasons=rejection_reasons,
            slope_left=slope_left,
            slope_right=slope_right,
            delta_bic=delta_bic,
        )

    median_knee = float(np.median(boot_knees))
    knee_cov = float(np.std(boot_knees, ddof=0) / np.maximum(np.abs(median_knee), 1e-12))
    L_min, L_max = float(np.min(Ls)), float(np.max(Ls))
    edge = edge_buffer_frac * (L_max - L_min)
    edge_ok = (median_knee > L_min + edge) and (median_knee < L_max - edge)
    stable = bool(knee_cov < bootstrap_cov_max and edge_ok)
    if not stable:
        if knee_cov >= bootstrap_cov_max:
            rejection_reasons.append("bootstrap_unstable")
        if not edge_ok:
            rejection_reasons.append("knee_at_edge")
        return _no_knee_output(
            bic0=bic0,
            bic1=bic1,
            forbidden=_detect_forbidden_band(
                Ls,
                ys,
                window=instability_window,
                zscore=instability_zscore,
            ),
            bootstrap_cov=knee_cov,
            bootstrap_count=int(boot_knees.size),
            delta_bic=delta_bic,
            rejection_reasons=rejection_reasons,
            slope_left=slope_left,
            slope_right=slope_right,
        )

    knee_lo = float(np.quantile(boot_knees, 0.25))
    knee_hi = float(np.quantile(boot_knees, 0.75))
    forbidden = _detect_forbidden_band(
        Ls,
        ys,
        window=instability_window,
        zscore=instability_zscore,
    )
    forbidden_detected = forbidden is not None
    if forbidden is None:
        forbidden = (float(median_knee / rho), float(median_knee * rho))
    bic_strength = float(np.clip((delta_bic - bic_delta_min) / max(10.0, bic_delta_min), 0.0, 1.0))
    stability_strength = float(np.clip(1.0 - knee_cov / bootstrap_cov_max, 0.0, 1.0))
    confidence = float(np.clip(0.6 * bic_strength + 0.4 * stability_strength, 0.0, 1.0))

    return KneeOutputs(
        L_k=median_knee,
        knee_window=(knee_lo, knee_hi),
        forbidden_band=forbidden,
        confidence=confidence,
        has_knee=True,
        delta_bic=delta_bic,
        bic_no_knee=float(bic0),
        bic_knee=float(bic1),
        bootstrap_cov=knee_cov,
        bootstrap_count=int(boot_knees.size),
        forbidden_detected=forbidden_detected,
        slope_left=float(slope_left),
        slope_right=float(slope_right),
        rejection_reasons=[],
    )


def _linear_rss(x: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return float(np.sum((y - pred) ** 2))


def _piecewise_fit(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float, float]:
    xk = x[idx]
    d = x - xk
    X = np.column_stack([np.ones_like(x), np.minimum(d, 0.0), np.maximum(d, 0.0)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    return rss, float(beta[1]), float(beta[2])


def _best_piecewise_break(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    min_points_each_side: int,
    min_frac_each_side: float,
) -> tuple[int | None, float, float | None, float | None]:
    n = x.size
    candidates = _candidate_breakpoints(n, min_points_each_side, min_frac_each_side)
    if not candidates:
        return None, float("inf"), None, None

    if method == "segmented":
        chosen = candidates
    elif method == "log_curvature":
        idx0 = _knee_log_curvature(x, y)
        chosen = [idx0] if idx0 in candidates else candidates
    elif method == "cpt":
        idx0 = _knee_cpt(y)
        chosen = [idx0] if idx0 in candidates else candidates
    else:
        return None, float("inf"), None, None

    best_idx: int | None = None
    best_rss = float("inf")
    best_left: float | None = None
    best_right: float | None = None
    for idx in chosen:
        rss, s_left, s_right = _piecewise_fit(x, y, idx)
        if rss < best_rss:
            best_rss = rss
            best_idx = idx
            best_left = s_left
            best_right = s_right
    return best_idx, best_rss, best_left, best_right


def _candidate_breakpoints(
    n: int,
    min_points_each_side: int,
    min_frac_each_side: float,
) -> list[int]:
    out: list[int] = []
    for idx in range(1, n - 1):
        left = idx + 1
        right = n - left
        if left < min_points_each_side or right < min_points_each_side:
            continue
        if left < min_frac_each_side * n or right < min_frac_each_side * n:
            continue
        out.append(idx)
    return out


def _bic(rss: float, n: int, k: int) -> float:
    rss_clamped = max(float(rss), 1e-12)
    return float(n * np.log(rss_clamped / n) + k * np.log(n))


def _bootstrap_knees(
    L: np.ndarray,
    eta: np.ndarray,
    method: str,
    n_boot: int,
    min_points_each_side: int,
    min_frac_each_side: float,
    rng: np.random.Generator,
) -> np.ndarray:
    knees: list[float] = []
    n = L.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Lb = L[idx]
        etab = eta[idx]
        order = np.argsort(Lb)
        Lb = Lb[order]
        yb = safe_log(np.maximum(etab[order], 1e-12))
        xb = safe_log(Lb)

        if np.unique(Lb).size < max(4, min_points_each_side + 1):
            continue

        bidx, _, _, _ = _best_piecewise_break(
            x=xb,
            y=yb,
            method=method,
            min_points_each_side=min_points_each_side,
            min_frac_each_side=min_frac_each_side,
        )
        if bidx is None:
            continue
        knees.append(float(Lb[bidx]))

    return np.asarray(knees, dtype=float)


def _detect_forbidden_band(
    L: np.ndarray,
    eta: np.ndarray,
    window: int,
    zscore: float,
) -> tuple[float, float] | None:
    n = L.size
    if n < 6:
        return None

    w = max(5, int(window))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return None

    x = safe_log(L)
    y = safe_log(np.maximum(eta, 1e-12))
    half = w // 2

    slopes = np.full(n, np.nan, dtype=float)
    residuals = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        if end - start < 4:
            continue
        xw = x[start:end]
        yw = y[start:end]
        slope, intercept, _, _ = theilslopes(yw, xw)
        pred = slope * xw + intercept
        slopes[i] = float(slope)
        residuals[i] = float(np.sqrt(np.mean((yw - pred) ** 2)))

    slope_var = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        vals = slopes[start:end]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            slope_var[i] = float(np.var(vals))

    slope_thr = _robust_threshold(slope_var, zscore)
    resid_thr = _robust_threshold(residuals, zscore)
    if slope_thr is None or resid_thr is None:
        return None

    mask = (slope_var > slope_thr) & (residuals > resid_thr)
    runs = _contiguous_true_runs(mask)
    if not runs:
        return None

    # Use the widest instability region.
    start_idx, end_idx = max(runs, key=lambda r: r[1] - r[0])
    if end_idx - start_idx < 1:
        return None
    return float(L[start_idx]), float(L[end_idx])


def _robust_threshold(values: np.ndarray, zscore: float) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    return med + zscore * max(scale, 1e-12)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _knee_log_curvature(x: np.ndarray, y: np.ndarray) -> int:
    y1 = np.gradient(y, x)
    y2 = np.gradient(y1, x)
    curvature = np.abs(y2) / np.power(1.0 + y1 * y1, 1.5)
    return int(np.nanargmax(curvature))


def _knee_cpt(y: np.ndarray) -> int:
    best_idx = 1
    best_score = -np.inf
    for idx in range(1, y.size - 1):
        left = y[:idx]
        right = y[idx:]
        delta = np.abs(np.mean(right) - np.mean(left))
        pooled = np.std(y) + 1e-12
        score = float(delta / pooled)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _no_knee_output(
    bic0: float,
    bic1: float,
    forbidden: tuple[float, float] | None,
    bootstrap_cov: float | None = None,
    bootstrap_count: int = 0,
    delta_bic: float | None = None,
    rejection_reasons: list[str] | None = None,
    slope_left: float | None = None,
    slope_right: float | None = None,
) -> KneeOutputs:
    forbidden_band: tuple[float | None, float | None]
    forbidden_detected = forbidden is not None
    if forbidden is None:
        forbidden_band = (None, None)
    else:
        forbidden_band = (float(forbidden[0]), float(forbidden[1]))

    return KneeOutputs(
        L_k=None,
        knee_window=(None, None),
        forbidden_band=forbidden_band,
        confidence=0.0,
        has_knee=False,
        delta_bic=float(0.0 if delta_bic is None else delta_bic),
        bic_no_knee=float(bic0),
        bic_knee=float(bic1),
        bootstrap_cov=bootstrap_cov,
        bootstrap_count=int(bootstrap_count),
        forbidden_detected=forbidden_detected,
        slope_left=slope_left,
        slope_right=slope_right,
        rejection_reasons=[] if rejection_reasons is None else list(rejection_reasons),
    )
