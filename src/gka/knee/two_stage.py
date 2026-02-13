"""Two-stage knee candidate proposal and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class KneeCandidate:
    idx: int
    L_k: float
    delta_bic: float
    bic_no_knee: float
    bic_knee: float
    delta_gamma: float
    slope_left: float
    slope_right: float
    resid_improvement: float
    curvature_peak_ratio: float
    curvature_alignment: float
    strength: float
    edge_fraction: float
    sanity_ok: bool
    reasons: list[str] = field(default_factory=list)


def propose_candidates(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    min_points_each_side: int,
    min_frac_each_side: float,
    max_candidates: int = 6,
) -> list[int]:
    """Stage-A candidate proposal with high recall."""
    candidates = candidate_breakpoints(
        n=x.size,
        min_points_each_side=min_points_each_side,
        min_frac_each_side=min_frac_each_side,
    )
    if not candidates:
        return []

    proposed: list[int] = []
    if method == "segmented":
        rss_scores = [(idx, _piecewise_fit(x, y, idx)[0]) for idx in candidates]
        rss_scores.sort(key=lambda p: p[1])
        proposed.extend([idx for idx, _ in rss_scores[:max_candidates]])
    elif method == "log_curvature":
        idx = _knee_log_curvature(x, y)
        proposed.append(int(idx))
    elif method == "cpt":
        idx = _knee_cpt(y)
        proposed.append(int(idx))
    else:
        raise ValueError(f"Unsupported knee method '{method}'")

    # Always include classic curvature/changepoint proposals to avoid hard misses.
    proposed.append(_knee_log_curvature(x, y))
    proposed.append(_knee_cpt(y))

    # Keep only valid, unique candidates with deterministic order.
    valid = set(candidates)
    out: list[int] = []
    seen: set[int] = set()
    for idx in proposed:
        i = int(idx)
        if i in valid and i not in seen:
            out.append(i)
            seen.add(i)
    return out


def evaluate_candidates(
    x: np.ndarray,
    y: np.ndarray,
    L: np.ndarray,
    candidate_indices: list[int],
    bic_no_knee: float,
    slope_abs_max: float,
    edge_buffer_frac: float,
    post_slope_max: float,
    resid_improvement_min: float,
    curvature_peak_ratio_min: float,
    curvature_alignment_min: float,
) -> list[KneeCandidate]:
    """Stage-B candidate scoring with multi-criterion effect size."""
    if not candidate_indices:
        return []
    n = x.size
    rss0, resid0 = _linear_fit(x, y)
    ac0 = _lag1_autocorr(np.asarray(resid0, dtype=float))
    curv = _curvature_series(x, y)
    curv_max = float(np.max(curv)) if curv.size else 0.0
    curv_mean = float(np.mean(np.abs(curv)) + 1e-12)

    raw_rows: list[dict[str, float | int | bool | list[str]]] = []
    for idx in candidate_indices:
        rss1, s_left, s_right, resid1 = _piecewise_fit(x, y, idx, return_residuals=True)
        bic1 = _bic(rss1, n=n, k=3)
        delta_bic = float(bic_no_knee - bic1)
        delta_gamma = float(abs(s_right - s_left))
        ac1 = _lag1_autocorr(np.asarray(resid1, dtype=float))
        resid_improvement = float(abs(ac0) - abs(ac1))
        curv_peak_ratio = float(curv_max / curv_mean) if curv_mean > 0 else 0.0
        if 0 <= idx < curv.size and curv_max > 1e-12:
            curv_alignment = float(curv[idx] / curv_max)
        else:
            curv_alignment = 0.0

        l_k = float(L[idx])
        l_min = float(np.min(L))
        l_max = float(np.max(L))
        span = max(l_max - l_min, 1e-12)
        edge_fraction = float(min(abs(l_k - l_min), abs(l_max - l_k)) / span)

        reasons: list[str] = []
        sanity_ok = True
        if not (np.isfinite(s_left) and np.isfinite(s_right)):
            reasons.append("slope_non_finite")
            sanity_ok = False
        elif abs(s_left) > slope_abs_max or abs(s_right) > slope_abs_max:
            reasons.append("slope_unphysical")
            sanity_ok = False
        if float(s_right) > float(post_slope_max):
            reasons.append("post_slope_too_flat")
            sanity_ok = False
        if float(resid_improvement) < float(resid_improvement_min):
            reasons.append("resid_improvement_weak")
            sanity_ok = False
        if float(curv_peak_ratio) < float(curvature_peak_ratio_min):
            reasons.append("curvature_peak_weak")
            sanity_ok = False
        if float(curv_alignment) < float(curvature_alignment_min):
            reasons.append("curvature_alignment_weak")
            sanity_ok = False
        if edge_fraction < edge_buffer_frac:
            reasons.append("knee_at_edge")
            sanity_ok = False

        raw_rows.append(
            {
                "idx": int(idx),
                "L_k": l_k,
                "delta_bic": delta_bic,
                "bic_knee": float(bic1),
                "delta_gamma": delta_gamma,
                "slope_left": float(s_left),
                "slope_right": float(s_right),
                "resid_improvement": resid_improvement,
                "curvature_peak_ratio": curv_peak_ratio,
                "curvature_alignment": curv_alignment,
                "edge_fraction": edge_fraction,
                "sanity_ok": sanity_ok,
                "reasons": reasons,
            }
        )

    if not raw_rows:
        return []

    dbic = np.asarray([float(r["delta_bic"]) for r in raw_rows], dtype=float)
    dgamma = np.asarray([float(r["delta_gamma"]) for r in raw_rows], dtype=float)
    rimpr = np.asarray([float(r["resid_improvement"]) for r in raw_rows], dtype=float)
    z_dbic = _robust_zscore(dbic)
    z_dgamma = _robust_zscore(dgamma)
    z_rimpr = _robust_zscore(rimpr)

    out: list[KneeCandidate] = []
    for i, row in enumerate(raw_rows):
        strength = float(z_dbic[i] + z_dgamma[i] + z_rimpr[i])
        out.append(
            KneeCandidate(
                idx=int(row["idx"]),
                L_k=float(row["L_k"]),
                delta_bic=float(row["delta_bic"]),
                bic_no_knee=float(bic_no_knee),
                bic_knee=float(row["bic_knee"]),
                delta_gamma=float(row["delta_gamma"]),
                slope_left=float(row["slope_left"]),
                slope_right=float(row["slope_right"]),
                resid_improvement=float(row["resid_improvement"]),
                curvature_peak_ratio=float(row["curvature_peak_ratio"]),
                curvature_alignment=float(row["curvature_alignment"]),
                strength=strength,
                edge_fraction=float(row["edge_fraction"]),
                sanity_ok=bool(row["sanity_ok"]),
                reasons=list(row["reasons"]),
            )
        )
    out.sort(key=lambda c: c.strength, reverse=True)
    return out


def candidate_breakpoints(
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


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid**2))
    return rss, resid


def _piecewise_fit(
    x: np.ndarray,
    y: np.ndarray,
    idx: int,
    return_residuals: bool = False,
) -> tuple[float, float, float, np.ndarray] | tuple[float, float, float]:
    xk = x[idx]
    d = x - xk
    X = np.column_stack([np.ones_like(x), np.minimum(d, 0.0), np.maximum(d, 0.0)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid**2))
    if return_residuals:
        return rss, float(beta[1]), float(beta[2]), resid
    return rss, float(beta[1]), float(beta[2])


def _lag1_autocorr(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    v0 = v[:-1] - float(np.mean(v[:-1]))
    v1 = v[1:] - float(np.mean(v[1:]))
    denom = float(np.sqrt(np.sum(v0**2) * np.sum(v1**2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(v0 * v1) / denom)


def _bic(rss: float, n: int, k: int) -> float:
    rss_clamped = max(float(rss), 1e-12)
    return float(n * np.log(rss_clamped / max(n, 1)) + k * np.log(max(n, 2)))


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        std = float(np.std(arr, ddof=0))
        if std <= 1e-12:
            return np.zeros_like(arr)
        return (arr - float(np.mean(arr))) / std
    return (arr - med) / scale


def _knee_log_curvature(x: np.ndarray, y: np.ndarray) -> int:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.size <= 2:
        return 0
    if np.unique(xv).size < xv.size:
        x_grid = np.arange(xv.size, dtype=float)
    else:
        x_grid = xv
    y1 = np.gradient(yv, x_grid)
    y2 = np.gradient(y1, x_grid)
    curvature = np.abs(y2) / np.power(1.0 + y1 * y1, 1.5)
    if not np.any(np.isfinite(curvature)):
        return int(max(1, min(xv.size - 2, xv.size // 2)))
    curv = np.where(np.isfinite(curvature), curvature, -np.inf)
    return int(np.argmax(curv))


def _curvature_series(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.size <= 2:
        return np.zeros_like(xv, dtype=float)
    if np.unique(xv).size < xv.size:
        x_grid = np.arange(xv.size, dtype=float)
    else:
        x_grid = xv
    y1 = np.gradient(yv, x_grid)
    y2 = np.gradient(y1, x_grid)
    curvature = np.abs(y2) / np.power(1.0 + y1 * y1, 1.5)
    curv = np.where(np.isfinite(curvature), curvature, 0.0)
    return np.asarray(curv, dtype=float)


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
