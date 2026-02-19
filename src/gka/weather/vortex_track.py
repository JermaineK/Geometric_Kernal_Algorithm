from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class VortexTrackConfig:
    max_speed_kmh: float = 300.0
    max_gap_hours: float = 3.0
    intensity_weight: float = 0.25


def track_vortex_candidates(
    candidates: pd.DataFrame,
    *,
    config: VortexTrackConfig | None = None,
) -> pd.DataFrame:
    """Associate per-time vortex candidates into tracks."""

    cfg = config or VortexTrackConfig()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "time",
                "lat0",
                "lon0",
                "zeta_peak",
                "ow_median",
                "score",
                "lead_bucket",
                "candidate_rank",
                "point_index",
                "dt_hours",
                "step_distance_km",
                "step_speed_kmh",
            ]
        )

    df = candidates.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "candidate_rank" not in df.columns:
        df["candidate_rank"] = 0
    df = df.dropna(subset=["time", "lat0", "lon0"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["time", "candidate_rank"]).reset_index(drop=True)

    out_rows: list[dict[str, Any]] = []
    next_track_id = 1
    active: dict[int, dict[str, Any]] = {}

    for t_val, grp in df.groupby("time", sort=True, dropna=False):
        cur = grp.reset_index(drop=True)
        if not active:
            for j in range(cur.shape[0]):
                tid = next_track_id
                next_track_id += 1
                rec = _emit_track_row(cur.iloc[j], tid=tid, point_index=1, dt_hours=None, distance_km=None)
                out_rows.append(rec)
                active[tid] = _state_from_row(rec)
            continue

        # Drop stale tracks before attempting assignment.
        stale: list[int] = []
        for tid, state in active.items():
            dt_h = float((pd.Timestamp(t_val) - pd.Timestamp(state["time"])).total_seconds() / 3600.0)
            if dt_h > float(cfg.max_gap_hours):
                stale.append(tid)
        for tid in stale:
            active.pop(tid, None)
        if not active:
            for j in range(cur.shape[0]):
                tid = next_track_id
                next_track_id += 1
                rec = _emit_track_row(cur.iloc[j], tid=tid, point_index=1, dt_hours=None, distance_km=None)
                out_rows.append(rec)
                active[tid] = _state_from_row(rec)
            continue

        track_ids = list(active.keys())
        cost = np.full((len(track_ids), cur.shape[0]), np.inf, dtype=float)
        dt_matrix = np.full_like(cost, np.nan, dtype=float)
        dist_matrix = np.full_like(cost, np.nan, dtype=float)

        for i, tid in enumerate(track_ids):
            st = active[tid]
            for j in range(cur.shape[0]):
                row = cur.iloc[j]
                dt_h = float((pd.Timestamp(row["time"]) - pd.Timestamp(st["time"])).total_seconds() / 3600.0)
                if dt_h <= 0:
                    continue
                max_dist = float(cfg.max_speed_kmh) * dt_h
                dist = haversine_km(
                    float(st["lat0"]),
                    float(st["lon0"]),
                    float(row["lat0"]),
                    float(row["lon0"]),
                )
                if dist > max_dist:
                    continue
                prev_int = float(np.log1p(abs(float(st.get("zeta_peak", 0.0)))))
                cur_int = float(np.log1p(abs(float(row.get("zeta_peak", 0.0)))))
                intensity_penalty = float(cfg.intensity_weight) * abs(cur_int - prev_int) * 25.0
                cost[i, j] = dist + intensity_penalty
                dt_matrix[i, j] = dt_h
                dist_matrix[i, j] = dist

        assigned_tracks: set[int] = set()
        assigned_rows: set[int] = set()
        if np.any(np.isfinite(cost)):
            r_idx, c_idx = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e12))
            for i, j in zip(r_idx.tolist(), c_idx.tolist()):
                if not np.isfinite(cost[i, j]):
                    continue
                tid = track_ids[int(i)]
                row = cur.iloc[int(j)]
                prev_idx = int(active[tid]["point_index"])
                rec = _emit_track_row(
                    row,
                    tid=tid,
                    point_index=prev_idx + 1,
                    dt_hours=float(dt_matrix[i, j]),
                    distance_km=float(dist_matrix[i, j]),
                )
                out_rows.append(rec)
                active[tid] = _state_from_row(rec)
                assigned_tracks.add(tid)
                assigned_rows.add(int(j))

        # Unassigned detections start new tracks.
        for j in range(cur.shape[0]):
            if j in assigned_rows:
                continue
            tid = next_track_id
            next_track_id += 1
            rec = _emit_track_row(cur.iloc[j], tid=tid, point_index=1, dt_hours=None, distance_km=None)
            out_rows.append(rec)
            active[tid] = _state_from_row(rec)
            assigned_tracks.add(tid)

        # Unassigned active tracks remain alive only if within gap budget.
        for tid in list(active.keys()):
            if tid in assigned_tracks:
                continue
            dt_h = float((pd.Timestamp(t_val) - pd.Timestamp(active[tid]["time"])).total_seconds() / 3600.0)
            if dt_h > float(cfg.max_gap_hours):
                active.pop(tid, None)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out = out.sort_values(["track_id", "time"]).reset_index(drop=True)
    return out


def summarize_tracks(
    tracks: pd.DataFrame,
    *,
    min_duration_hours: float = 6.0,
    min_points: int = 4,
) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "lead_bucket",
                "n_points",
                "duration_hours",
                "max_score",
                "max_abs_zeta",
                "mean_ow_median",
                "path_length_km",
                "mean_speed_kmh",
                "quality_score",
                "passes_minimums",
            ]
        )

    df = tracks.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    summary = (
        df.groupby(["track_id", "lead_bucket"], dropna=False)
        .agg(
            n_points=("time", "size"),
            t_min=("time", "min"),
            t_max=("time", "max"),
            max_score=("score", "max"),
            max_abs_zeta=("zeta_peak", lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce"))))),
            mean_ow_median=("ow_median", "mean"),
            path_length_km=("step_distance_km", lambda s: float(np.nansum(pd.to_numeric(s, errors="coerce")))),
        )
        .reset_index()
    )
    summary["duration_hours"] = (
        pd.to_datetime(summary["t_max"], errors="coerce") - pd.to_datetime(summary["t_min"], errors="coerce")
    ).dt.total_seconds() / 3600.0
    summary["mean_speed_kmh"] = summary["path_length_km"] / np.maximum(summary["duration_hours"], 1e-6)
    summary["quality_score"] = (
        pd.to_numeric(summary["max_score"], errors="coerce").fillna(0.0)
        + np.log1p(pd.to_numeric(summary["n_points"], errors="coerce").fillna(0.0))
        + (pd.to_numeric(summary["duration_hours"], errors="coerce").fillna(0.0) / 24.0)
    )
    summary["passes_minimums"] = (
        (pd.to_numeric(summary["n_points"], errors="coerce").fillna(0) >= int(min_points))
        & (pd.to_numeric(summary["duration_hours"], errors="coerce").fillna(0.0) >= float(min_duration_hours))
    )
    return summary.sort_values("quality_score", ascending=False).reset_index(drop=True)


def select_track_centers(
    tracks: pd.DataFrame,
    track_summary: pd.DataFrame,
    *,
    max_events: int,
    lead_bucket: str,
) -> list[dict[str, Any]]:
    if tracks.empty or track_summary.empty:
        return []
    ok_tracks = track_summary.loc[pd.to_numeric(track_summary["passes_minimums"], errors="coerce").fillna(0).astype(bool)]
    if ok_tracks.empty:
        ok_tracks = track_summary.copy()
    chosen_tracks = ok_tracks.sort_values("quality_score", ascending=False)["track_id"].astype(int).tolist()
    out: list[dict[str, Any]] = []
    idx = 0
    for tid in chosen_tracks:
        grp = tracks.loc[pd.to_numeric(tracks["track_id"], errors="coerce").fillna(-1).astype(int) == int(tid)].copy()
        if grp.empty:
            continue
        grp["score"] = pd.to_numeric(grp["score"], errors="coerce")
        grp = grp.sort_values(["score", "time"], ascending=[False, True]).reset_index(drop=True)
        row = grp.iloc[0]
        out.append(
            {
                "case_id": f"storm_lead{lead_bucket}_{idx:04d}",
                "case_type": "storm",
                "event_label": "vortex_track",
                "storm_id": f"track_{lead_bucket}_{int(tid)}",
                "track_id": int(tid),
                "lead_bucket": str(lead_bucket),
                "time0": pd.Timestamp(row["time"]),
                "lat0": float(row["lat0"]),
                "lon0": float(row["lon0"]),
                "track_quality_score": float(row.get("score", np.nan)),
                "zeta_bin": float(np.sign(float(row.get("zeta_peak", 0.0))) * np.floor(abs(float(row.get("zeta_peak", 0.0))) / 1.0e-5) * 1.0e-5),
                "speed_bin": None,
            }
        )
        idx += 1
        if idx >= int(max_events):
            break
    return out


def _emit_track_row(
    row: pd.Series,
    *,
    tid: int,
    point_index: int,
    dt_hours: float | None,
    distance_km: float | None,
) -> dict[str, Any]:
    spd = None
    if dt_hours is not None and distance_km is not None and dt_hours > 1e-9:
        spd = float(distance_km / dt_hours)
    return {
        "track_id": int(tid),
        "time": pd.Timestamp(row["time"]),
        "lat0": float(row["lat0"]),
        "lon0": float(row["lon0"]),
        "zeta_peak": float(row.get("zeta_peak", np.nan)),
        "ow_median": float(row.get("ow_median", np.nan)),
        "score": float(row.get("score", np.nan)),
        "lead_bucket": str(row.get("lead_bucket")) if row.get("lead_bucket") is not None else None,
        "candidate_rank": int(row.get("candidate_rank", 0)),
        "point_index": int(point_index),
        "dt_hours": dt_hours,
        "step_distance_km": distance_km,
        "step_speed_kmh": spd,
    }


def _state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": pd.Timestamp(row["time"]),
        "lat0": float(row["lat0"]),
        "lon0": float(row["lon0"]),
        "zeta_peak": float(row.get("zeta_peak", np.nan)),
        "point_index": int(row.get("point_index", 1)),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371.0
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(float(lat2))
    dp = p2 - p1
    dl = np.deg2rad(float(lon2) - float(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 0.0)))
    return float(r * c)
