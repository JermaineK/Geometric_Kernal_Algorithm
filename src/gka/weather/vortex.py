from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from gka.weather.vortex_detect import (
    VortexDetectConfig,
    compute_okubo_weiss,
    compute_relative_vorticity,
    detect_vortex_candidates,
)
from gka.weather.vortex_track import (
    VortexTrackConfig,
    select_track_centers,
    summarize_tracks,
    track_vortex_candidates,
)


@dataclass(frozen=True)
class VortexDiscoveryConfig:
    detect: VortexDetectConfig = VortexDetectConfig()
    track: VortexTrackConfig = VortexTrackConfig()
    max_fragments_per_lead: int = 0
    scan_batch_rows: int = 200_000
    min_track_duration_hours: float = 6.0
    min_track_points: int = 4


def discover_vortex_tracks(
    *,
    prepared_root: str | Path,
    lead_buckets: list[str],
    config: VortexDiscoveryConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run streamed vortex candidate detection + track association on prepared weather parquet."""

    cfg = config or VortexDiscoveryConfig()
    root = Path(prepared_root)
    if not root.exists():
        raise FileNotFoundError(f"Prepared root not found: {root}")

    dataset = ds.dataset(root, format="parquet", partitioning="hive", exclude_invalid_files=True)
    available = set(dataset.schema.names)
    base_cols = ["time", "lat", "lon", "u10", "v10", "lead_partition"]
    mslp_col = "mslp" if "mslp" in available else None
    if mslp_col:
        base_cols.append(mslp_col)
    missing = [c for c in ("time", "lat", "lon", "u10", "v10", "lead_partition") if c not in available]
    if missing:
        raise ValueError(f"Prepared dataset missing required columns for vortex discovery: {missing}")

    all_candidates: list[pd.DataFrame] = []
    all_tracks: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for lead in [str(v) for v in lead_buckets]:
        filt = ds.field("lead") == str(lead)
        scanner = dataset.scanner(columns=base_cols, filter=filt, batch_size=max(int(cfg.scan_batch_rows), 1))
        lead_cands: list[pd.DataFrame] = []
        batch_idx = 0
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            batch_idx += 1
            if int(cfg.max_fragments_per_lead) > 0 and batch_idx > int(cfg.max_fragments_per_lead):
                break
            cand = detect_vortex_candidates(
                frame,
                mslp_col=mslp_col,
                lead_bucket=str(lead),
                config=cfg.detect,
            )
            if not cand.empty:
                lead_cands.append(cand)

        if not lead_cands:
            continue
        cand_df = pd.concat(lead_cands, ignore_index=True)
        tracks = track_vortex_candidates(cand_df, config=cfg.track)
        if not tracks.empty:
            tracks["lead_bucket"] = str(lead)
            summary = summarize_tracks(
                tracks,
                min_duration_hours=float(cfg.min_track_duration_hours),
                min_points=int(cfg.min_track_points),
            )
            all_tracks.append(tracks)
            all_summary.append(summary)
        all_candidates.append(cand_df)

    cand_out = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    track_out = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    summary_out = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    return cand_out, track_out, summary_out


def save_vortex_discovery_artifacts(
    *,
    prepared_root: str | Path,
    lead_buckets: list[str],
    out_candidates: str | Path,
    out_tracks: str | Path,
    out_summary: str | Path,
    config: VortexDiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Run discovery and persist candidate/track artifacts as parquet."""

    candidates, tracks, summary = discover_vortex_tracks(
        prepared_root=prepared_root,
        lead_buckets=lead_buckets,
        config=config,
    )
    p_c = Path(out_candidates)
    p_t = Path(out_tracks)
    p_s = Path(out_summary)
    p_c.parent.mkdir(parents=True, exist_ok=True)
    p_t.parent.mkdir(parents=True, exist_ok=True)
    p_s.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(p_c, index=False)
    tracks.to_parquet(p_t, index=False)
    summary.to_parquet(p_s, index=False)
    return {
        "candidates_path": str(p_c.resolve()),
        "tracks_path": str(p_t.resolve()),
        "summary_path": str(p_s.resolve()),
        "n_candidates": int(candidates.shape[0]),
        "n_track_points": int(tracks.shape[0]),
        "n_tracks": int(tracks["track_id"].nunique()) if not tracks.empty and "track_id" in tracks.columns else 0,
    }


__all__ = [
    "VortexDiscoveryConfig",
    "VortexDetectConfig",
    "VortexTrackConfig",
    "compute_relative_vorticity",
    "compute_okubo_weiss",
    "detect_vortex_candidates",
    "track_vortex_candidates",
    "summarize_tracks",
    "select_track_centers",
    "discover_vortex_tracks",
    "save_vortex_discovery_artifacts",
]
