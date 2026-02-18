from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from gka.adapters.weather_era5 import WeatherERA5Adapter
from gka.utils.time import utc_now_iso


DEFAULT_COLUMNS = [
    "row_id",
    "time",
    "lat",
    "lon",
    "ilat",
    "ilon",
    "u10",
    "v10",
    "lead_h",
    "lead_h_bucket",
    "storm_point",
    "storm_window",
    "storm",
    "near_storm",
    "pregen",
    "storm_id",
    "ms_id",
    "tc_id",
    "sid",
    "gka_score",
    "gka_parity_eta",
    "gka_chirality",
    "S",
    "zeta",
    "div",
    "shear_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weather parquet into lead/date partitions")
    parser.add_argument("--input", required=True, help="Input weather parquet")
    parser.add_argument("--out", required=True, help="Output prepared root directory")
    parser.add_argument("--lon0", type=float, default=150.0, help="Longitude mirror center")
    parser.add_argument(
        "--max-lon-distance",
        type=float,
        default=0.13,
        help="Maximum tolerated snapped mirror distance in lon degrees",
    )
    parser.add_argument(
        "--expected-rows-per-time",
        type=int,
        default=24321,
        help="Expected full-grid rows per timestamp (used for row-group streaming flush)",
    )
    parser.add_argument("--date-from", default=None, help="Optional inclusive lower date bound (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Optional inclusive upper date bound (YYYY-MM-DD)")
    parser.add_argument(
        "--lon0-sweep",
        nargs="+",
        type=float,
        default=None,
        help="Optional mirror-axis sweep for parity sensitivity audit (e.g. 145 150 155 160)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed (logged for reproducibility)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Input parquet not found: {src}")

    adapter = WeatherERA5Adapter()
    pf = pq.ParquetFile(src)
    schema_names = [f.name for f in pf.schema_arrow]
    read_cols = [c for c in DEFAULT_COLUMNS if c in schema_names]
    missing_required = [c for c in ["time", "lat", "lon", "u10", "v10"] if c not in read_cols]
    if missing_required:
        raise ValueError(f"Input parquet missing required columns: {missing_required}")

    scalar_cols = [c for c in ["S", "zeta", "div", "shear_proxy", "gka_score", "gka_parity_eta", "gka_chirality"] if c in read_cols]
    mirror_audit = _init_mirror_audit(scalar_cols)
    lon0_sweep = _init_lon0_sensitivity(args.lon0_sweep or [])
    buffers: dict[pd.Timestamp, list[pd.DataFrame]] = defaultdict(list)
    counts: dict[pd.Timestamp, int] = defaultdict(int)
    file_counter = 0
    written_rows = 0
    written_files = 0
    completed_times = 0
    partition_row_counts: dict[str, int] = defaultdict(int)

    date_from = pd.Timestamp(args.date_from) if args.date_from else None
    date_to = pd.Timestamp(args.date_to) if args.date_to else None

    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=read_cols)
        chunk = table.to_pandas()
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time", "lat", "lon", "u10", "v10"])
        if chunk.empty:
            continue
        if date_from is not None:
            chunk = chunk.loc[chunk["time"] >= date_from]
        if date_to is not None:
            chunk = chunk.loc[chunk["time"] <= date_to + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)]
        if chunk.empty:
            continue

        for ts, part in chunk.groupby("time", sort=False):
            ts_key = pd.Timestamp(ts)
            buffers[ts_key].append(part.copy())
            counts[ts_key] += int(part.shape[0])
            if counts[ts_key] >= int(args.expected_rows_per_time):
                frame = pd.concat(buffers.pop(ts_key), ignore_index=True)
                counts.pop(ts_key, None)
                if frame.shape[0] != int(args.expected_rows_per_time):
                    # Keep deterministic behavior: require exact full-time grid.
                    continue
                processed, audit_delta = _process_one_time(
                    frame=frame,
                    adapter=adapter,
                    lon0=float(args.lon0),
                    max_lon_distance=float(args.max_lon_distance),
                    scalar_cols=scalar_cols,
                )
                _accumulate_mirror_audit(mirror_audit, audit_delta)
                if lon0_sweep:
                    _accumulate_lon0_sensitivity(
                        acc=lon0_sweep,
                        frame=frame,
                        adapter=adapter,
                        max_lon_distance=float(args.max_lon_distance),
                        scalar_cols=scalar_cols,
                    )
                if processed.empty:
                    continue
                written_files_delta, written_rows_delta, file_counter, partition_rows = _write_partitioned(
                    df=processed,
                    out_root=out_root,
                    file_counter=file_counter,
                )
                written_files += written_files_delta
                written_rows += written_rows_delta
                completed_times += 1
                for key, value in partition_rows.items():
                    partition_row_counts[key] += int(value)

    meta = {
        "created_at_utc": utc_now_iso(),
        "source_parquet": str(src.resolve()),
        "out_root": str(out_root.resolve()),
        "seed": int(args.seed),
        "num_row_groups": int(pf.num_row_groups),
        "expected_rows_per_time": int(args.expected_rows_per_time),
        "date_from": str(date_from.date()) if date_from is not None else None,
        "date_to": str(date_to.date()) if date_to is not None else None,
        "mirror": {"lon0": float(args.lon0), "max_lon_distance": float(args.max_lon_distance)},
        "read_columns": read_cols,
        "completed_time_slices": int(completed_times),
        "written_files": int(written_files),
        "written_rows": int(written_rows),
        "incomplete_times_dropped": int(len(buffers)),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
    }
    (out_root / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_root / "mirror_audit.json").write_text(
        json.dumps(_finalize_mirror_audit(mirror_audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if lon0_sweep:
        (out_root / "lon0_sensitivity.json").write_text(
            json.dumps(_finalize_lon0_sensitivity(lon0_sweep), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"Prepared weather partitions at {out_root}")
    print(f"Completed time slices: {completed_times}")
    print(f"Written files: {written_files}")
    print(f"Written rows: {written_rows}")
    if buffers:
        print(f"Dropped incomplete trailing time slices: {len(buffers)}")
    return 0


def _process_one_time(
    frame: pd.DataFrame,
    adapter: WeatherERA5Adapter,
    lon0: float,
    max_lon_distance: float,
    scalar_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    in_rows = int(frame.shape[0])
    mirrored = adapter.mirror_lon_about(
        frame,
        lon0=lon0,
        max_lon_distance=max_lon_distance,
        scalar_cols=scalar_cols,
    )
    snap_guard_rows = int(mirrored.shape[0])
    mirror_hits = int(pd.to_numeric(mirrored.get("_mirror_hit"), errors="coerce").fillna(0).astype(bool).sum())
    dist_vals = pd.to_numeric(mirrored.get("_mirror_lon_dist"), errors="coerce").to_numpy(dtype=float)
    dist_vals = dist_vals[np.isfinite(dist_vals)]

    scalar_ks: dict[str, dict[str, float]] = {}
    for col in scalar_cols:
        mirror_col = f"{col}_mirror"
        if col not in mirrored.columns or mirror_col not in mirrored.columns:
            continue
        left = pd.to_numeric(mirrored[col], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(mirrored[mirror_col], errors="coerce").to_numpy(dtype=float)
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size < 8 or right.size < 8:
            continue
        ks = ks_2samp(left, right, method="asymp")
        scalar_ks[col] = {
            "n_left": float(left.size),
            "n_right": float(right.size),
            "ks_stat": float(ks.statistic),
            "pvalue": float(ks.pvalue),
        }

    with_parity = adapter.add_parity_channels(mirrored)
    with_parity["lead_partition"] = with_parity["lead_h_bucket"].apply(_lead_partition_value)
    with_parity["date"] = pd.to_datetime(with_parity["time"]).dt.strftime("%Y-%m-%d")

    keep_cols = [
        "row_id",
        "time",
        "date",
        "lat",
        "lon",
        "ilat",
        "ilon",
        "lead_h",
        "lead_h_bucket",
        "lead_partition",
        "storm_point",
        "storm_window",
        "storm",
        "near_storm",
        "pregen",
        "storm_id",
        "ms_id",
        "tc_id",
        "sid",
        "u10",
        "v10",
        "u10_mirror",
        "v10_mirror",
        "u_even",
        "u_odd",
        "v_even",
        "v_odd",
        "speed_l",
        "speed_r",
        "eta_parity",
        "_mirror_lon",
        "_mirror_lon_dist",
    ] + [c for c in scalar_cols if c in with_parity.columns] + [
        f"{c}_mirror" for c in scalar_cols if f"{c}_mirror" in with_parity.columns
    ]
    keep_cols = [c for c in keep_cols if c in with_parity.columns]
    out = with_parity[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["speed_l", "speed_r", "eta_parity"])
    written_rows = int(out.shape[0])
    audit = {
        "rows_input": int(in_rows),
        "rows_after_snap_guard": int(snap_guard_rows),
        "rows_mirror_hit": int(mirror_hits),
        "rows_written": int(written_rows),
        "rows_dropped_snap_guard": int(max(0, in_rows - snap_guard_rows)),
        "rows_dropped_missing_mirror": int(max(0, snap_guard_rows - mirror_hits)),
        "rows_dropped_post_parity": int(max(0, snap_guard_rows - written_rows)),
        "snap_distances": dist_vals.astype(float),
        "scalar_ks": scalar_ks,
    }
    return out, audit


def _write_partitioned(
    df: pd.DataFrame,
    out_root: Path,
    file_counter: int,
) -> tuple[int, int, int, dict[str, int]]:
    files = 0
    rows = 0
    partition_rows: dict[str, int] = {}
    for (lead, date), part in df.groupby(["lead_partition", "date"], dropna=False):
        lead_str = str(lead)
        date_str = str(date)
        target = out_root / f"lead={lead_str}" / f"date={date_str}"
        target.mkdir(parents=True, exist_ok=True)
        file_counter += 1
        out_path = target / f"part-{file_counter:08d}.parquet"
        part.to_parquet(out_path, index=False)
        files += 1
        part_rows = int(part.shape[0])
        rows += part_rows
        partition_rows[f"lead={lead_str}/date={date_str}"] = partition_rows.get(
            f"lead={lead_str}/date={date_str}",
            0,
        ) + part_rows
    return files, rows, file_counter, partition_rows


def _lead_partition_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    try:
        fv = float(value)
    except Exception:
        return "none"
    if np.isnan(fv):
        return "none"
    if abs(fv - round(fv)) < 1e-6:
        return str(int(round(fv)))
    return str(fv)


def _init_mirror_audit(scalar_cols: list[str]) -> dict[str, Any]:
    return {
        "rows_input": 0,
        "rows_after_snap_guard": 0,
        "rows_mirror_hit": 0,
        "rows_written": 0,
        "rows_dropped_snap_guard": 0,
        "rows_dropped_missing_mirror": 0,
        "rows_dropped_post_parity": 0,
        "snap_distance_count": 0,
        "snap_distance_sum": 0.0,
        "snap_distance_min": None,
        "snap_distance_max": None,
        "snap_distance_bins": {
            "edges": [0.0, 0.01, 0.02, 0.05, 0.10, 0.13, float("inf")],
            "counts": [0, 0, 0, 0, 0, 0],
        },
        "scalar_ks": {
            col: {
                "n_tests": 0,
                "weighted_ks_stat_sum": 0.0,
                "weight_sum": 0.0,
                "pvalue_sum": 0.0,
                "min_pvalue": None,
            }
            for col in scalar_cols
        },
    }


def _accumulate_mirror_audit(acc: dict[str, Any], delta: dict[str, Any]) -> None:
    for key in (
        "rows_input",
        "rows_after_snap_guard",
        "rows_mirror_hit",
        "rows_written",
        "rows_dropped_snap_guard",
        "rows_dropped_missing_mirror",
        "rows_dropped_post_parity",
    ):
        acc[key] += int(delta.get(key, 0))

    dists = np.asarray(delta.get("snap_distances", []), dtype=float)
    dists = dists[np.isfinite(dists)]
    if dists.size > 0:
        acc["snap_distance_count"] += int(dists.size)
        acc["snap_distance_sum"] += float(dists.sum())
        d_min = float(dists.min())
        d_max = float(dists.max())
        acc["snap_distance_min"] = d_min if acc["snap_distance_min"] is None else min(acc["snap_distance_min"], d_min)
        acc["snap_distance_max"] = d_max if acc["snap_distance_max"] is None else max(acc["snap_distance_max"], d_max)
        edges = np.asarray(acc["snap_distance_bins"]["edges"], dtype=float)
        hist, _ = np.histogram(dists, bins=edges)
        acc["snap_distance_bins"]["counts"] = (
            np.asarray(acc["snap_distance_bins"]["counts"], dtype=int) + hist.astype(int)
        ).tolist()

    for col, stats in delta.get("scalar_ks", {}).items():
        if col not in acc["scalar_ks"]:
            continue
        n = float(min(stats.get("n_left", 0.0), stats.get("n_right", 0.0)))
        if n <= 0:
            continue
        ks_stat = float(stats.get("ks_stat", 0.0))
        pvalue = float(stats.get("pvalue", 1.0))
        rec = acc["scalar_ks"][col]
        rec["n_tests"] += 1
        rec["weighted_ks_stat_sum"] += ks_stat * n
        rec["weight_sum"] += n
        rec["pvalue_sum"] += pvalue
        rec["min_pvalue"] = pvalue if rec["min_pvalue"] is None else min(float(rec["min_pvalue"]), pvalue)


def _finalize_mirror_audit(acc: dict[str, Any]) -> dict[str, Any]:
    rows_input = max(1, int(acc["rows_input"]))
    rows_after_snap = int(acc["rows_after_snap_guard"])
    rows_hit = int(acc["rows_mirror_hit"])
    rows_written = int(acc["rows_written"])
    d_count = int(acc["snap_distance_count"])

    scalar_ks: dict[str, Any] = {}
    for col, rec in acc["scalar_ks"].items():
        weight_sum = float(rec["weight_sum"])
        n_tests = int(rec["n_tests"])
        scalar_ks[col] = {
            "n_tests": n_tests,
            "ks_stat_weighted_mean": float(rec["weighted_ks_stat_sum"] / weight_sum) if weight_sum > 0 else None,
            "pvalue_mean": float(rec["pvalue_sum"] / n_tests) if n_tests > 0 else None,
            "pvalue_min": float(rec["min_pvalue"]) if rec["min_pvalue"] is not None else None,
        }

    return {
        "rows_input": int(acc["rows_input"]),
        "rows_after_snap_guard": rows_after_snap,
        "rows_mirror_hit": rows_hit,
        "rows_written": rows_written,
        "map_hit_rate_after_snap": float(rows_hit / max(1, rows_after_snap)),
        "map_hit_rate_global": float(rows_hit / rows_input),
        "drop_rate_snap_guard": float(acc["rows_dropped_snap_guard"] / rows_input),
        "drop_rate_missing_mirror": float(acc["rows_dropped_missing_mirror"] / rows_input),
        "drop_rate_post_parity": float(acc["rows_dropped_post_parity"] / rows_input),
        "snap_distance": {
            "count": d_count,
            "mean": float(acc["snap_distance_sum"] / d_count) if d_count > 0 else None,
            "min": acc["snap_distance_min"],
            "max": acc["snap_distance_max"],
            "bins": acc["snap_distance_bins"],
        },
        "scalar_distribution_ks": scalar_ks,
    }


def _init_lon0_sensitivity(lon_values: list[float]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lon in sorted({float(v) for v in lon_values}):
        out[f"{lon:.3f}"] = {
            "lon0": float(lon),
            "rows_input": 0,
            "rows_after_snap": 0,
            "rows_written": 0,
            "rows_events": 0,
            "rows_background": 0,
            "eta_sum": 0.0,
            "eta_events_sum": 0.0,
            "eta_background_sum": 0.0,
        }
    return out


def _accumulate_lon0_sensitivity(
    *,
    acc: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    adapter: WeatherERA5Adapter,
    max_lon_distance: float,
    scalar_cols: list[str],
) -> None:
    if not acc:
        return
    for key, rec in acc.items():
        lon0 = float(rec["lon0"])
        mirrored = adapter.mirror_lon_about(
            frame,
            lon0=lon0,
            max_lon_distance=max_lon_distance,
            scalar_cols=scalar_cols,
        )
        if mirrored.empty:
            rec["rows_input"] += int(frame.shape[0])
            continue
        with_parity = adapter.add_parity_channels(mirrored)
        with_parity = with_parity.replace([np.inf, -np.inf], np.nan).dropna(subset=["eta_parity"])
        rec["rows_input"] += int(frame.shape[0])
        rec["rows_after_snap"] += int(mirrored.shape[0])
        rec["rows_written"] += int(with_parity.shape[0])
        if with_parity.empty:
            continue

        eta = pd.to_numeric(with_parity["eta_parity"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(eta)
        eta = eta[valid]
        if eta.size == 0:
            continue
        rec["eta_sum"] += float(np.sum(eta))

        storm = pd.to_numeric(with_parity.get("storm", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
        near = pd.to_numeric(with_parity.get("near_storm", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
        pregen = pd.to_numeric(with_parity.get("pregen", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
        event_mask = ((storm >= 1) | (near >= 1) | (pregen >= 1)) & valid
        bg_mask = ((storm <= 0) & (near <= 0) & (pregen <= 0)) & valid

        n_event = int(np.sum(event_mask))
        n_bg = int(np.sum(bg_mask))
        rec["rows_events"] += n_event
        rec["rows_background"] += n_bg
        if n_event > 0:
            eta_event = pd.to_numeric(with_parity.loc[event_mask, "eta_parity"], errors="coerce").to_numpy(dtype=float)
            eta_event = eta_event[np.isfinite(eta_event)]
            rec["eta_events_sum"] += float(np.sum(eta_event))
        if n_bg > 0:
            eta_bg = pd.to_numeric(with_parity.loc[bg_mask, "eta_parity"], errors="coerce").to_numpy(dtype=float)
            eta_bg = eta_bg[np.isfinite(eta_bg)]
            rec["eta_background_sum"] += float(np.sum(eta_bg))


def _finalize_lon0_sensitivity(acc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_lon: dict[str, Any] = {}
    bg_vals: list[float] = []
    event_vals: list[float] = []
    all_vals: list[float] = []

    for key, rec in sorted(acc.items(), key=lambda kv: kv[0]):
        n_all = max(1, int(rec["rows_written"]))
        n_event = int(rec["rows_events"])
        n_bg = int(rec["rows_background"])
        eta_all = float(rec["eta_sum"] / n_all)
        eta_event = float(rec["eta_events_sum"] / n_event) if n_event > 0 else None
        eta_bg = float(rec["eta_background_sum"] / n_bg) if n_bg > 0 else None

        all_vals.append(eta_all)
        if eta_event is not None:
            event_vals.append(float(eta_event))
        if eta_bg is not None:
            bg_vals.append(float(eta_bg))

        per_lon[key] = {
            "lon0": float(rec["lon0"]),
            "rows_input": int(rec["rows_input"]),
            "rows_after_snap": int(rec["rows_after_snap"]),
            "rows_written": int(rec["rows_written"]),
            "map_hit_rate_after_snap": float(rec["rows_written"] / max(1, rec["rows_after_snap"])),
            "eta_parity_mean_all": eta_all,
            "eta_parity_mean_events": eta_event,
            "eta_parity_mean_background": eta_bg,
        }

    def _rng(vals: list[float]) -> float | None:
        if not vals:
            return None
        return float(max(vals) - min(vals))

    return {
        "per_lon0": per_lon,
        "sensitivity_ranges": {
            "eta_parity_mean_all_range": _rng(all_vals),
            "eta_parity_mean_events_range": _rng(event_vals),
            "eta_parity_mean_background_range": _rng(bg_vals),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
