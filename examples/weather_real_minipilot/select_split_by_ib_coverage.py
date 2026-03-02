from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank split dates by strict IBTrACS event/far coverage before running evaluation."
    )
    parser.add_argument("--dataset", required=True, help="Tile dataset directory containing samples.parquet")
    parser.add_argument("--out", required=True, help="Output directory for coverage scan artifacts")
    parser.add_argument(
        "--candidate-splits",
        default="weekly",
        help="Split cadence: weekly|biweekly|10d|daily|monthly, or comma-separated explicit dates",
    )
    parser.add_argument("--split-dates", nargs="*", default=[], help="Optional explicit split dates (YYYY-MM-DD)")
    parser.add_argument("--time-buffer-hours", type=float, default=24.0, help="Blocked split buffer hours")
    parser.add_argument("--min-train-cases", type=int, default=15, help="Minimum train cases")
    parser.add_argument("--min-test-cases", type=int, default=15, help="Minimum test cases")
    parser.add_argument("--min-event-train-cases", type=int, default=10, help="Minimum strict event train cases")
    parser.add_argument("--min-far-nonstorm-train-cases", type=int, default=20, help="Minimum strict far train cases")
    parser.add_argument("--min-event-test-cases", type=int, default=15, help="Minimum strict event test cases")
    parser.add_argument("--min-far-nonstorm-test-cases", type=int, default=15, help="Minimum strict far test cases")
    parser.add_argument("--min-event-train-per-lead", type=int, default=0, help="Optional strict per-lead event train minimum")
    parser.add_argument("--min-far-nonstorm-train-per-lead", type=int, default=0, help="Optional strict per-lead far train minimum")
    parser.add_argument("--min-event-test-per-lead", type=int, default=0, help="Optional strict per-lead event minimum")
    parser.add_argument("--min-far-nonstorm-test-per-lead", type=int, default=0, help="Optional strict per-lead far minimum")
    parser.add_argument(
        "--claim-far-quality-tags",
        nargs="+",
        default=["A_strict_clean"],
        help="Allowed strict far quality tags",
    )
    parser.add_argument("--strict-far-min-row-quality-frac", type=float, default=0.0, help="Minimum per-case row-quality fraction for strict far")
    parser.add_argument("--strict-far-min-kinematic-clean-frac", type=float, default=0.0, help="Minimum per-case kinematic-clean fraction for strict far")
    parser.add_argument("--strict-far-min-any-storm-km", type=float, default=0.0, help="Minimum per-case min-dist-any-storm for strict far")
    parser.add_argument("--strict-far-min-nearest-storm-km", type=float, default=0.0, help="Minimum per-case nearest-storm distance for strict far")
    parser.add_argument(
        "--time-basis",
        choices=["selected", "source", "valid"],
        default="selected",
        help="Time basis for strict IB coverage check",
    )
    parser.add_argument(
        "--ibtracs-strict-use-flags",
        dest="strict_use_flags",
        action="store_true",
        help="Use strict IB flags when available",
    )
    parser.add_argument(
        "--ibtracs-strict-no-flags",
        dest="strict_use_flags",
        action="store_false",
        help="Do not require strict IB flags",
    )
    parser.set_defaults(strict_use_flags=True)
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommended split dates")
    return parser.parse_args()


def _load_eval_module() -> Any:
    mod_path = Path(__file__).resolve().parent / "evaluate_minipilot.py"
    spec = importlib.util.spec_from_file_location("eval_minipilot_covscan", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _candidate_dates(samples: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.split_dates:
        return sorted({str(v) for v in args.split_dates})
    raw = str(args.candidate_splits or "weekly").strip()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            return sorted(set(parts))
    if samples.empty or "t" not in samples.columns:
        return []
    t = pd.to_datetime(samples["t"], errors="coerce").dropna()
    if t.empty:
        return []
    t_min = pd.Timestamp(t.min()).floor("D")
    t_max = pd.Timestamp(t.max()).floor("D")
    freq_map = {
        "daily": "1D",
        "weekly": "7D",
        "biweekly": "14D",
        "10d": "10D",
        "monthly": "MS",
    }
    freq = freq_map.get(raw.lower(), "7D")
    idx = pd.date_range(start=t_min, end=t_max, freq=freq)
    return [str(v.date()) for v in idx]


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _runtime_estimate(n_test_cases: int, n_test_rows: int) -> float:
    # Coarse minutes estimate for quick planning.
    return float(max(0.5, (float(n_test_cases) / 40.0) + (float(n_test_rows) / 40000.0)))


def _normalize_object_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        series = out[col]
        if not series.map(lambda x: isinstance(x, (dict, list, tuple, set))).any():
            continue
        out[col] = series.map(
            lambda x: json.dumps(x, sort_keys=True)
            if isinstance(x, dict)
            else (json.dumps(list(x)) if isinstance(x, (list, tuple, set)) else (None if pd.isna(x) else str(x)))
        )
    return out


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(dataset / "samples.parquet")
    samples["t"] = pd.to_datetime(samples["t"], errors="coerce")
    mod = _load_eval_module()

    dates = _candidate_dates(samples, args)
    if not dates:
        raise ValueError("no_candidate_split_dates")

    storm_id_col = mod._resolve_storm_id_col(samples, requested="auto")
    case_meta = mod._build_case_meta(samples, storm_id_col=storm_id_col)
    allowed_tags = tuple(
        str(v).strip()
        for tok in (args.claim_far_quality_tags or [])
        for v in str(tok).split(",")
        if str(v).strip()
    )
    rows: list[dict[str, Any]] = []
    for split_date in dates:
        split_ts = pd.Timestamp(str(split_date))
        split = mod._blocked_time_split(
            case_meta=case_meta,
            split_date=split_ts,
            buffer_hours=float(args.time_buffer_hours),
        )
        train_cases = set(split.get("train_cases", set()))
        test_cases = set(split.get("test_cases", set()))
        buffer_cases = set(split.get("buffer_cases", set()))
        strict_cov_test = mod._strict_ib_coverage(
            samples=samples,
            case_ids=test_cases,
            time_basis=str(args.time_basis),
            use_flags=bool(args.strict_use_flags),
            far_quality_tags=allowed_tags,
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        strict_cov_train = mod._strict_ib_coverage(
            samples=samples,
            case_ids=train_cases,
            time_basis=str(args.time_basis),
            use_flags=bool(args.strict_use_flags),
            far_quality_tags=allowed_tags,
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        ok_cov_test, reasons_test = mod._strict_coverage_ok(
            coverage=strict_cov_test,
            min_event_total=int(args.min_event_test_cases),
            min_far_total=int(args.min_far_nonstorm_test_cases),
            min_event_per_lead=int(args.min_event_test_per_lead),
            min_far_per_lead=int(args.min_far_nonstorm_test_per_lead),
        )
        ok_cov_train, reasons_train = mod._strict_coverage_ok(
            coverage=strict_cov_train,
            min_event_total=int(args.min_event_train_cases),
            min_far_total=int(args.min_far_nonstorm_train_cases),
            min_event_per_lead=int(args.min_event_train_per_lead),
            min_far_per_lead=int(args.min_far_nonstorm_train_per_lead),
        )
        reasons = [f"test:{v}" for v in reasons_test] + [f"train:{v}" for v in reasons_train]
        ok_split = bool(
            (int(len(train_cases)) >= int(args.min_train_cases))
            and (int(len(test_cases)) >= int(args.min_test_cases))
        )
        if not ok_split:
            if int(len(train_cases)) < int(args.min_train_cases):
                reasons = [*reasons, f"train_cases<{int(args.min_train_cases)}"]
            if int(len(test_cases)) < int(args.min_test_cases):
                reasons = [*reasons, f"test_cases<{int(args.min_test_cases)}"]
        ok = bool(ok_cov_test and ok_cov_train and ok_split)
        test_rows = samples.loc[samples["case_id"].astype(str).isin(test_cases)].copy()
        n_test_rows = int(test_rows.shape[0])
        ev_test = int(strict_cov_test.get("event_total", 0))
        fr_test = int(strict_cov_test.get("far_total", 0))
        ev_train = int(strict_cov_train.get("event_total", 0))
        fr_train = int(strict_cov_train.get("far_total", 0))
        min_event_pair = int(min(ev_train, ev_test))
        min_far_pair = int(min(fr_train, fr_test))
        event_balance_penalty = float(abs(ev_train - ev_test))
        far_balance_penalty = float(abs(fr_train - fr_test))
        # Prioritize trainable splits: maximize joint train/test strict coverage and balance.
        score = (
            (1_000_000.0 if ok else 0.0)
            + 2.0 * float(min_event_pair + min_far_pair)
            + 0.5 * float(ev_train + ev_test + fr_train + fr_test)
            - 0.25 * (event_balance_penalty + far_balance_penalty)
        )
        rows.append(
            {
                "split_date": str(split_date),
                "n_train_cases": int(len(train_cases)),
                "n_test_cases": int(len(test_cases)),
                "n_buffer_cases": int(len(buffer_cases)),
                "strict_event_train_cases": ev_train,
                "strict_far_train_cases": fr_train,
                "strict_event_train_by_lead": strict_cov_train.get("event_by_lead", {}),
                "strict_far_train_by_lead": strict_cov_train.get("far_by_lead", {}),
                "strict_event_test_cases": ev_test,
                "strict_far_test_cases": fr_test,
                "strict_event_test_by_lead": strict_cov_test.get("event_by_lead", {}),
                "strict_far_test_by_lead": strict_cov_test.get("far_by_lead", {}),
                "strict_min_event_train_test_pair": int(min_event_pair),
                "strict_min_far_train_test_pair": int(min_far_pair),
                "claim_coverage_ok": bool(ok),
                "coverage_reasons": [str(v) for v in reasons],
                "time_basis": str(args.time_basis),
                "strict_use_flags": bool(args.strict_use_flags),
                "min_train_cases": int(args.min_train_cases),
                "min_test_cases": int(args.min_test_cases),
                "far_quality_tags": [str(v) for v in allowed_tags],
                "strict_far_min_row_quality_frac": float(args.strict_far_min_row_quality_frac),
                "strict_far_min_kinematic_clean_frac": float(args.strict_far_min_kinematic_clean_frac),
                "strict_far_min_any_storm_km": float(args.strict_far_min_any_storm_km),
                "strict_far_min_nearest_storm_km": float(args.strict_far_min_nearest_storm_km),
                "n_test_rows": int(n_test_rows),
                "runtime_estimate_min": _runtime_estimate(int(len(test_cases)), n_test_rows),
                "score": float(score),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=[
            "score",
            "strict_min_event_train_test_pair",
            "strict_min_far_train_test_pair",
            "strict_event_test_cases",
            "strict_far_test_cases",
        ],
        ascending=[False, False, False, False, False],
    )
    df_write = _normalize_object_columns_for_parquet(df)
    parquet_path = outdir / "split_coverage_scan.parquet"
    csv_path = outdir / "split_coverage_scan.csv"
    df_write.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    ok_df = df.loc[df["claim_coverage_ok"] == True].copy() if "claim_coverage_ok" in df.columns else pd.DataFrame()
    rec_df = ok_df.head(int(max(1, args.top_n))) if not ok_df.empty else df.head(int(max(1, args.top_n)))
    recommended = {
        "dataset": str(dataset.resolve()),
        "time_buffer_hours": float(args.time_buffer_hours),
        "thresholds": {
            "min_train_cases": int(args.min_train_cases),
            "min_test_cases": int(args.min_test_cases),
            "min_event_train_cases": int(args.min_event_train_cases),
            "min_far_nonstorm_train_cases": int(args.min_far_nonstorm_train_cases),
            "min_event_test_cases": int(args.min_event_test_cases),
            "min_far_nonstorm_test_cases": int(args.min_far_nonstorm_test_cases),
            "min_event_train_per_lead": int(args.min_event_train_per_lead),
            "min_far_nonstorm_train_per_lead": int(args.min_far_nonstorm_train_per_lead),
            "min_event_test_per_lead": int(args.min_event_test_per_lead),
            "min_far_nonstorm_test_per_lead": int(args.min_far_nonstorm_test_per_lead),
            "time_basis": str(args.time_basis),
            "strict_use_flags": bool(args.strict_use_flags),
            "strict_far_min_row_quality_frac": float(args.strict_far_min_row_quality_frac),
            "strict_far_min_kinematic_clean_frac": float(args.strict_far_min_kinematic_clean_frac),
            "strict_far_min_any_storm_km": float(args.strict_far_min_any_storm_km),
            "strict_far_min_nearest_storm_km": float(args.strict_far_min_nearest_storm_km),
            "claim_far_quality_tags": [str(v) for v in allowed_tags],
        },
        "n_candidates": int(df.shape[0]),
        "n_coverage_ok": int(ok_df.shape[0]),
        "recommended_split_dates": [str(v) for v in rec_df["split_date"].tolist()],
        "recommended_rows": rec_df.to_dict(orient="records"),
        "artifacts": {
            "scan_parquet": str(parquet_path.resolve()),
            "scan_csv": str(csv_path.resolve()),
        },
    }
    rec_path = outdir / "recommended_splits.json"
    rec_path.write_text(json.dumps(recommended, indent=2, sort_keys=True), encoding="utf-8")
    md_path = outdir / "split_coverage_summary.md"
    lines = [
        "# Split Coverage Scan",
        "",
        f"- Candidates scanned: `{int(df.shape[0])}`",
        f"- Coverage-OK splits: `{int(ok_df.shape[0])}`",
        f"- Recommended splits (top {int(max(1, args.top_n))}): `{', '.join(recommended['recommended_split_dates'])}`",
        "",
        "## Thresholds",
        f"- `min_event_test_cases={int(args.min_event_test_cases)}`",
        f"- `min_far_nonstorm_test_cases={int(args.min_far_nonstorm_test_cases)}`",
        f"- `min_event_train_cases={int(args.min_event_train_cases)}`",
        f"- `min_far_nonstorm_train_cases={int(args.min_far_nonstorm_train_cases)}`",
        f"- `min_train_cases={int(args.min_train_cases)}`",
        f"- `min_test_cases={int(args.min_test_cases)}`",
        f"- `min_event_train_per_lead={int(args.min_event_train_per_lead)}`",
        f"- `min_far_nonstorm_train_per_lead={int(args.min_far_nonstorm_train_per_lead)}`",
        f"- `min_event_test_per_lead={int(args.min_event_test_per_lead)}`",
        f"- `min_far_nonstorm_test_per_lead={int(args.min_far_nonstorm_test_per_lead)}`",
        f"- `time_basis={str(args.time_basis)}`",
        f"- `strict_use_flags={bool(args.strict_use_flags)}`",
        f"- `strict_far_min_row_quality_frac={float(args.strict_far_min_row_quality_frac)}`",
        f"- `strict_far_min_kinematic_clean_frac={float(args.strict_far_min_kinematic_clean_frac)}`",
        f"- `strict_far_min_any_storm_km={float(args.strict_far_min_any_storm_km)}`",
        f"- `strict_far_min_nearest_storm_km={float(args.strict_far_min_nearest_storm_km)}`",
        f"- `claim_far_quality_tags={','.join([str(v) for v in allowed_tags])}`",
        "",
        "## Outputs",
        f"- `{parquet_path}`",
        f"- `{csv_path}`",
        f"- `{rec_path}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote split coverage scan to {parquet_path}")
    print(f"Wrote recommendations to {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
