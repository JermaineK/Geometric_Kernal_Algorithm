from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coverage scan v2 using strict IBTrACS labels plus soft eventness/farness windows."
    )
    parser.add_argument("--dataset", required=True, help="Tile dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for coverage scan artifacts")
    parser.add_argument(
        "--candidate-splits",
        default="weekly",
        help="weekly|biweekly|10d|daily|monthly or comma-separated split dates",
    )
    parser.add_argument("--split-dates", nargs="*", default=[], help="Optional explicit split dates")
    parser.add_argument("--time-buffer-hours", type=float, default=24.0, help="Split buffer around boundary")
    parser.add_argument("--min-train-cases", type=int, default=15, help="Minimum train cases")
    parser.add_argument("--min-test-cases", type=int, default=15, help="Minimum test cases")
    parser.add_argument("--min-event-train-cases", type=int, default=10, help="Minimum strict IB event train cases")
    parser.add_argument("--min-far-train-cases", type=int, default=20, help="Minimum strict IB far train cases")
    parser.add_argument("--min-event-test-cases", type=int, default=15, help="Minimum strict IB event test cases")
    parser.add_argument("--min-far-test-cases", type=int, default=15, help="Minimum strict IB far test cases")
    parser.add_argument(
        "--min-event-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict IB event train minimum for coverage pass",
    )
    parser.add_argument(
        "--min-far-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict IB far train minimum for coverage pass",
    )
    parser.add_argument(
        "--min-event-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict IB event minimum for coverage pass",
    )
    parser.add_argument(
        "--min-far-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict IB far minimum for coverage pass",
    )
    parser.add_argument(
        "--claim-far-quality-tags",
        nargs="+",
        default=["A_strict_clean"],
        help="Allowed ib_far_quality_tag values for strict far coverage counting",
    )
    parser.add_argument("--strict-far-min-row-quality-frac", type=float, default=0.0, help="Minimum per-case row-quality fraction for strict far")
    parser.add_argument("--strict-far-min-kinematic-clean-frac", type=float, default=0.0, help="Minimum per-case kinematic-clean fraction for strict far")
    parser.add_argument("--strict-far-min-any-storm-km", type=float, default=0.0, help="Minimum per-case min-dist-any-storm for strict far")
    parser.add_argument("--strict-far-min-nearest-storm-km", type=float, default=0.0, help="Minimum per-case nearest-storm distance for strict far")
    parser.add_argument(
        "--strict-coverage-time-basis",
        choices=["selected", "source", "valid"],
        default="selected",
        help="Time basis for strict IB coverage checks",
    )
    parser.add_argument(
        "--strict-coverage-use-flags",
        dest="strict_coverage_use_flags",
        action="store_true",
        help="Use strict precomputed IB flags in strict coverage checks",
    )
    parser.add_argument(
        "--strict-coverage-no-flags",
        dest="strict_coverage_use_flags",
        action="store_false",
        help="Ignore strict precomputed flags and rely on raw distance/time fields",
    )
    parser.set_defaults(strict_coverage_use_flags=True)
    parser.add_argument("--eventness-min", type=float, default=0.15, help="Soft eventness threshold for coverage")
    parser.add_argument("--farness-min", type=float, default=0.70, help="Soft farness threshold for coverage")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommended splits")
    return parser.parse_args()


def _load_eval_module() -> Any:
    mod_path = Path(__file__).resolve().parent / "evaluate_minipilot.py"
    spec = importlib.util.spec_from_file_location("eval_covscan_v2", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _candidate_dates(samples: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.split_dates:
        return sorted({str(v) for v in args.split_dates})
    raw = str(args.candidate_splits or "weekly").strip()
    if "," in raw:
        vals = [str(v).strip() for v in raw.split(",") if str(v).strip()]
        if vals:
            return sorted(set(vals))
    if samples.empty or "t" not in samples.columns:
        return []
    t = pd.to_datetime(samples["t"], errors="coerce").dropna()
    if t.empty:
        return []
    start = pd.Timestamp(t.min()).floor("D")
    end = pd.Timestamp(t.max()).floor("D")
    freq_map = {"daily": "1D", "weekly": "7D", "biweekly": "14D", "10d": "10D", "monthly": "MS"}
    freq = freq_map.get(raw.lower(), "7D")
    idx = pd.date_range(start=start, end=end, freq=freq)
    return [str(v.date()) for v in idx]


def _load_case_windows(dataset_dir: Path, samples: pd.DataFrame) -> pd.DataFrame:
    path = dataset_dir / "case_windows.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        agg = {
            "eventness": ("eventness", "mean") if "eventness" in samples.columns else ("case_id", "size"),
            "farness": ("farness", "mean") if "farness" in samples.columns else ("case_id", "size"),
            "ib_event_strict": ("ib_event_strict", "max") if "ib_event_strict" in samples.columns else ("case_id", "size"),
            "ib_far_strict": ("ib_far_strict", "max") if "ib_far_strict" in samples.columns else ("case_id", "size"),
            "lead_bucket": ("lead_bucket", "first") if "lead_bucket" in samples.columns else ("case_id", "size"),
        }
        df = samples.groupby("case_id", as_index=False).agg(**{k: v for k, v in agg.items() if v[0] in samples.columns})
    if "case_id" not in df.columns:
        return pd.DataFrame(columns=["case_id"])
    df["case_id"] = df["case_id"].astype(str)
    for col in ("eventness", "farness"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    for col in ("ib_event_strict", "ib_far_strict"):
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)
        else:
            df[col] = 0
    if "lead_bucket" not in df.columns:
        df["lead_bucket"] = "none"
    df["lead_bucket"] = df["lead_bucket"].astype(str)
    return df


def _runtime_estimate(n_test_cases: int) -> float:
    return float(max(0.5, float(n_test_cases) / 30.0))


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
    dataset_dir = Path(args.dataset)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(dataset_dir / "samples.parquet")
    samples["t"] = pd.to_datetime(samples["t"], errors="coerce")
    case_windows = _load_case_windows(dataset_dir, samples)
    dates = _candidate_dates(samples, args)
    if not dates:
        raise ValueError("no_candidate_split_dates")

    mod = _load_eval_module()
    storm_id_col = mod._resolve_storm_id_col(samples, requested="auto")
    case_meta = mod._build_case_meta(samples, storm_id_col=storm_id_col)
    allowed_far_tags = {
        str(v).strip()
        for tok in (args.claim_far_quality_tags or [])
        for v in str(tok).split(",")
        if str(v).strip()
    }

    rows: list[dict[str, Any]] = []
    for split_date in dates:
        split = mod._blocked_time_split(
            case_meta=case_meta,
            split_date=pd.Timestamp(str(split_date)),
            buffer_hours=float(args.time_buffer_hours),
        )
        train_cases = set(split.get("train_cases", set()))
        test_cases = set(split.get("test_cases", set()))
        buffer_cases = set(split.get("buffer_cases", set()))

        strict_cov_test = mod._strict_ib_coverage(
            samples=samples,
            case_ids=set(test_cases),
            time_basis=str(args.strict_coverage_time_basis),
            use_flags=bool(args.strict_coverage_use_flags),
            far_quality_tags=tuple(sorted(allowed_far_tags)),
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        strict_cov_train = mod._strict_ib_coverage(
            samples=samples,
            case_ids=set(train_cases),
            time_basis=str(args.strict_coverage_time_basis),
            use_flags=bool(args.strict_coverage_use_flags),
            far_quality_tags=tuple(sorted(allowed_far_tags)),
            min_far_quality_fraction=float(args.strict_far_min_row_quality_frac),
            min_far_kinematic_clean_fraction=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        strict_test_ok, strict_test_reasons = mod._strict_coverage_ok(
            coverage=strict_cov_test,
            min_event_total=int(args.min_event_test_cases),
            min_far_total=int(args.min_far_test_cases),
            min_event_per_lead=int(args.min_event_test_per_lead),
            min_far_per_lead=int(args.min_far_test_per_lead),
        )
        strict_train_ok, strict_train_reasons = mod._strict_coverage_ok(
            coverage=strict_cov_train,
            min_event_total=int(args.min_event_train_cases),
            min_far_total=int(args.min_far_train_cases),
            min_event_per_lead=int(args.min_event_train_per_lead),
            min_far_per_lead=int(args.min_far_train_per_lead),
        )
        strict_event = int(strict_cov_test.get("event_total", 0))
        strict_far = int(strict_cov_test.get("far_total", 0))
        strict_event_train = int(strict_cov_train.get("event_total", 0))
        strict_far_train = int(strict_cov_train.get("far_total", 0))

        test_win = case_windows.loc[case_windows["case_id"].astype(str).isin({str(v) for v in test_cases})].copy()
        soft_event = int((pd.to_numeric(test_win.get("eventness"), errors="coerce").fillna(0.0) >= float(args.eventness_min)).sum())
        soft_far = int((pd.to_numeric(test_win.get("farness"), errors="coerce").fillna(0.0) >= float(args.farness_min)).sum())

        coverage_ok = bool(
            strict_test_ok
            and strict_train_ok
            and (len(train_cases) >= int(args.min_train_cases))
            and (len(test_cases) >= int(args.min_test_cases))
        )
        min_event_pair = int(min(strict_event_train, strict_event))
        min_far_pair = int(min(strict_far_train, strict_far))
        score = (
            (1_000_000.0 if coverage_ok else 0.0)
            + 2.0 * float(min_event_pair + min_far_pair)
            + 0.5 * float(strict_event + strict_far + strict_event_train + strict_far_train)
            + 0.25 * float(soft_event + soft_far)
            - 0.25 * float(abs(strict_event_train - strict_event) + abs(strict_far_train - strict_far))
        )
        reasons: list[str] = []
        reasons.extend([f"test:{v}" for v in strict_test_reasons])
        reasons.extend([f"train:{v}" for v in strict_train_reasons])
        if len(train_cases) < int(args.min_train_cases):
            reasons.append(f"train_cases<{int(args.min_train_cases)}")
        if len(test_cases) < int(args.min_test_cases):
            reasons.append(f"test_cases<{int(args.min_test_cases)}")

        rows.append(
            {
                "split_date": str(split_date),
                "n_train_cases": int(len(train_cases)),
                "n_test_cases": int(len(test_cases)),
                "n_buffer_cases": int(len(buffer_cases)),
                "strict_event_train_cases": int(strict_event_train),
                "strict_far_train_cases": int(strict_far_train),
                "strict_event_train_by_lead": strict_cov_train.get("event_by_lead", {}),
                "strict_far_train_by_lead": strict_cov_train.get("far_by_lead", {}),
                "strict_event_test_cases": int(strict_event),
                "strict_far_test_cases": int(strict_far),
                "strict_event_test_by_lead": strict_cov_test.get("event_by_lead", {}),
                "strict_far_test_by_lead": strict_cov_test.get("far_by_lead", {}),
                "strict_event_col_used": strict_cov_test.get("event_col_used"),
                "strict_far_col_used": strict_cov_test.get("far_col_used"),
                "strict_time_basis": strict_cov_test.get("time_basis", str(args.strict_coverage_time_basis)),
                "strict_use_flags": bool(strict_cov_test.get("use_flags", args.strict_coverage_use_flags)),
                "soft_event_test_cases": int(soft_event),
                "soft_far_test_cases": int(soft_far),
                "strict_min_event_train_test_pair": int(min_event_pair),
                "strict_min_far_train_test_pair": int(min_far_pair),
                "coverage_ok": bool(coverage_ok),
                "coverage_reasons": [str(v) for v in reasons],
                "claim_far_quality_tags": sorted(allowed_far_tags),
                "runtime_estimate_min": _runtime_estimate(int(len(test_cases))),
                "score": float(score),
            }
        )

    scan = pd.DataFrame(rows).sort_values(
        ["score", "strict_min_event_train_test_pair", "strict_min_far_train_test_pair", "strict_event_test_cases", "strict_far_test_cases"],
        ascending=[False, False, False, False, False],
    )
    scan_parquet = outdir / "coverage_scan_v2.parquet"
    scan_csv = outdir / "coverage_scan_v2.csv"
    scan_write = _normalize_object_columns_for_parquet(scan)
    scan_write.to_parquet(scan_parquet, index=False)
    scan.to_csv(scan_csv, index=False)

    ok = scan.loc[scan["coverage_ok"] == True].copy() if "coverage_ok" in scan.columns else pd.DataFrame()
    rec = ok.head(int(max(1, args.top_n))) if not ok.empty else scan.head(int(max(1, args.top_n)))
    rec_json = {
        "dataset": str(dataset_dir.resolve()),
        "n_candidates": int(scan.shape[0]),
        "n_coverage_ok": int(ok.shape[0]),
        "recommended_split_dates": [str(v) for v in rec["split_date"].tolist()],
        "recommended_rows": rec.to_dict(orient="records"),
        "thresholds": {
            "min_train_cases": int(args.min_train_cases),
            "min_test_cases": int(args.min_test_cases),
            "min_event_train_cases": int(args.min_event_train_cases),
            "min_far_train_cases": int(args.min_far_train_cases),
            "min_event_test_cases": int(args.min_event_test_cases),
            "min_far_test_cases": int(args.min_far_test_cases),
            "min_event_train_per_lead": int(args.min_event_train_per_lead),
            "min_far_train_per_lead": int(args.min_far_train_per_lead),
            "min_event_test_per_lead": int(args.min_event_test_per_lead),
            "min_far_test_per_lead": int(args.min_far_test_per_lead),
            "eventness_min": float(args.eventness_min),
            "farness_min": float(args.farness_min),
            "claim_far_quality_tags": sorted(allowed_far_tags),
            "strict_coverage_time_basis": str(args.strict_coverage_time_basis),
            "strict_coverage_use_flags": bool(args.strict_coverage_use_flags),
            "strict_far_min_row_quality_frac": float(args.strict_far_min_row_quality_frac),
            "strict_far_min_kinematic_clean_frac": float(args.strict_far_min_kinematic_clean_frac),
            "strict_far_min_any_storm_km": float(args.strict_far_min_any_storm_km),
            "strict_far_min_nearest_storm_km": float(args.strict_far_min_nearest_storm_km),
        },
        "artifacts": {
            "coverage_scan_v2_parquet": str(scan_parquet.resolve()),
            "coverage_scan_v2_csv": str(scan_csv.resolve()),
        },
    }
    rec_path = outdir / "coverage_scan_v2_recommended.json"
    rec_path.write_text(json.dumps(rec_json, indent=2, sort_keys=True), encoding="utf-8")

    md = outdir / "coverage_scan_v2_summary.md"
    md.write_text(
        "\n".join(
            [
                "# Coverage Scan v2",
                "",
                f"- Candidate splits: `{int(scan.shape[0])}`",
                f"- Coverage-OK splits: `{int(ok.shape[0])}`",
                f"- Recommended: `{', '.join([str(v) for v in rec_json['recommended_split_dates']])}`",
                "",
                "## Thresholds",
                f"- `min_event_test_cases={int(args.min_event_test_cases)}`",
                f"- `min_far_test_cases={int(args.min_far_test_cases)}`",
                f"- `min_event_train_cases={int(args.min_event_train_cases)}`",
                f"- `min_far_train_cases={int(args.min_far_train_cases)}`",
                f"- `eventness_min={float(args.eventness_min):.3f}`",
                f"- `farness_min={float(args.farness_min):.3f}`",
                f"- `strict_coverage_time_basis={str(args.strict_coverage_time_basis)}`",
                f"- `strict_coverage_use_flags={bool(args.strict_coverage_use_flags)}`",
                f"- `strict_far_min_row_quality_frac={float(args.strict_far_min_row_quality_frac)}`",
                f"- `strict_far_min_kinematic_clean_frac={float(args.strict_far_min_kinematic_clean_frac)}`",
                f"- `strict_far_min_any_storm_km={float(args.strict_far_min_any_storm_km)}`",
                f"- `strict_far_min_nearest_storm_km={float(args.strict_far_min_nearest_storm_km)}`",
                f"- `min_event_train_per_lead={int(args.min_event_train_per_lead)}`",
                f"- `min_far_train_per_lead={int(args.min_far_train_per_lead)}`",
                f"- `min_event_test_per_lead={int(args.min_event_test_per_lead)}`",
                f"- `min_far_test_per_lead={int(args.min_far_test_per_lead)}`",
                "",
                "## Outputs",
                f"- `{scan_parquet}`",
                f"- `{scan_csv}`",
                f"- `{rec_path}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote coverage scan v2 to {scan_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
