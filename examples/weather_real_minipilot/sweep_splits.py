from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather minipilot evaluation across multiple split dates")
    parser.add_argument("--dataset", required=True, help="Canonical tile dataset directory")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument("--outdir", required=True, help="Output directory for sweep artifacts")
    parser.add_argument(
        "--split-dates",
        nargs="*",
        default=[],
        help="Explicit split dates (YYYY-MM-DD). If omitted, date range args are used.",
    )
    parser.add_argument("--split-start", default="2025-02-15", help="Start date when generating split dates")
    parser.add_argument("--split-end", default="2025-05-15", help="End date when generating split dates")
    parser.add_argument("--split-freq", default="7D", help="Frequency for generated split dates (pandas offset alias)")
    parser.add_argument("--time-buffer-hours", type=float, default=72.0, help="Blocked split buffer hours")
    parser.add_argument(
        "--min-ibtracs-event-test-cases",
        "--min-strict-event-test-cases",
        dest="min_strict_event_test_cases",
        type=int,
        default=10,
        help="Skip split when test-set IBTrACS event case coverage is below this threshold",
    )
    parser.add_argument(
        "--min-ibtracs-far-test-cases",
        "--min-strict-far-test-cases",
        dest="min_strict_far_test_cases",
        type=int,
        default=10,
        help="Skip split when test-set IBTrACS far case coverage is below this threshold",
    )
    parser.add_argument("--min-train-cases", type=int, default=15, help="Minimum train cases required for evaluation")
    parser.add_argument("--min-test-cases", type=int, default=15, help="Minimum test cases required for evaluation")
    parser.add_argument(
        "--min-strict-event-train-cases",
        type=int,
        default=10,
        help="Minimum strict event train coverage required for split precheck",
    )
    parser.add_argument(
        "--min-strict-far-train-cases",
        type=int,
        default=20,
        help="Minimum strict far train coverage required for split precheck",
    )
    parser.add_argument(
        "--min-strict-event-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict event minimum for split precheck",
    )
    parser.add_argument(
        "--min-strict-far-test-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict far minimum for split precheck",
    )
    parser.add_argument(
        "--min-strict-event-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict event train minimum for split precheck",
    )
    parser.add_argument(
        "--min-strict-far-train-per-lead",
        type=int,
        default=0,
        help="Optional per-lead strict far train minimum for split precheck",
    )
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
    parser.add_argument(
        "--claim-far-quality-tags",
        nargs="+",
        default=["A_strict_clean"],
        help="Allowed far quality tags used for strict IB far coverage checks",
    )
    parser.add_argument("--strict-far-min-row-quality-frac", type=float, default=0.0, help="Minimum per-case row-quality fraction for strict far")
    parser.add_argument("--strict-far-min-kinematic-clean-frac", type=float, default=0.0, help="Minimum per-case kinematic-clean fraction for strict far")
    parser.add_argument("--strict-far-min-any-storm-km", type=float, default=0.0, help="Minimum per-case min-dist-any-storm for strict far")
    parser.add_argument("--strict-far-min-nearest-storm-km", type=float, default=0.0, help="Minimum per-case nearest-storm distance for strict far")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to call evaluator")
    parser.add_argument("--eval-extra-arg", action="append", default=[], help="Extra arg token forwarded to evaluator")
    return parser.parse_args()


def _split_dates(args: argparse.Namespace) -> list[str]:
    if args.split_dates:
        return sorted({str(v) for v in args.split_dates})
    idx = pd.date_range(start=str(args.split_start), end=str(args.split_end), freq=str(args.split_freq))
    return [str(v.date()) for v in idx]


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    return value


def _serialize_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) != "object":
            continue
        series = out[col]
        has_nested = bool(series.map(lambda v: isinstance(v, (dict, list, tuple, set))).any())
        if has_nested:
            out[col] = series.map(_serialize_cell)
    return out


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        out = float(v)
        if not pd.notna(out):
            return None
        return out
    except Exception:
        return None


def _load_eval_module():
    mod_path = Path(__file__).resolve().parent / "evaluate_minipilot.py"
    spec = importlib.util.spec_from_file_location("eval_minipilot_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_quality_tags(values: list[str] | None) -> tuple[str, ...]:
    tags = {
        str(v).strip()
        for tok in (values or [])
        for v in str(tok).split(",")
        if str(v).strip()
    }
    return tuple(sorted(tags))


def _precheck_ibtracs_coverage(
    *,
    samples: pd.DataFrame,
    eval_mod: Any,
    split_date: str,
    time_buffer_hours: float,
    min_event_cases: int,
    min_far_cases: int,
    min_event_train_cases: int,
    min_far_train_cases: int,
    min_train_cases: int,
    min_test_cases: int,
    min_event_per_lead: int,
    min_far_per_lead: int,
    min_event_train_per_lead: int,
    min_far_train_per_lead: int,
    strict_time_basis: str,
    strict_use_flags: bool,
    far_quality_tags: tuple[str, ...],
    strict_far_min_row_quality_frac: float = 0.0,
    strict_far_min_kinematic_clean_frac: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> dict[str, Any]:
    if samples.empty:
        return {"available": False, "ok": False, "reason": "empty_samples", "n_event_test_cases": 0, "n_far_test_cases": 0}
    storm_col = eval_mod._resolve_storm_id_col(samples, requested="auto")
    case_meta = eval_mod._build_case_meta(samples, storm_id_col=storm_col)
    split = eval_mod._blocked_time_split(
        case_meta=case_meta,
        split_date=pd.Timestamp(split_date),
        buffer_hours=float(time_buffer_hours),
    )
    train_cases = set(split.get("train_cases", set()))
    test_cases = set(split.get("test_cases", set()))
    try:
        eval_mod._enforce_split_case_counts(
            train_cases=train_cases,
            test_cases=test_cases,
            min_train=int(min_train_cases),
            min_test=int(min_test_cases),
        )
    except Exception as exc:
        return {
            "available": True,
            "ok": False,
            "reason": str(exc),
            "n_train_cases": int(len(train_cases)),
            "n_test_cases": int(len(test_cases)),
            "n_event_test_cases": 0,
            "n_far_test_cases": 0,
        }
    if not test_cases:
        return {"available": True, "ok": False, "reason": "no_test_cases", "n_event_test_cases": 0, "n_far_test_cases": 0}
    strict_cov_test = eval_mod._strict_ib_coverage(
        samples=samples,
        case_ids=test_cases,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    strict_cov_train = eval_mod._strict_ib_coverage(
        samples=samples,
        case_ids=train_cases,
        time_basis=str(strict_time_basis),
        use_flags=bool(strict_use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(strict_far_min_row_quality_frac),
        min_far_kinematic_clean_fraction=float(strict_far_min_kinematic_clean_frac),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    ok_test, reasons_test = eval_mod._strict_coverage_ok(
        coverage=strict_cov_test,
        min_event_total=int(min_event_cases),
        min_far_total=int(min_far_cases),
        min_event_per_lead=int(min_event_per_lead),
        min_far_per_lead=int(min_far_per_lead),
    )
    ok_train, reasons_train = eval_mod._strict_coverage_ok(
        coverage=strict_cov_train,
        min_event_total=int(min_event_train_cases),
        min_far_total=int(min_far_train_cases),
        min_event_per_lead=int(min_event_train_per_lead),
        min_far_per_lead=int(min_far_train_per_lead),
    )
    reasons = [f"test:{v}" for v in reasons_test] + [f"train:{v}" for v in reasons_train]
    ok = bool(ok_test and ok_train)
    reason = "ok" if ok else "insufficient_strict_ib_coverage:" + ";".join([str(v) for v in reasons])
    return {
        "available": bool(strict_cov_test.get("available", True) and strict_cov_train.get("available", True)),
        "ok": ok,
        "reason": reason,
        "n_event_train_cases": int(strict_cov_train.get("event_total", 0)),
        "n_far_train_cases": int(strict_cov_train.get("far_total", 0)),
        "n_event_test_cases": int(strict_cov_test.get("event_total", 0)),
        "n_far_test_cases": int(strict_cov_test.get("far_total", 0)),
        "n_train_cases": int(len(train_cases)),
        "n_test_cases": int(len(test_cases)),
        "event_train_by_lead": strict_cov_train.get("event_by_lead", {}),
        "far_train_by_lead": strict_cov_train.get("far_by_lead", {}),
        "event_by_lead": strict_cov_test.get("event_by_lead", {}),
        "far_by_lead": strict_cov_test.get("far_by_lead", {}),
        "event_col_used": strict_cov_test.get("event_col_used"),
        "far_col_used": strict_cov_test.get("far_col_used"),
        "time_basis": strict_cov_test.get("time_basis"),
        "use_flags": bool(strict_cov_test.get("use_flags", strict_use_flags)),
        "far_quality_tags": list(strict_cov_test.get("far_quality_tags", list(far_quality_tags))),
    }


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = _split_dates(args)
    if not dates:
        raise ValueError("No split dates resolved for sweep")
    dataset_samples = pd.read_parquet(Path(args.dataset) / "samples.parquet")
    eval_mod = _load_eval_module()
    quality_tags = _parse_quality_tags(args.claim_far_quality_tags)

    rows: list[dict[str, Any]] = []
    successes = 0
    for i, split_date in enumerate(dates):
        ib_cov = _precheck_ibtracs_coverage(
            samples=dataset_samples,
            eval_mod=eval_mod,
            split_date=str(split_date),
            time_buffer_hours=float(args.time_buffer_hours),
            min_event_cases=int(args.min_strict_event_test_cases),
            min_far_cases=int(args.min_strict_far_test_cases),
            min_event_train_cases=int(args.min_strict_event_train_cases),
            min_far_train_cases=int(args.min_strict_far_train_cases),
            min_train_cases=int(args.min_train_cases),
            min_test_cases=int(args.min_test_cases),
            min_event_per_lead=int(args.min_strict_event_test_per_lead),
            min_far_per_lead=int(args.min_strict_far_test_per_lead),
            min_event_train_per_lead=int(args.min_strict_event_train_per_lead),
            min_far_train_per_lead=int(args.min_strict_far_train_per_lead),
            strict_time_basis=str(args.strict_coverage_time_basis),
            strict_use_flags=bool(args.strict_coverage_use_flags),
            far_quality_tags=tuple(quality_tags),
            strict_far_min_row_quality_frac=float(args.strict_far_min_row_quality_frac),
            strict_far_min_kinematic_clean_frac=float(args.strict_far_min_kinematic_clean_frac),
            strict_far_min_any_storm_km=float(args.strict_far_min_any_storm_km),
            strict_far_min_nearest_storm_km=float(args.strict_far_min_nearest_storm_km),
        )
        if bool(ib_cov.get("available", False)) and (not bool(ib_cov.get("ok", False))):
            reason = str(ib_cov.get("reason", "insufficient_strict_ib_coverage"))
            status = "skipped_insufficient_ibtracs_coverage"
            if reason.startswith("insufficient_train_cases") or reason.startswith("insufficient_test_cases"):
                status = "skipped_insufficient_split_cases"
            rows.append(
                {
                    "split_date": str(split_date),
                    "return_code": 0,
                    "status": status,
                    "evaluation_path": None,
                    "all_gates_pass": False,
                    "n_train_cases": int(ib_cov.get("n_train_cases", 0)),
                    "n_test_cases": int(ib_cov.get("n_test_cases", 0)),
                    "ib_event_train_cases": int(ib_cov.get("n_event_train_cases", 0)),
                    "ib_far_train_cases": int(ib_cov.get("n_far_train_cases", 0)),
                    "ib_event_test_cases": int(ib_cov.get("n_event_test_cases", 0)),
                    "ib_far_test_cases": int(ib_cov.get("n_far_test_cases", 0)),
                    "ib_event_train_by_lead": ib_cov.get("event_train_by_lead", {}),
                    "ib_far_train_by_lead": ib_cov.get("far_train_by_lead", {}),
                    "ib_event_test_by_lead": ib_cov.get("event_by_lead", {}),
                    "ib_far_test_by_lead": ib_cov.get("far_by_lead", {}),
                    "strict_event_col_used": ib_cov.get("event_col_used"),
                    "strict_far_col_used": ib_cov.get("far_col_used"),
                    "strict_time_basis": ib_cov.get("time_basis", str(args.strict_coverage_time_basis)),
                    "strict_use_flags": bool(ib_cov.get("use_flags", args.strict_coverage_use_flags)),
                    "claim_far_quality_tags": list(ib_cov.get("far_quality_tags", list(quality_tags))),
                    "ibtracs_reason_codes": reason,
                }
            )
            continue
        eval_path = outdir / f"evaluation_{split_date}.json"
        cmd = [
            str(args.python_exe),
            str((Path(__file__).resolve().parent / "evaluate_minipilot.py")),
            "--dataset",
            str(args.dataset),
            "--config",
            str(args.config),
            "--out",
            str(eval_path),
            "--split-date",
            str(split_date),
            "--time-buffer-hours",
            str(float(args.time_buffer_hours)),
            "--min-train-cases",
            str(int(args.min_train_cases)),
            "--min-test-cases",
            str(int(args.min_test_cases)),
            "--strict-coverage-time-basis",
            str(args.strict_coverage_time_basis),
            "--min-strict-event-test-cases",
            str(int(args.min_strict_event_test_cases)),
            "--min-strict-far-test-cases",
            str(int(args.min_strict_far_test_cases)),
            "--min-strict-event-train-cases",
            str(int(args.min_strict_event_train_cases)),
            "--min-strict-far-train-cases",
            str(int(args.min_strict_far_train_cases)),
            "--min-strict-event-test-per-lead",
            str(int(args.min_strict_event_test_per_lead)),
            "--min-strict-far-test-per-lead",
            str(int(args.min_strict_far_test_per_lead)),
            "--min-strict-event-train-per-lead",
            str(int(args.min_strict_event_train_per_lead)),
            "--min-strict-far-train-per-lead",
            str(int(args.min_strict_far_train_per_lead)),
            "--seed",
            str(int(args.seed) + i),
        ]
        if bool(args.strict_coverage_use_flags):
            cmd.append("--strict-coverage-use-flags")
        else:
            cmd.append("--strict-coverage-no-flags")
        if quality_tags:
            cmd.append("--claim-far-quality-tags")
            cmd.extend([str(v) for v in quality_tags])
        cmd.extend(
            [
                "--strict-far-min-row-quality-frac",
                str(float(args.strict_far_min_row_quality_frac)),
                "--strict-far-min-kinematic-clean-frac",
                str(float(args.strict_far_min_kinematic_clean_frac)),
                "--strict-far-min-any-storm-km",
                str(float(args.strict_far_min_any_storm_km)),
                "--strict-far-min-nearest-storm-km",
                str(float(args.strict_far_min_nearest_storm_km)),
            ]
        )
        for token in args.eval_extra_arg:
            if token:
                cmd.append(str(token))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        row: dict[str, Any] = {
            "split_date": str(split_date),
            "return_code": int(proc.returncode),
            "evaluation_path": str(eval_path.resolve()),
        }
        if proc.returncode != 0 or (not eval_path.exists()):
            row["status"] = "failed"
            row["stderr_tail"] = "\n".join((proc.stderr or "").splitlines()[-20:])
            rows.append(row)
            continue

        obj = _load_eval(eval_path)
        successes += 1
        parity_gate = obj.get("parity_confound_gate", {}) or {}
        anomaly_gate = obj.get("anomaly_mode_gate", {}) or {}
        geom_gate = obj.get("geometry_null_collapse", {}) or {}
        angular_gate = (obj.get("angular_witness", {}) or {}).get("gate", {}) or {}
        ibtracs_gate = obj.get("ibtracs_alignment_gate", {}) or {}
        diag = obj.get("interpretability", {}) or {}
        strict_cov_eval = obj.get("strict_test_coverage", {}) or {}
        obs_ang = (obj.get("angular_witness", {}) or {}).get("observed", {}) or {}
        row.update(
            {
                "status": "ok",
                "all_gates_pass": bool(
                    parity_gate.get("passed", False)
                    and anomaly_gate.get("passed", False)
                    and geom_gate.get("passed", False)
                    and angular_gate.get("passed", False)
                    and ibtracs_gate.get("passed", False)
                    and (not bool(diag.get("diagnostic_only", True)))
                ),
                "n_train_cases": int(ib_cov.get("n_train_cases", 0)),
                "n_test_cases": int(ib_cov.get("n_test_cases", 0)),
                "ib_event_train_cases": int(ib_cov.get("n_event_train_cases", 0)),
                "ib_far_train_cases": int(ib_cov.get("n_far_train_cases", 0)),
                "ib_event_test_cases": int(strict_cov_eval.get("event_total", ib_cov.get("n_event_test_cases", 0))),
                "ib_far_test_cases": int(strict_cov_eval.get("far_total", ib_cov.get("n_far_test_cases", 0))),
                "ib_event_train_by_lead": ib_cov.get("event_train_by_lead", {}),
                "ib_far_train_by_lead": ib_cov.get("far_train_by_lead", {}),
                "ib_event_test_by_lead": strict_cov_eval.get("event_by_lead", ib_cov.get("event_by_lead", {})),
                "ib_far_test_by_lead": strict_cov_eval.get("far_by_lead", ib_cov.get("far_by_lead", {})),
                "parity_gate_pass": bool(parity_gate.get("passed", False)),
                "anomaly_gate_pass": bool(anomaly_gate.get("passed", False)),
                "geometry_gate_pass": bool(geom_gate.get("passed", False)),
                "angular_gate_pass": bool(angular_gate.get("passed", False)),
                "ibtracs_alignment_gate_pass": bool(ibtracs_gate.get("passed", False)),
                "diagnostic_only": bool(diag.get("diagnostic_only", True)),
                "event_minus_far": _to_float(parity_gate.get("event_minus_far_rate")),
                "far_rate": _to_float(parity_gate.get("far_nonstorm_parity_rate")),
                "confound_rate": _to_float(parity_gate.get("confound_rate")),
                "angular_margin": _to_float(obs_ang.get("margin")),
                "angular_d": _to_float(obs_ang.get("effect_size_d")),
                "angular_p": _to_float(obs_ang.get("p_value_perm")),
                "ibtracs_event_minus_far": _to_float(ibtracs_gate.get("event_minus_far_parity_rate")),
                "ibtracs_confound_rate": _to_float(ibtracs_gate.get("confound_rate")),
                "ibtracs_reason_codes": ";".join([str(v) for v in (ibtracs_gate.get("reason_codes") or [])]),
                "strict_event_col_used": strict_cov_eval.get("event_col_used", ib_cov.get("event_col_used")),
                "strict_far_col_used": strict_cov_eval.get("far_col_used", ib_cov.get("far_col_used")),
                "strict_time_basis": strict_cov_eval.get("time_basis", ib_cov.get("time_basis", str(args.strict_coverage_time_basis))),
                "strict_use_flags": bool(strict_cov_eval.get("use_flags", ib_cov.get("use_flags", args.strict_coverage_use_flags))),
                "claim_far_quality_tags": list(strict_cov_eval.get("far_quality_tags", ib_cov.get("far_quality_tags", list(quality_tags)))),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df_write = _serialize_nested_columns(df)
    parquet_path = outdir / "split_sweep.parquet"
    csv_path = outdir / "split_sweep.csv"
    df_write.to_parquet(parquet_path, index=False)
    df_write.to_csv(csv_path, index=False)

    n_total = int(df.shape[0])
    n_ok = int((df.get("status") == "ok").sum()) if "status" in df.columns else 0
    n_skipped = int((df.get("status") == "skipped_insufficient_ibtracs_coverage").sum()) if "status" in df.columns else 0
    n_skipped_split = int((df.get("status") == "skipped_insufficient_split_cases").sum()) if "status" in df.columns else 0
    n_all_pass = int((df.get("all_gates_pass") == True).sum()) if "all_gates_pass" in df.columns else 0
    pass_rate = float(n_all_pass / max(1, n_ok))
    summary_lines = [
        "# Split Sweep Summary",
        "",
        f"- Total split dates: `{n_total}`",
        f"- Successful evaluations: `{n_ok}`",
        f"- Skipped (insufficient IBTrACS coverage): `{n_skipped}`",
        f"- Skipped (insufficient split train/test cases): `{n_skipped_split}`",
        f"- All-gates pass count: `{n_all_pass}`",
        f"- All-gates pass rate over successful runs: `{pass_rate:.3f}`",
        f"- Stability target met (`>=0.70`): `{pass_rate >= 0.70}`",
        "",
        "## Targets",
        "- Stability target: `all_gates_pass_rate >= 0.70`",
        "",
        "## Strict Coverage Contract",
        f"- `strict_time_basis={str(args.strict_coverage_time_basis)}`",
        f"- `strict_use_flags={bool(args.strict_coverage_use_flags)}`",
        f"- `strict_far_min_row_quality_frac={float(args.strict_far_min_row_quality_frac)}`",
        f"- `strict_far_min_kinematic_clean_frac={float(args.strict_far_min_kinematic_clean_frac)}`",
        f"- `strict_far_min_any_storm_km={float(args.strict_far_min_any_storm_km)}`",
        f"- `strict_far_min_nearest_storm_km={float(args.strict_far_min_nearest_storm_km)}`",
        f"- `min_strict_event_test_cases={int(args.min_strict_event_test_cases)}`",
        f"- `min_strict_far_test_cases={int(args.min_strict_far_test_cases)}`",
        f"- `min_strict_event_train_cases={int(args.min_strict_event_train_cases)}`",
        f"- `min_strict_far_train_cases={int(args.min_strict_far_train_cases)}`",
        f"- `min_strict_event_test_per_lead={int(args.min_strict_event_test_per_lead)}`",
        f"- `min_strict_far_test_per_lead={int(args.min_strict_far_test_per_lead)}`",
        f"- `min_strict_event_train_per_lead={int(args.min_strict_event_train_per_lead)}`",
        f"- `min_strict_far_train_per_lead={int(args.min_strict_far_train_per_lead)}`",
        f"- `claim_far_quality_tags={list(quality_tags)}`",
        "",
        "## Outputs",
        f"- `{parquet_path}`",
        f"- `{csv_path}`",
    ]
    md_path = outdir / "split_sweep_summary.md"
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote split sweep to {parquet_path}")
    print(f"Wrote split sweep summary to {md_path}")
    if successes <= 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
