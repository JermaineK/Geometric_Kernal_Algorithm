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
        type=int,
        default=10,
        help="Skip split when test-set IBTrACS event case coverage is below this threshold",
    )
    parser.add_argument(
        "--min-ibtracs-far-test-cases",
        type=int,
        default=10,
        help="Skip split when test-set IBTrACS far case coverage is below this threshold",
    )
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
    mod_path = Path("examples/weather_real_minipilot/evaluate_minipilot.py")
    spec = importlib.util.spec_from_file_location("eval_minipilot_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _precheck_ibtracs_coverage(
    *,
    samples: pd.DataFrame,
    eval_mod: Any,
    split_date: str,
    time_buffer_hours: float,
    min_event_cases: int,
    min_far_cases: int,
) -> dict[str, Any]:
    if samples.empty:
        return {"available": False, "ok": False, "reason": "empty_samples", "n_event_test_cases": 0, "n_far_test_cases": 0}
    if ("ib_event" not in samples.columns) or ("ib_far" not in samples.columns):
        return {"available": False, "ok": True, "reason": "ibtracs_columns_missing", "n_event_test_cases": 0, "n_far_test_cases": 0}
    storm_col = eval_mod._resolve_storm_id_col(samples, requested="auto")
    case_meta = eval_mod._build_case_meta(samples, storm_id_col=storm_col)
    split = eval_mod._blocked_time_split(
        case_meta=case_meta,
        split_date=pd.Timestamp(split_date),
        buffer_hours=float(time_buffer_hours),
    )
    test_cases = set(split.get("test_cases", set()))
    if not test_cases:
        return {"available": True, "ok": False, "reason": "no_test_cases", "n_event_test_cases": 0, "n_far_test_cases": 0}
    sub = samples.loc[samples["case_id"].astype(str).isin(test_cases)].copy()
    if sub.empty:
        return {"available": True, "ok": False, "reason": "no_test_rows", "n_event_test_cases": 0, "n_far_test_cases": 0}
    event_cases = int(sub.loc[pd.to_numeric(sub["ib_event"], errors="coerce").fillna(0) > 0, "case_id"].astype(str).nunique())
    far_cases = int(sub.loc[pd.to_numeric(sub["ib_far"], errors="coerce").fillna(0) > 0, "case_id"].astype(str).nunique())
    ok = bool((event_cases >= int(min_event_cases)) and (far_cases >= int(min_far_cases)))
    reason = "ok" if ok else "insufficient_ibtracs_coverage"
    return {
        "available": True,
        "ok": ok,
        "reason": reason,
        "n_event_test_cases": int(event_cases),
        "n_far_test_cases": int(far_cases),
        "n_test_cases": int(len(test_cases)),
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

    rows: list[dict[str, Any]] = []
    successes = 0
    for i, split_date in enumerate(dates):
        ib_cov = _precheck_ibtracs_coverage(
            samples=dataset_samples,
            eval_mod=eval_mod,
            split_date=str(split_date),
            time_buffer_hours=float(args.time_buffer_hours),
            min_event_cases=int(args.min_ibtracs_event_test_cases),
            min_far_cases=int(args.min_ibtracs_far_test_cases),
        )
        if bool(ib_cov.get("available", False)) and (not bool(ib_cov.get("ok", False))):
            rows.append(
                {
                    "split_date": str(split_date),
                    "return_code": 0,
                    "status": "skipped_insufficient_ibtracs_coverage",
                    "evaluation_path": None,
                    "all_gates_pass": False,
                    "ib_event_test_cases": int(ib_cov.get("n_event_test_cases", 0)),
                    "ib_far_test_cases": int(ib_cov.get("n_far_test_cases", 0)),
                    "ibtracs_reason_codes": str(ib_cov.get("reason", "insufficient_ibtracs_coverage")),
                }
            )
            continue
        eval_path = outdir / f"evaluation_{split_date}.json"
        cmd = [
            str(args.python_exe),
            "examples/weather_real_minipilot/evaluate_minipilot.py",
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
            "--seed",
            str(int(args.seed) + i),
        ]
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
                "ib_event_test_cases": int(ib_cov.get("n_event_test_cases", 0)),
                "ib_far_test_cases": int(ib_cov.get("n_far_test_cases", 0)),
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
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = outdir / "split_sweep.parquet"
    csv_path = outdir / "split_sweep.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    n_total = int(df.shape[0])
    n_ok = int((df.get("status") == "ok").sum()) if "status" in df.columns else 0
    n_skipped = int((df.get("status") == "skipped_insufficient_ibtracs_coverage").sum()) if "status" in df.columns else 0
    n_all_pass = int((df.get("all_gates_pass") == True).sum()) if "all_gates_pass" in df.columns else 0
    pass_rate = float(n_all_pass / max(1, n_ok))
    summary_lines = [
        "# Split Sweep Summary",
        "",
        f"- Total split dates: `{n_total}`",
        f"- Successful evaluations: `{n_ok}`",
        f"- Skipped (insufficient IBTrACS coverage): `{n_skipped}`",
        f"- All-gates pass count: `{n_all_pass}`",
        f"- All-gates pass rate over successful runs: `{pass_rate:.3f}`",
        f"- Stability target met (`>=0.70`): `{pass_rate >= 0.70}`",
        "",
        "## Targets",
        "- Stability target: `all_gates_pass_rate >= 0.70`",
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
