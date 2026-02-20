from __future__ import annotations

import argparse
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


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = _split_dates(args)
    if not dates:
        raise ValueError("No split dates resolved for sweep")

    rows: list[dict[str, Any]] = []
    successes = 0
    for i, split_date in enumerate(dates):
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
                    and (not bool(diag.get("diagnostic_only", True)))
                ),
                "parity_gate_pass": bool(parity_gate.get("passed", False)),
                "anomaly_gate_pass": bool(anomaly_gate.get("passed", False)),
                "geometry_gate_pass": bool(geom_gate.get("passed", False)),
                "angular_gate_pass": bool(angular_gate.get("passed", False)),
                "diagnostic_only": bool(diag.get("diagnostic_only", True)),
                "event_minus_far": _to_float(parity_gate.get("event_minus_far_rate")),
                "far_rate": _to_float(parity_gate.get("far_nonstorm_parity_rate")),
                "confound_rate": _to_float(parity_gate.get("confound_rate")),
                "angular_margin": _to_float(obs_ang.get("margin")),
                "angular_d": _to_float(obs_ang.get("effect_size_d")),
                "angular_p": _to_float(obs_ang.get("p_value_perm")),
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
    n_all_pass = int((df.get("all_gates_pass") == True).sum()) if "all_gates_pass" in df.columns else 0
    pass_rate = float(n_all_pass / max(1, n_ok))
    summary_lines = [
        "# Split Sweep Summary",
        "",
        f"- Total split dates: `{n_total}`",
        f"- Successful evaluations: `{n_ok}`",
        f"- All-gates pass count: `{n_all_pass}`",
        f"- All-gates pass rate over successful runs: `{pass_rate:.3f}`",
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
