from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run evaluator twice (calibration OFF/ON) and emit a compact A/B diff."
    )
    parser.add_argument("--dataset", required=True, help="Tile dataset directory")
    parser.add_argument("--config", required=True, help="Pipeline config path")
    parser.add_argument("--split-date", required=True, help="Split date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--time-buffer-hours", type=float, default=24.0, help="Split buffer in hours")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--eval-script",
        default=str(Path(__file__).resolve().parent / "evaluate_minipilot.py"),
        help="Path to evaluate_minipilot.py",
    )
    parser.add_argument("--base-name", default="calibration_ab", help="Base name for generated files")
    args, extra = parser.parse_known_args()
    return args, extra


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, check=False, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    parity = payload.get("parity_confound_gate", {}) or {}
    geometry = payload.get("geometry_null_collapse", {}) or {}
    witness = (payload.get("angular_witness", {}) or {}).get("gate", {}) if isinstance(payload.get("angular_witness"), dict) else {}
    calib = payload.get("parity_threshold_calibration", {}) or {}
    return {
        "interpretability_diagnostic_only": bool((payload.get("interpretability", {}) or {}).get("diagnostic_only", True)),
        "claim_mode": payload.get("claim_mode"),
        "claim_reason_codes": payload.get("claim_reason_codes", []),
        "parity_passed": bool(parity.get("passed", False)),
        "parity_event_rate": parity.get("event_parity_rate"),
        "parity_far_rate": parity.get("far_nonstorm_parity_rate"),
        "parity_margin": parity.get("event_minus_far_rate"),
        "parity_confound": parity.get("confound_rate"),
        "geometry_passed": bool(geometry.get("passed", False)),
        "geometry_failing_required": [
            k
            for k, v in (geometry.get("checks", {}) or {}).items()
            if isinstance(v, dict) and bool(v.get("required", False)) and not bool(v.get("passed", False))
        ],
        "angular_passed": bool(witness.get("passed", False)),
        "angular_reason": witness.get("reason"),
        "calibration_available": bool(calib.get("available", False)),
        "calibration_reason": calib.get("reason"),
        "calibration_thresholds": calib.get("thresholds", {}),
    }


def main() -> int:
    args, extra_args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    eval_script = Path(args.eval_script).resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"evaluate script not found: {eval_script}")

    env = os.environ.copy()
    repo_root_str = str(repo_root)
    src_root_str = str(src_root)
    existing_pythonpath = env.get("PYTHONPATH", "")
    prefix = src_root_str + os.pathsep + repo_root_str
    if existing_pythonpath:
        env["PYTHONPATH"] = prefix + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = prefix

    out_off = out_dir / f"{args.base_name}_off.json"
    out_on = out_dir / f"{args.base_name}_on.json"

    common = [
        str(args.python_exe),
        str(eval_script),
        "--dataset",
        str(Path(args.dataset)),
        "--config",
        str(Path(args.config)),
        "--split-date",
        str(args.split_date),
        "--time-buffer-hours",
        str(float(args.time_buffer_hours)),
    ]
    # Place extra args before explicit calibration flag so ON/OFF mode wins deterministically.
    cmd_off = [*common, *extra_args, "--out", str(out_off), "--no-calibrate-parity-thresholds"]
    cmd_on = [*common, *extra_args, "--out", str(out_on), "--calibrate-parity-thresholds"]

    _run_cmd(cmd_off, cwd=repo_root, env=env)
    _run_cmd(cmd_on, cwd=repo_root, env=env)

    off_payload = _load_json(out_off)
    on_payload = _load_json(out_on)
    off_metrics = _extract_metrics(off_payload)
    on_metrics = _extract_metrics(on_payload)

    diff = {
        "off_path": str(out_off.resolve()),
        "on_path": str(out_on.resolve()),
        "off": off_metrics,
        "on": on_metrics,
        "delta": {
            "parity_margin": (
                (on_metrics.get("parity_margin") or 0.0) - (off_metrics.get("parity_margin") or 0.0)
            ),
            "parity_event_rate": (
                (on_metrics.get("parity_event_rate") or 0.0) - (off_metrics.get("parity_event_rate") or 0.0)
            ),
            "parity_far_rate": (
                (on_metrics.get("parity_far_rate") or 0.0) - (off_metrics.get("parity_far_rate") or 0.0)
            ),
        },
    }
    diff_path = out_dir / f"{args.base_name}_diff.json"
    diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Calibration A/B Diff",
        "",
        f"- OFF: `{out_off}`",
        f"- ON: `{out_on}`",
        f"- Diff: `{diff_path}`",
        "",
        "## OFF",
        f"- parity_passed: `{off_metrics['parity_passed']}`",
        f"- parity_margin: `{off_metrics['parity_margin']}`",
        f"- parity_event_rate: `{off_metrics['parity_event_rate']}`",
        f"- parity_far_rate: `{off_metrics['parity_far_rate']}`",
        f"- calibration_available: `{off_metrics['calibration_available']}` ({off_metrics['calibration_reason']})",
        "",
        "## ON",
        f"- parity_passed: `{on_metrics['parity_passed']}`",
        f"- parity_margin: `{on_metrics['parity_margin']}`",
        f"- parity_event_rate: `{on_metrics['parity_event_rate']}`",
        f"- parity_far_rate: `{on_metrics['parity_far_rate']}`",
        f"- calibration_available: `{on_metrics['calibration_available']}` ({on_metrics['calibration_reason']})",
        "",
        "## Delta (ON - OFF)",
        f"- parity_margin: `{diff['delta']['parity_margin']}`",
        f"- parity_event_rate: `{diff['delta']['parity_event_rate']}`",
        f"- parity_far_rate: `{diff['delta']['parity_far_rate']}`",
    ]
    report_path = out_dir / f"{args.base_name}_diff.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote calibration A/B diff to {diff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
