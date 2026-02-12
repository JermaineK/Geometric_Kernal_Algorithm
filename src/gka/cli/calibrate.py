"""Implementation of `gka calibrate`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("calibrate", help="Run synthetic calibration and derive thresholds")
    parser.add_argument("--suite", default="synthetic", choices=["synthetic"], help="Calibration suite")
    parser.add_argument("--runs", type=int, default=200, help="Monte Carlo runs per synthetic test")
    parser.add_argument("--configs", nargs="+", default=["tests/synthetic/configs/*.yaml"], help="Config paths or globs")
    parser.add_argument("--outroot", default="tests/synthetic/outputs", help="Synthetic output root")
    parser.add_argument("--seed", type=int, default=12345, help="Suite random seed")
    parser.set_defaults(func=cmd_calibrate)


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.suite != "synthetic":
        raise ValueError(f"Unsupported suite '{args.suite}'. Only synthetic is currently implemented.")

    repo_root = Path(__file__).resolve().parents[3]
    suite_script = repo_root / "tests" / "synthetic" / "run_synthetic_suite.py"
    if not suite_script.exists():
        raise FileNotFoundError(f"Synthetic suite runner not found: {suite_script}")

    cmd = [
        sys.executable,
        str(suite_script),
        "--configs",
        *args.configs,
        "--runs",
        str(args.runs),
        "--outroot",
        str(args.outroot),
        "--seed",
        str(args.seed),
    ]
    env = None
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(repo_root))
    if proc.returncode != 0:
        raise RuntimeError(
            "Synthetic calibration run failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    outroot = Path(args.outroot)
    suite_path = outroot / "suite_results.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"Missing suite results at {suite_path}")

    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    recommended = _recommend_thresholds(payload)
    rec_path = outroot / "recommended_thresholds.yaml"
    rec_path.write_text(yaml.safe_dump(recommended, sort_keys=False), encoding="utf-8")

    print(proc.stdout.strip())
    print(f"Calibration suite results: {suite_path}")
    print(f"Recommended thresholds: {rec_path}")
    return 0


def _recommend_thresholds(payload: dict[str, Any]) -> dict[str, Any]:
    tests = payload.get("tests", {})
    rec: dict[str, Any] = {"generated_from": "gka calibrate", "tests": {}}
    for test_name, test_info in tests.items():
        cal = test_info.get("calibration", {})
        if not isinstance(cal, dict) or not cal:
            continue
        test_rec: dict[str, Any] = {}
        for key, value in cal.items():
            if not isinstance(value, (int, float)):
                continue
            if key.endswith("_p50"):
                test_rec[f"{key}_max"] = float(value * 1.15)
            elif key.endswith("_p90"):
                test_rec[f"{key}_max"] = float(value * 1.15)
            elif key.endswith("_cat_rate"):
                test_rec[f"{key}_max"] = float(min(1.0, value * 1.2 + 0.02))
            elif key.endswith("_accuracy"):
                test_rec[f"{key}_min"] = float(max(0.0, value * 0.95))
        rec["tests"][test_name] = test_rec
    return rec
