"""Implementation of `gka calibrate`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gka.calibration.fit import fit_calibration_from_parameter_runs, write_calibration
from gka.calibration.score import score_parameter_runs, write_score_report
from gka.calibration.schema import CALIBRATION_SCHEMA_VERSION, validate_calibration_payload
from gka.utils.hash import file_sha256


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("calibrate", help="Run synthetic calibration and derive thresholds")
    parser.add_argument("--suite", default="synthetic", choices=["synthetic", "stress"], help="Calibration suite")
    parser.add_argument("--runs", type=int, default=200, help="Monte Carlo runs per test")
    parser.add_argument("--configs", nargs="+", default=None, help="Config paths or globs")
    parser.add_argument("--outroot", default=None, help="Suite output root")
    parser.add_argument("--seed", type=int, default=12345, help="Suite random seed")
    parser.add_argument("--parameter-runs", default=None, help="Path to frozen parameter_runs.json")
    parser.add_argument("--fp-max", type=float, default=0.10, help="Maximum false positive rate for threshold fitting")
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta objective for threshold fitting")
    parser.add_argument("--calibration-out", default=None, help="Calibration JSON output path")
    parser.add_argument(
        "--score-after-calibrate",
        action="store_true",
        help="Run scoring immediately after calibration (requires --allow-calibrate-and-score)",
    )
    parser.add_argument(
        "--allow-calibrate-and-score",
        action="store_true",
        help="Explicitly allow calibrate+score in one invocation",
    )
    parser.add_argument("--score-out", default=None, help="Score report JSON output path")
    parser.set_defaults(func=cmd_calibrate)


def cmd_calibrate(args: argparse.Namespace) -> int:
    if bool(args.score_after_calibrate) and not bool(args.allow_calibrate_and_score):
        raise ValueError(
            "calibrate+score is disabled by default. "
            "Re-run with --allow-calibrate-and-score to enable both in one command."
        )

    repo_root = Path(__file__).resolve().parents[3]
    suite_outroot = _default_outroot(args.suite, args.outroot)

    parameter_runs_path: Path | None = None
    if args.parameter_runs:
        parameter_runs_path = Path(args.parameter_runs)
    else:
        suite_stdout = _run_suite(
            repo_root=repo_root,
            suite=args.suite,
            runs=int(args.runs),
            configs=args.configs,
            outroot=suite_outroot,
            seed=int(args.seed),
        )
        if suite_stdout:
            print(suite_stdout.strip())
        parameter_runs_path = _resolve_parameter_runs_from_suite(args.suite, suite_outroot)

    if parameter_runs_path is not None and parameter_runs_path.exists():
        calibration_payload = fit_calibration_from_parameter_runs(
            parameter_runs_path=parameter_runs_path,
            target_fp_max=float(args.fp_max),
            objective_beta=float(args.beta),
        )
    else:
        suite_path = suite_outroot / "suite_results.json"
        if not suite_path.exists():
            raise FileNotFoundError(
                "Unable to build calibration: missing parameter_runs and suite_results.json"
            )
        suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
        thresholds = _recommend_thresholds(suite_payload)
        calibration_payload = validate_calibration_payload(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "generated_at_utc": suite_payload.get("timestamp_utc")
                or datetime.now(tz=timezone.utc).isoformat(),
                "source": {
                    "suite_results_path": str(suite_path.resolve()),
                    "suite_results_sha256": file_sha256(suite_path),
                },
                "objective": {
                    "false_positive_rate_max": float(args.fp_max),
                    "beta": float(args.beta),
                },
                "thresholds": thresholds,
            }
        )

    calibration_out = Path(args.calibration_out) if args.calibration_out else suite_outroot / "calibration.json"
    write_calibration(calibration_payload, calibration_out)
    print(f"Calibration written to {calibration_out}")

    if bool(args.score_after_calibrate):
        if parameter_runs_path is None or not parameter_runs_path.exists():
            raise FileNotFoundError(
                "score-after-calibrate requires parameter_runs.json. "
                "Provide --parameter-runs or run --suite stress."
            )
        score_payload = score_parameter_runs(
            parameter_runs_path=parameter_runs_path,
            calibration_path=calibration_out,
        )
        score_out = Path(args.score_out) if args.score_out else suite_outroot / "score_report.json"
        write_score_report(score_payload, score_out)
        print(f"Score report written to {score_out}")
        print(
            "score metrics: "
            f"fp_rate={score_payload['metrics']['false_positive_rate']:.4f} "
            f"fn_rate={score_payload['metrics']['false_negative_rate']:.4f} "
            f"f1={score_payload['metrics']['f1']:.4f}"
        )
    return 0


def _default_outroot(suite: str, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    if suite == "synthetic":
        return Path("tests/synthetic/outputs")
    if suite == "stress":
        return Path("tests/stress/outputs")
    raise ValueError(f"Unsupported suite '{suite}'")


def _run_suite(
    repo_root: Path,
    suite: str,
    runs: int,
    configs: list[str] | None,
    outroot: Path,
    seed: int,
) -> str:
    if suite == "synthetic":
        suite_script = repo_root / "tests" / "synthetic" / "run_synthetic_suite.py"
        default_configs = ["tests/synthetic/configs/*.yaml"]
    elif suite == "stress":
        suite_script = repo_root / "tests" / "stress" / "run_stress_suite.py"
        default_configs = ["tests/stress/configs/*.yaml"]
    else:
        raise ValueError(f"Unsupported suite '{suite}'")
    if not suite_script.exists():
        raise FileNotFoundError(f"Suite runner not found: {suite_script}")

    config_args = configs if configs else default_configs
    cmd = [
        sys.executable,
        str(suite_script),
        "--configs",
        *config_args,
        "--runs",
        str(runs),
        "--outroot",
        str(outroot),
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if proc.returncode != 0:
        raise RuntimeError(
            "Calibration suite run failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def _resolve_parameter_runs_from_suite(suite: str, outroot: Path) -> Path | None:
    if suite != "stress":
        return None
    robustness_report = outroot / "robustness_report.json"
    if not robustness_report.exists():
        return None
    payload = json.loads(robustness_report.read_text(encoding="utf-8"))
    runs_path = payload.get("parameter_robustness", {}).get("runs_path")
    if isinstance(runs_path, str) and runs_path:
        path = Path(runs_path)
        return path if path.exists() else outroot / Path(runs_path).name
    return None


def _recommend_thresholds(payload: dict[str, Any]) -> dict[str, Any]:
    tests = payload.get("tests", {})
    defaults = {
        "knee_p_min": 0.35,
        "knee_strength_min": 0.5,
        "knee_delta_bic_min": 8.0,
        "middle_score_band": {"forbidden_min": 0.6},
    }
    rec: dict[str, Any] = defaults.copy()
    tests_rec: dict[str, Any] = {}
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
        tests_rec[test_name] = test_rec
    if tests_rec:
        rec["tests"] = tests_rec
    return rec
