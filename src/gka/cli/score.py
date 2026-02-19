"""Implementation of `gka score` with frozen calibration thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

from gka.calibration.score import score_parameter_runs, write_score_report


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("score", help="Score parameter runs with frozen calibration")
    parser.add_argument("--parameter-runs", required=True, help="Path to parameter_runs.json")
    parser.add_argument("--calibration", required=True, help="Path to calibration.json")
    parser.add_argument("--out", default=None, help="Optional output path for score JSON")
    parser.set_defaults(func=cmd_score)


def cmd_score(args: argparse.Namespace) -> int:
    payload = score_parameter_runs(
        parameter_runs_path=args.parameter_runs,
        calibration_path=args.calibration,
    )
    if args.out:
        write_score_report(payload, args.out)
        print(f"Score report written to {Path(args.out)}")
    print(
        "score metrics: "
        f"fp_rate={payload['metrics']['false_positive_rate']:.4f} "
        f"fn_rate={payload['metrics']['false_negative_rate']:.4f} "
        f"f1={payload['metrics']['f1']:.4f}"
    )
    return 0
