"""Implementation of `gka report`."""

from __future__ import annotations

import argparse
from pathlib import Path

from gka.viz.reports import generate_html_report


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Build HTML report from run outputs")
    parser.add_argument("results", nargs="?", help="Results directory containing results.parquet")
    parser.add_argument("--in", dest="results_in", default=None, help="Results directory containing results.parquet")
    parser.add_argument("--out", required=True, help="Output html path")
    parser.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> int:
    results_arg = args.results_in or args.results
    if not results_arg:
        raise ValueError("Provide results directory as positional argument or with --in")
    results_dir = Path(results_arg)
    out_file = Path(args.out)
    generate_html_report(results_dir=results_dir, out_html=out_file)
    print(f"Report written to {out_file}")
    return 0
