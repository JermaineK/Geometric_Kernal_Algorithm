"""Implementation of `gka report`."""

from __future__ import annotations

import argparse
from pathlib import Path

from gka.reporting.json import build_report_payload, write_report_json
from gka.reporting.md import write_report_md
from gka.reporting.plots import write_report_figures
from gka.viz.reports import generate_html_report


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Build HTML report from run outputs")
    parser.add_argument("results", nargs="?", help="Results directory containing results.parquet")
    parser.add_argument("--in", dest="results_in", default=None, help="Results directory containing results.parquet")
    parser.add_argument("--out", default=None, help="Output html path")
    parser.add_argument("--out-json", default=None, help="Output report.json path")
    parser.add_argument("--out-md", default=None, help="Output report.md path")
    parser.add_argument("--figures-dir", default=None, help="Optional directory for exported figures")
    parser.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> int:
    results_arg = args.results_in or args.results
    if not results_arg:
        raise ValueError("Provide results directory as positional argument or with --in")
    results_dir = Path(results_arg)
    out_html = Path(args.out) if args.out else results_dir / "report.html"
    out_json = Path(args.out_json) if args.out_json else results_dir / "report.json"
    out_md = Path(args.out_md) if args.out_md else results_dir / "report.md"
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"

    payload = build_report_payload(results_dir=results_dir)
    write_report_json(payload, out_json)
    write_report_md(payload, out_md)
    figure_paths = write_report_figures(results_dir=results_dir, out_dir=figures_dir)
    generate_html_report(results_dir=results_dir, out_html=out_html)

    print(f"Report written to {out_html}")
    print(f"Report JSON written to {out_json}")
    print(f"Report Markdown written to {out_md}")
    print(f"Figures written to {figures_dir} ({', '.join(figure_paths.keys())})")
    return 0
