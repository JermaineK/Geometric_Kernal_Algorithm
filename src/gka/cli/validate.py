"""Implementation of `gka validate`."""

from __future__ import annotations

import argparse
import json

from gka.data.validators import report_to_dict, validate_dataset


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate", help="Validate canonical dataset schema")
    parser.add_argument("dataset", help="Dataset folder")
    parser.add_argument("--allow-missing", action="store_true", help="Allow incomplete pairs")
    parser.add_argument("--json", action="store_true", help="Print report as JSON")
    parser.set_defaults(func=cmd_validate)


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_dataset(args.dataset, allow_missing=args.allow_missing)
    payload = report_to_dict(report)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.valid else "FAIL"
        print(f"Validation: {status}")
        print(f"Pairing completeness: {report.pairing_completeness:.3f}")
        if not report.issues:
            print("No issues found")
        for issue in report.issues:
            print(f"- {issue.level.upper()} [{issue.code}] {issue.message}")

    return 0 if report.valid else 2
