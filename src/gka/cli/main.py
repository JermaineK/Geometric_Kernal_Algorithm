"""Main CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from gka.cli import calibrate, diagnose, prepare, report, run, schema, validate
from gka.core.logging import setup_logging
from gka.domains import register_builtin_adapters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gka", description="GKA-II modular diagnostics toolkit")
    subparsers = parser.add_subparsers(dest="command")

    prepare.register(subparsers)
    validate.register(subparsers)
    run.register(subparsers)
    report.register(subparsers)
    diagnose.register(subparsers)
    calibrate.register(subparsers)
    schema.register(subparsers)

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    register_builtin_adapters()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
