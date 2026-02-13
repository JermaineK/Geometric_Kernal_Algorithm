"""Implementation of `gka run`."""

from __future__ import annotations

import argparse
import sys

from gka.core.pipeline import run_pipeline


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run GKA pipeline")
    parser.add_argument("dataset", help="Dataset folder")
    parser.add_argument("--domain", default=None, help="Domain adapter name")
    parser.add_argument("--config", default=None, help="Config YAML")
    parser.add_argument("--out", required=True, help="Output results folder")
    parser.add_argument("--null", type=int, default=None, help="Override null replication count")
    parser.add_argument("--allow-missing", action="store_true", help="Allow incomplete pairs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="Write intermediate diagnostics (series, candidates, scaling points, stability curves)",
    )
    parser.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> int:
    result = run_pipeline(
        dataset_path=args.dataset,
        domain=args.domain,
        out_dir=args.out,
        config_path=args.config,
        null_n=args.null,
        allow_missing=args.allow_missing,
        seed=args.seed,
        dump_intermediates=bool(args.dump_intermediates),
        argv=sys.argv,
    )

    print(f"Wrote {len(result.summary)} rows to {args.out}/results.parquet")
    print(f"Wrote run metadata to {args.out}/run_metadata.json")
    print(f"Wrote resolved config to {args.out}/config_resolved.yaml")
    if result.null_summary is not None:
        print(f"Wrote null distribution rows: {len(result.null_summary)}")
    return 0
