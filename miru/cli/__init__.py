"""Command-line entry point for Miru utilities.

Subcommands
-----------

``miru record list``    — list recorded trace files in ``MIRU_RECORD_PATH``
``miru record export``  — concatenate recorded traces into a single
                          JSONL or CSV file
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from miru.cli.record import run_export, run_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miru",
        description="Miru — multimodal reasoning tracer CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    record = sub.add_parser("record", help="Recorded-trace utilities")
    record_sub = record.add_subparsers(dest="record_cmd", required=True)

    p_list = record_sub.add_parser("list", help="List recorded trace files")
    p_list.add_argument(
        "--path",
        default=None,
        help="Recording directory (defaults to MIRU_RECORD_PATH or ./miru_traces)",
    )

    p_export = record_sub.add_parser("export", help="Concatenate trace files")
    p_export.add_argument(
        "--path",
        default=None,
        help="Recording directory (defaults to MIRU_RECORD_PATH or ./miru_traces)",
    )
    p_export.add_argument(
        "--out",
        required=True,
        help="Output file path",
    )
    p_export.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Export format",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "record":
        if args.record_cmd == "list":
            return run_list(args.path)
        if args.record_cmd == "export":
            return run_export(args.path, args.out, args.format)

    parser.error(f"unhandled command: {args.cmd}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
