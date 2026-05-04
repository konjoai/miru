"""Command-line entry point for Miru utilities.

Subcommands
-----------

``miru record list``    — list recorded trace files in ``MIRU_RECORD_PATH``
``miru record export``  — concatenate recorded traces into a single
                          JSONL or CSV file
``miru bench run``      — run the saliency benchmark over a backend
``miru bench show``     — pretty-print a saved benchmark result
``miru bench compare``  — paired delta between two saved results
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from miru.cli.bench import run_compare, run_run, run_show
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

    # ----- bench --------------------------------------------------------
    bench = sub.add_parser("bench", help="Saliency benchmark utilities")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)

    p_run = bench_sub.add_parser("run", help="Run a benchmark on a backend")
    p_run.add_argument("--backend", default="mock", help="Registered backend name")
    p_run.add_argument("--n", type=int, default=30, help="Number of samples")
    p_run.add_argument("--seed", type=int, default=42, help="Top-level RNG seed")
    p_run.add_argument("--out", default=None, help="Save the JSON result to this path")
    p_run.add_argument("--top-pct", type=float, default=0.20,
                       help="Top-percentile threshold for IoU (default: 0.20)")
    p_run.add_argument("--k", type=int, default=1, help="K used for hit@k (default: 1)")

    p_show = bench_sub.add_parser("show", help="Pretty-print a saved result")
    p_show.add_argument("path", help="Path to a saved JSON result")

    p_cmp = bench_sub.add_parser("compare", help="Paired comparison of two results")
    p_cmp.add_argument("a", help="First result JSON")
    p_cmp.add_argument("b", help="Second result JSON")
    p_cmp.add_argument("--metric", default="iou", choices=("iou", "auc", "hit1", "latency_ms"))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "record":
        if args.record_cmd == "list":
            return run_list(args.path)
        if args.record_cmd == "export":
            return run_export(args.path, args.out, args.format)

    if args.cmd == "bench":
        if args.bench_cmd == "run":
            return run_run(
                args.backend,
                args.n,
                args.seed,
                args.out,
                top_pct=args.top_pct,
                k_for_hit=args.k,
            )
        if args.bench_cmd == "show":
            return run_show(args.path)
        if args.bench_cmd == "compare":
            return run_compare(args.a, args.b, metric=args.metric)

    parser.error(f"unhandled command: {args.cmd}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
