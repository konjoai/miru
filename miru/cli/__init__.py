"""Command-line entry point for Miru utilities.

Subcommands
-----------

``miru record list``    — list recorded trace files in ``MIRU_RECORD_PATH``
``miru record export``  — concatenate recorded traces into a single
                          JSONL or CSV file
``miru bench run``      — run the saliency benchmark over a backend
``miru bench show``     — pretty-print a saved benchmark result
``miru bench compare``  — paired delta between two saved results
``miru export``         — export a bench result to an HTML report + PNG tiles
``miru compare``        — live CLIP-vs-mock backend comparison artefact
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from miru.cli.bench import run_compare, run_run, run_show
from miru.cli.export import run_export_report
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

    # ----- export -------------------------------------------------------
    p_exp = sub.add_parser(
        "export",
        help="Export a bench result to an HTML report + PNG tiles",
    )
    p_exp.add_argument("result", help="Path to a saved bench result JSON")
    p_exp.add_argument("out_dir", help="Output directory (created if absent)")
    p_exp.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Heatmap opacity 0-1 (default: 0.50)",
    )
    p_exp.add_argument(
        "--colormap",
        default="jet",
        choices=("jet", "hot", "viridis"),
        help="Heatmap colormap (default: jet)",
    )
    p_exp.add_argument(
        "--no-mask-border",
        action="store_true",
        help="Suppress the yellow GT mask boundary on overlay tiles",
    )
    p_exp.add_argument(
        "--no-png-tiles",
        action="store_true",
        help="Skip writing individual PNG tile files; HTML report only",
    )

    # ----- compare (live backend-vs-backend) ----------------------------
    p_compare = sub.add_parser(
        "compare",
        help="Run two backends live and produce a comparison artefact",
    )
    p_compare.add_argument(
        "backend_a",
        help="First backend registry name (e.g. mock)",
    )
    p_compare.add_argument(
        "backend_b",
        help="Second backend registry name (e.g. clip)",
    )
    p_compare.add_argument(
        "--n", type=int, default=30,
        help="Number of synth samples per backend (default: 30)",
    )
    p_compare.add_argument(
        "--seed", type=int, default=42,
        help="Top-level RNG seed (default: 42)",
    )
    p_compare.add_argument(
        "--name", default="",
        help="Human-readable label for the comparison (optional)",
    )
    p_compare.add_argument(
        "--save", action="store_true",
        help="Persist the comparison JSON to benchmarks/results/",
    )
    p_compare.add_argument(
        "--out-dir", default=None,
        help="Override the output directory when --save is set",
    )

    return parser


def _run_compare_backends(args, *, stream=None) -> int:  # type: ignore[no-untyped-def]
    """Handler for the top-level ``miru compare`` subcommand."""
    import sys as _sys
    from pathlib import Path
    from miru.bench.comparison import compare_backends

    out = stream or _sys.stdout
    out.write(
        f"comparing backends: a={args.backend_a} b={args.backend_b} "
        f"n={args.n} seed={args.seed}\n"
    )
    out.flush()

    try:
        bc = compare_backends(
            args.backend_a,
            args.backend_b,
            n_samples=args.n,
            seed=args.seed,
            comparison_name=args.name,
            output_dir=Path(args.out_dir) if args.out_dir else None,
            save=args.save,
        )
    except RuntimeError as exc:
        out.write(f"error: {exc}\n")
        return 1

    cmp = bc.comparison or {}
    delta = cmp.get("mean_delta", 0.0)
    a_mean = cmp.get("a_mean", "n/a")
    b_mean = cmp.get("b_mean", "n/a")

    out.write(
        f"\nComparison: {bc.name}\n"
        f"  backend A ({bc.backend_a}): iou mean = {a_mean}\n"
        f"  backend B ({bc.backend_b}): iou mean = {b_mean}\n"
        f"  mean_delta (b - a)         = {delta:+.4f}\n"
        f"  winner                     = {bc.winner}\n"
    )
    if args.save:
        out.write(f"  (result saved to output_dir)\n")
    return 0


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

    if args.cmd == "export":
        return run_export_report(
            args.result,
            args.out_dir,
            alpha=args.alpha,
            colormap=args.colormap,
            no_mask_border=args.no_mask_border,
            no_png_tiles=args.no_png_tiles,
        )

    if args.cmd == "compare":
        return _run_compare_backends(args)

    parser.error(f"unhandled command: {args.cmd}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
