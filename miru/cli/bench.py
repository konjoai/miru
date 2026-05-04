"""``miru bench …`` subcommand implementations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from miru.bench.runner import (
    compare_results,
    load_result,
    run_benchmark,
    save_result,
)


def run_run(
    backend: str,
    n: int,
    seed: int,
    out_path: Optional[str],
    *,
    top_pct: float = 0.20,
    k_for_hit: int = 1,
    stream=None,
) -> int:
    """Execute a benchmark run and print/save the result."""
    out = stream or sys.stdout
    out.write(f"running bench: backend={backend} n={n} seed={seed}\n")
    out.flush()

    result = run_benchmark(
        backend, n=n, seed=seed, top_pct=top_pct, k_for_hit=k_for_hit
    )

    out.write(_format_summary(result))

    if out_path:
        path = save_result(result, out_path)
        out.write(f"\nsaved → {path}\n")
    return 0


def run_show(in_path: str, *, stream=None) -> int:
    """Pretty-print an existing result file."""
    out = stream or sys.stdout
    result = load_result(in_path)
    out.write(_format_summary(result))
    return 0


def run_compare(
    a_path: str, b_path: str, *, metric: str = "iou", stream=None
) -> int:
    """Compare two runs on the named metric and report a paired delta."""
    out = stream or sys.stdout
    a = load_result(a_path)
    b = load_result(b_path)
    cmp = compare_results(a, b, metric=metric)

    direction = (
        "→ b WINS" if cmp["mean_delta"] > 0
        else "→ a WINS" if cmp["mean_delta"] < 0
        else "→ tie"
    )
    out.write(
        f"\ncompare {metric} on n={cmp['n']} paired samples:\n"
        f"  a ({Path(a_path).name}): mean = {cmp['a_mean']:.4f}\n"
        f"  b ({Path(b_path).name}): mean = {cmp['b_mean']:.4f}\n"
        f"  mean_delta (b - a) = {cmp['mean_delta']:+.4f}  ± {cmp['std_delta']:.4f}\n"
        f"  paired t = {cmp['t_statistic']:+.3f}  (df = {cmp['degrees_of_freedom']})\n"
        f"  {direction}\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_summary(result: dict) -> str:
    """Render a compact human-readable summary of a benchmark result."""
    agg = result["aggregate"]
    lines = []
    lines.append("")
    lines.append(f"Miru bench · backend={result['backend']} n={result['n']} "
                 f"seed={result['seed']} size={result['size']}")
    lines.append(f"  ts={result['timestamp_utc']}  py={result['hardware']['python']}")
    lines.append("")
    lines.append("  metric              mean   ±std    p50    p95")
    lines.append("  ──────────────  ──────  ─────  ─────  ─────")
    for key in ("iou", "auc", "hit1", "latency_ms"):
        a = agg[key]
        unit = "ms " if key == "latency_ms" else "   "
        lines.append(
            f"  {key:14s} {unit}{a['mean']:6.3f} {a['std']:6.3f} "
            f"{a['p50']:6.3f} {a['p95']:6.3f}"
        )
    # Per-variant breakdown for IoU.
    by_var: dict[str, list[float]] = {}
    for s in result["samples"]:
        by_var.setdefault(s["variant"], []).append(s["iou"])
    if by_var:
        lines.append("")
        lines.append("  iou by variant:")
        for var, vals in sorted(by_var.items()):
            mean = sum(vals) / len(vals)
            lines.append(f"    {var:8s}  n={len(vals):3d}  mean={mean:.3f}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["run_run", "run_show", "run_compare"]
