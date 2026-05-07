"""``miru profile`` subcommand — latency profiler CLI."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def run_profile(
    backend: str,
    n_warmup: int,
    n_timed: int,
    image_size: int,
    seed: int,
    out_path: Optional[str],
    *,
    stream=None,
) -> int:
    """Run the latency profiler and print a summary table."""
    out = stream or sys.stdout
    out.write(
        f"profiling: backend={backend}  warmup={n_warmup}  timed={n_timed}"
        f"  size={image_size}  seed={seed}\n"
    )
    out.flush()

    from miru.bench.profile import profile_backend

    try:
        result = profile_backend(
            backend,
            n_warmup=n_warmup,
            n_timed=n_timed,
            image_size=image_size,
            seed=seed,
            save=bool(out_path),
            output_dir=Path(out_path) if out_path else None,
        )
    except (RuntimeError, ValueError) as exc:
        out.write(f"error: {exc}\n")
        return 1

    lms = result.latency_ms
    out.write(_format_profile(result.backend, result.n_timed, lms, result.calls_per_second))

    if out_path:
        import json
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        out.write(f"saved → {p}\n")

    return 0


def _format_profile(
    backend: str,
    n_timed: int,
    lms: dict,
    cps: float,
) -> str:
    """Render a compact profile summary table."""
    lines = [
        "",
        f"Profile: backend={backend}  n={n_timed}",
        "",
        f"  {'metric':<10}  {'ms':>8}",
        f"  {'──────────':<10}  {'────────':>8}",
        f"  {'mean':<10}  {lms['mean']:>8.3f}",
        f"  {'std':<10}  {lms['std']:>8.3f}",
        f"  {'min':<10}  {lms['min']:>8.3f}",
        f"  {'p50':<10}  {lms['p50']:>8.3f}",
        f"  {'p95':<10}  {lms['p95']:>8.3f}",
        f"  {'p99':<10}  {lms['p99']:>8.3f}",
        f"  {'p99.9':<10}  {lms['p999']:>8.3f}",
        f"  {'max':<10}  {lms['max']:>8.3f}",
        "",
        f"  throughput  {cps:>8.1f} calls/s",
        "",
    ]
    return "\n".join(lines)


__all__ = ["run_profile", "_format_profile"]
