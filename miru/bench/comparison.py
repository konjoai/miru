"""Backend comparison: run benchmark on two backends and produce a delta report."""
from __future__ import annotations

import json
import logging
import socket
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class BackendComparison:
    name: str
    timestamp: str
    backend_a: str
    backend_b: str
    result_a: Any  # dict from run_benchmark
    result_b: Any  # dict from run_benchmark
    comparison: Any  # dict from compare_results, or None
    winner: str    # "a", "b", or "tie"
    hardware: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path = Path("benchmarks/results")) -> Path:
        """Write the comparison to a JSON file.  Never overwrites an existing file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = self.timestamp
        filename = f"comparison-{self.backend_a}-vs-{self.backend_b}-{ts}.json"
        path = output_dir / filename
        if path.exists():
            ts2 = time.strftime("%Y%m%dT%H%M%S")
            filename = f"comparison-{self.backend_a}-vs-{self.backend_b}-{ts}-{ts2}.json"
            path = output_dir / filename

        data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "winner": self.winner,
            "hardware": self.hardware,
            "result_a": self.result_a,
            "result_b": self.result_b,
            "comparison": self.comparison,
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved comparison to %s", path)
        return path


def _determine_winner(comparison_obj: Any) -> str:
    """Determine winner from compare_results output.

    compare_results returns a dict.  The key ``mean_delta`` is positive when
    B is better, negative when A is better.  Falls back to counting all
    numeric deltas if ``mean_delta`` is absent.
    """
    if comparison_obj is None:
        return "tie"

    # Fast path: compare_results always returns mean_delta
    if isinstance(comparison_obj, dict):
        if "mean_delta" in comparison_obj:
            delta = comparison_obj["mean_delta"]
            if delta > 0:
                return "b"
            if delta < 0:
                return "a"
            return "tie"
        # Generic fallback: count positive vs negative numeric values
        numeric = {k: v for k, v in comparison_obj.items() if isinstance(v, (int, float))}
        if not numeric:
            return "tie"
        pos = sum(1 for v in numeric.values() if v > 0)
        neg = sum(1 for v in numeric.values() if v < 0)
        if pos > neg:
            return "b"
        if neg > pos:
            return "a"
        return "tie"

    # Dataclass / object fallback
    if hasattr(comparison_obj, "__dict__"):
        return _determine_winner(comparison_obj.__dict__)

    return "tie"


def compare_backends(
    backend_a_name: str,
    backend_b_name: str,
    n_samples: int = 30,
    seed: int = 42,
    comparison_name: str = "",
    output_dir: Optional[Path] = None,
    save: bool = False,
) -> BackendComparison:
    """Run the benchmark harness on two backends and produce a comparison.

    Both backends are addressed by their registry name (e.g. ``"mock"``,
    ``"clip"``).  Only the ``mock`` backend is available without installing
    the ``[backends]`` extras and setting ``MIRU_TEST_REAL_BACKENDS=1``.

    Args:
        backend_a_name: Registry name of the first backend.
        backend_b_name: Registry name of the second backend.
        n_samples: Number of synth samples to run per backend.
        seed: Top-level RNG seed (must match for a paired comparison).
        comparison_name: Human-readable label; defaults to ``"a-vs-b"``.
        output_dir: Directory for the saved JSON (default:
            ``benchmarks/results``).
        save: If ``True``, call :meth:`BackendComparison.save` before
            returning.

    Returns:
        A :class:`BackendComparison` dataclass.

    Raises:
        RuntimeError: If either backend name is not in the registry.
    """
    from miru.models.registry import register_defaults, available
    from miru.bench.runner import run_benchmark, compare_results

    register_defaults()
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    name = comparison_name or f"{backend_a_name}-vs-{backend_b_name}"

    _avail = available()

    def _check(bname: str) -> None:
        if bname not in _avail:
            raise RuntimeError(
                f"Backend '{bname}' not available. "
                f"Available: {_avail}. "
                "Real backends need MIRU_TEST_REAL_BACKENDS=1."
            )

    _check(backend_a_name)
    _check(backend_b_name)

    result_a = run_benchmark(backend_a_name, n=n_samples, seed=seed)
    result_b = run_benchmark(backend_b_name, n=n_samples, seed=seed)

    try:
        cmp = compare_results(result_a, result_b)
    except Exception as exc:  # pragma: no cover — paired runs always match
        logger.warning("compare_results failed: %s", exc)
        cmp = None

    winner = _determine_winner(cmp)
    hw: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    bc = BackendComparison(
        name=name,
        timestamp=timestamp,
        backend_a=backend_a_name,
        backend_b=backend_b_name,
        result_a=result_a,
        result_b=result_b,
        comparison=cmp,
        winner=winner,
        hardware=hw,
    )
    if save:
        bc.save(output_dir or Path("benchmarks/results"))
    return bc


__all__ = ["BackendComparison", "compare_backends", "_determine_winner"]
