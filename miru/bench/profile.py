"""Latency profiler — measure per-backend inference latency with warmup.

Design notes
------------

- ``warmup`` calls are made first and discarded; they ensure the backend's
  lazy-loader fires and any Python-level JIT / import caches are warm before
  we start the clock.
- Timed iterations use ``time.perf_counter`` around ``backend.infer()`` only;
  the synth image generation is done up-front so it does not pollute the
  latency numbers.
- Percentiles are computed by ``numpy.percentile`` with linear interpolation
  — the same convention used in ``runner.py``.
- Throughput (``calls_per_second``) is derived from the mean timed latency,
  not from wall-clock elapsed time, so it is single-call throughput.  A
  future concurrent-ramp extension can add multi-thread throughput.
- No new runtime dependencies — pure NumPy + stdlib.
"""
from __future__ import annotations

import json
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ProfileResult:
    """Latency profile for a single backend.

    Attributes:
        backend:          Registry name of the profiled backend.
        timestamp:        ISO-8601 timestamp at the start of the run.
        n_warmup:         Number of discarded warm-up calls.
        n_timed:          Number of calls included in the statistics.
        image_size:       Side length (pixels) of the synth probe image used.
        latency_ms:       Per-call millisecond statistics dict.
        calls_per_second: Inverse of mean latency (single-call throughput).
        hardware:         Platform / runtime snapshot.
        raw_ms:           Raw per-call latency list (milliseconds).
    """

    backend: str
    timestamp: str
    n_warmup: int
    n_timed: int
    image_size: int
    latency_ms: Dict[str, float]
    calls_per_second: float
    hardware: Dict[str, Any]
    raw_ms: List[float] = field(default_factory=list)

    def save(self, output_dir: Path = Path("benchmarks/results")) -> Path:
        """Persist to a timestamped JSON file.  Never overwrites an existing file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = self.timestamp.replace(":", "").replace("-", "").replace("T", "T")[:15]
        filename = f"profile-{self.backend}-{ts}.json"
        path = output_dir / filename
        if path.exists():
            ts2 = time.strftime("%Y%m%dT%H%M%S")
            filename = f"profile-{self.backend}-{ts}-{ts2}.json"
            path = output_dir / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (suitable for JSON)."""
        return {
            "schema_version": 1,
            "backend": self.backend,
            "timestamp": self.timestamp,
            "n_warmup": self.n_warmup,
            "n_timed": self.n_timed,
            "image_size": self.image_size,
            "latency_ms": self.latency_ms,
            "calls_per_second": self.calls_per_second,
            "hardware": self.hardware,
            "raw_ms": self.raw_ms,
        }


def profile_backend(
    backend_name: str = "mock",
    *,
    n_warmup: int = 3,
    n_timed: int = 20,
    image_size: int = 64,
    seed: int = 0,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> ProfileResult:
    """Profile a registered backend's per-call inference latency.

    Generates a single synthetic probe image (reproducible from *seed* and
    *image_size*) and runs *n_warmup* discarded calls followed by *n_timed*
    measured calls.  Returns a :class:`ProfileResult` with full percentile
    statistics.

    Args:
        backend_name: Registry name of the backend to profile.
        n_warmup:     Number of warm-up calls before timing starts.
        n_timed:      Number of timed calls (must be >= 1).
        image_size:   Side length of the probe image in pixels.
        seed:         RNG seed for the probe image (default 0).
        save:         If ``True`` persist the result JSON.
        output_dir:   Directory for the saved file; defaults to
                      ``benchmarks/results``.

    Returns:
        A :class:`ProfileResult` dataclass.

    Raises:
        ValueError:  If *n_timed* < 1.
        RuntimeError: If *backend_name* is not in the registry.
    """
    if n_timed < 1:
        raise ValueError(f"n_timed must be >= 1, got {n_timed}")

    from miru.bench.synth import generate_sample
    from miru.models.registry import available, register_defaults

    register_defaults()
    avail = available()
    if backend_name not in avail:
        raise RuntimeError(
            f"Backend '{backend_name}' not in registry. "
            f"Available: {avail}. "
            "Real backends need MIRU_TEST_REAL_BACKENDS=1."
        )

    from miru.models.registry import get as registry_get

    backend = registry_get(backend_name)

    # Build a fixed probe image outside the timing loop.
    sample = generate_sample(seed=seed, index=0, size=image_size)
    image = sample.image
    question = sample.question

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # --- warm-up ---
    for _ in range(n_warmup):
        backend.infer(image, question)

    # --- timed iterations ---
    raw_ms: List[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        backend.infer(image, question)
        raw_ms.append((time.perf_counter() - t0) * 1_000.0)

    stats = _percentile_stats(raw_ms)
    mean_ms = stats["mean"]
    cps = (1_000.0 / mean_ms) if mean_ms > 0 else float("inf")

    result = ProfileResult(
        backend=backend_name,
        timestamp=timestamp,
        n_warmup=n_warmup,
        n_timed=n_timed,
        image_size=image_size,
        latency_ms=stats,
        calls_per_second=cps,
        hardware=_hardware_metadata(),
        raw_ms=raw_ms,
    )

    if save:
        result.save(output_dir or Path("benchmarks/results"))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, and percentile statistics."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "p50", "p95", "p99", "p999")}
    pcts = np.percentile(arr, [50.0, 95.0, 99.0, 99.9]).tolist()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": pcts[0],
        "p95": pcts[1],
        "p99": pcts[2],
        "p999": pcts[3],
    }


def _hardware_metadata() -> Dict[str, str]:
    """Snapshot runtime so saved profiles are reproducible-ish."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }


__all__ = ["ProfileResult", "profile_backend", "_percentile_stats"]
