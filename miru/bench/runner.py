"""Benchmark runner — drive a backend over a synth dataset, score, persist.

Output schema
-------------

A run produces one JSON document with the following top-level keys::

    {
      "schema_version": 1,
      "timestamp_utc":  "2026-05-03T...",
      "backend":        "mock" | "clip" | ...,
      "n":              <int>,
      "seed":           <int>,
      "size":           <int>,
      "hardware":       { "platform": "...", "python": "...", "machine": "...", ... },
      "config":         { "top_pct": 0.20, "k_for_hit": 1, "attention_grid": 16 },
      "aggregate":      { "iou": {mean, std, p50, p95}, "auc": {...}, "hit1": {...}, "latency_ms": {...} },
      "samples":        [ { index, variant, iou, auc, hit1, latency_ms, centroids }, ... ]
    }

The aggregate block uses the same key shape across metrics so downstream
consumers (the CLI, the test suite, future visualisations) can iterate
generically over them.

Design notes
------------

- Latency is measured end-to-end around ``backend.infer()`` only — the
  attention extractor and metrics are fast and would only add noise.
- The runner does **not** depend on the FastAPI surface or the recorder.
  Benchmarks run against ``VLMBackend`` directly so they exercise the
  inference layer in isolation.
- Hardware metadata is captured automatically from ``platform`` so
  archived results are reproducible-ish without manual annotation.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.bench.metrics import auc_roc, hit_at_k, iou_at_topk_pct
from miru.bench.synth import SynthSample, generate_dataset
from miru.models import registry

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Sample-level scoring
# ---------------------------------------------------------------------------


def _score_sample(
    attn_grid: np.ndarray,
    sample: SynthSample,
    *,
    top_pct: float,
    k_for_hit: int,
) -> dict[str, float]:
    return {
        "iou": iou_at_topk_pct(attn_grid, sample.mask, top_pct=top_pct),
        "auc": auc_roc(attn_grid, sample.mask),
        "hit1": hit_at_k(attn_grid, sample.mask, k=k_for_hit),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    backend_name: str = "mock",
    *,
    n: int = 30,
    seed: int = 42,
    size: int = 64,
    top_pct: float = 0.20,
    k_for_hit: int = 1,
    attention_grid: int = 16,
) -> dict[str, Any]:
    """Run the named backend over ``n`` synth samples and return scored results.

    Args:
        backend_name: Registered backend; falls back to mock on KeyError.
        n: Number of samples to run.
        seed: Top-level seed for the synth dataset.
        size: Side length of synth images.
        top_pct: ``iou_at_topk_pct`` threshold.
        k_for_hit: K used for ``hit_at_k``.
        attention_grid: Output resolution of the AttentionExtractor.
    """
    registry.register_defaults()
    try:
        backend = registry.get(backend_name)
    except KeyError:
        backend = registry.get("mock")
        backend_name = backend.name

    extractor = AttentionExtractor(resolution=attention_grid)
    samples = generate_dataset(seed=seed, n=n, size=size)
    sample_results: list[dict[str, Any]] = []

    for sample in samples:
        t0 = time.perf_counter()
        out = backend.infer(sample.image, sample.question)
        latency_ms = (time.perf_counter() - t0) * 1_000.0
        attn_grid = extractor.extract(out.attention_weights)
        scores = _score_sample(
            attn_grid, sample, top_pct=top_pct, k_for_hit=k_for_hit
        )
        sample_results.append({
            "index": sample.meta["index"],
            "variant": sample.meta["variant"],
            "centroids": sample.meta["centroids"],
            "iou": scores["iou"],
            "auc": scores["auc"],
            "hit1": scores["hit1"],
            "latency_ms": latency_ms,
        })

    aggregate = {
        "iou": _agg(s["iou"] for s in sample_results),
        "auc": _agg(s["auc"] for s in sample_results),
        "hit1": _agg(s["hit1"] for s in sample_results),
        "latency_ms": _agg(s["latency_ms"] for s in sample_results),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "n": n,
        "seed": seed,
        "size": size,
        "hardware": _hardware_metadata(),
        "config": {
            "top_pct": top_pct,
            "k_for_hit": k_for_hit,
            "attention_grid": attention_grid,
        },
        "aggregate": aggregate,
        "samples": sample_results,
    }


def save_result(result: dict[str, Any], out_path: Path | str) -> Path:
    """Write *result* as a single JSON document.  Returns the resolved path."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)
    return p


def load_result(in_path: Path | str) -> dict[str, Any]:
    p = Path(in_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison — paired t-test on per-sample metric vectors
# ---------------------------------------------------------------------------


def compare_results(
    a: dict[str, Any], b: dict[str, Any], *, metric: str = "iou"
) -> dict[str, Any]:
    """Compare two runs on a per-sample basis.

    Both runs must have the same ``n`` and ``seed`` — otherwise the
    samples don't pair up and the test is meaningless.

    Returns a dict with the per-sample mean delta, the paired t statistic,
    and a degrees-of-freedom value the caller can hand to scipy if they
    need a p-value (we deliberately skip the p-value to avoid taking a
    scipy dependency).
    """
    if a["n"] != b["n"] or a["seed"] != b["seed"]:
        raise ValueError(
            f"runs are not paired: n={a['n']} vs {b['n']}, seed={a['seed']} vs {b['seed']}"
        )
    va = np.array([s[metric] for s in a["samples"]], dtype=np.float64)
    vb = np.array([s[metric] for s in b["samples"]], dtype=np.float64)
    diff = vb - va
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    se = std_diff / max(1.0, np.sqrt(diff.size)) if std_diff > 0 else 0.0
    t = float(mean_diff / se) if se > 0 else 0.0
    return {
        "metric": metric,
        "n": int(diff.size),
        "a_mean": float(va.mean()),
        "b_mean": float(vb.mean()),
        "mean_delta": mean_diff,  # positive ⇒ b is better
        "std_delta": std_diff,
        "t_statistic": t,
        "degrees_of_freedom": int(diff.size - 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hardware_metadata() -> dict[str, str]:
    """Snapshot the runtime so archived results are reproducible-ish."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }


__all__ = [
    "SCHEMA_VERSION",
    "run_benchmark",
    "save_result",
    "load_result",
    "compare_results",
]
