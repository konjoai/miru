"""Post-hoc consensus from already-recorded explanation analysis IDs.

Different from :func:`miru.consensus.compute_consensus` (Phase 13) which
takes a fresh image plus a list of methods and runs every method live.
This module takes a list of **existing** analysis IDs that ran
previously, loads each record from the recorder store, and builds a
weighted-average consensus saliency map without doing any fresh
inference.

Use cases
---------

- "I ran these 4 methods on this input — give me one map that
  reconciles them."
- "I have hourly explanations from a CI job — average them across the
  week."

Weighting modes
---------------

- ``fidelity`` (default) — each record's weight = its
  ``fidelity_score``.  Records without fidelity get the population
  minimum so they still contribute something but aren't trusted as
  much as deletion-test-validated ones.  Falls back to ``uniform``
  when no record carries fidelity.
- ``confidence`` — each record's weight = its ``confidence``.
- ``uniform`` — every record gets weight 1.0.

Agreement score
---------------

For each record, we report a per-record ``agreement_score`` ∈ [-1, 1]:
the cosine similarity between that record's attribution grid and the
final consensus grid.  Records that agree with the consensus get
scores near 1; outliers get scores near 0 (or negative when their
attribution actively contradicts the consensus).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from miru.bench.metrics import bilinear_upsample


WeightingMode = Literal["fidelity", "confidence", "uniform"]
_VALID_MODES: tuple[str, ...] = ("fidelity", "confidence", "uniform")


@dataclass(frozen=True)
class PerRecordContribution:
    """One slot in :class:`PosthocConsensusResult.per_record`."""

    analysis_id: str
    method: str
    backend: str
    weight: float           # the weight actually used (after fallback)
    agreement_score: float  # cosine(this_grid, consensus_grid) ∈ [-1, 1]


@dataclass(frozen=True)
class TopConsensusRegion:
    """One cell in :class:`PosthocConsensusResult.top_regions`."""

    row: int
    col: int
    score: float


@dataclass(frozen=True)
class PosthocConsensusResult:
    """Output of :func:`build_consensus`."""

    consensus_grid: list[list[float]]
    per_record: list[PerRecordContribution]
    top_regions: list[TopConsensusRegion]
    weighting: str
    n_records: int
    grid_h: int
    grid_w: int


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_grid(record: dict[str, Any], idx: int) -> np.ndarray:
    """Pull the attention_grid from one record; validate shape."""
    trace = record.get("trace") or {}
    raw = trace.get("attention_grid")
    if raw is None:
        raise ValueError(
            f"record at index {idx} (analysis_id={record.get('analysis_id')!r}) "
            "has no attention_grid"
        )
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            f"record at index {idx} attention_grid must be non-empty 2-D; "
            f"got shape {arr.shape}"
        )
    return arr


def _resolve_weights(
    records: list[dict[str, Any]],
    mode: WeightingMode,
) -> list[float]:
    """Build per-record weights according to *mode*.

    For ``fidelity``: records without fidelity get the minimum
    non-None fidelity in the population.  When no record has
    fidelity, fall back to ``uniform`` so we don't return all-zero
    weights and divide by zero downstream.
    """
    if mode == "uniform":
        return [1.0] * len(records)

    if mode == "confidence":
        weights: list[float] = []
        for r in records:
            conf = (r.get("trace") or {}).get("confidence")
            weights.append(float(conf) if conf is not None else 0.0)
        # Avoid all-zero weights (would div-by-zero); fall back to uniform.
        if not any(w > 0 for w in weights):
            return [1.0] * len(records)
        return weights

    # mode == "fidelity"
    fids: list[Optional[float]] = []
    for r in records:
        fidelity = (r.get("trace") or {}).get("fidelity") or {}
        score = fidelity.get("fidelity_score") if isinstance(fidelity, dict) else None
        fids.append(float(score) if score is not None else None)

    valid = [f for f in fids if f is not None]
    if not valid:
        # Nothing has fidelity — fall back to uniform; clients see the
        # echoed ``weighting`` and understand what happened.
        return [1.0] * len(records)

    floor = min(valid)
    return [f if f is not None else floor for f in fids]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe on zero-magnitude vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, float(a @ b / denom)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_consensus(
    records: list[dict[str, Any]],
    *,
    weighting: WeightingMode = "fidelity",
    top_k: int = 5,
) -> PosthocConsensusResult:
    """Weighted-average consensus over recorded attention grids.

    Args:
        records: List of recorder dicts (typically the output of
            :func:`miru.recorder.find_record_by_id` for each ID the
            caller wants to combine).  Length 2..16 enforced by the
            API; this function accepts ≥ 1.
        weighting: One of ``fidelity`` / ``confidence`` / ``uniform``.
        top_k: Number of highest-activation cells in the consensus to
            report. Clamped ``1..64``.

    Returns:
        :class:`PosthocConsensusResult`.

    Raises:
        ValueError: When records is empty, weighting is unknown,
            top_k is out of range, or a record is missing an
            attention_grid / has the wrong shape.
    """
    if not records:
        raise ValueError("records must be non-empty")
    if weighting not in _VALID_MODES:
        raise ValueError(
            f"weighting must be one of {_VALID_MODES}; got {weighting!r}"
        )
    if not 1 <= top_k <= 64:
        raise ValueError(f"top_k must be in 1..64, got {top_k}")

    grids = [_extract_grid(rec, i) for i, rec in enumerate(records)]
    target_h = max(g.shape[0] for g in grids)
    target_w = max(g.shape[1] for g in grids)
    aligned = [
        g if g.shape == (target_h, target_w)
        else bilinear_upsample(g.astype(np.float32), target_h, target_w).astype(np.float64)
        for g in grids
    ]

    raw_weights = _resolve_weights(records, weighting)
    total_w = sum(raw_weights)
    if total_w <= 0:
        # Defensive: _resolve_weights guarantees this can't happen,
        # but if a custom caller injects zeros, fall back to uniform.
        raw_weights = [1.0] * len(records)
        total_w = float(len(records))

    consensus = np.zeros((target_h, target_w), dtype=np.float64)
    for grid, w in zip(aligned, raw_weights):
        consensus += grid * (w / total_w)

    # Top-K regions in the consensus.
    flat = consensus.flatten()
    k_clamped = min(top_k, flat.size)
    idxs = np.argpartition(flat, -k_clamped)[-k_clamped:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    top_rows, top_cols = np.unravel_index(idxs, consensus.shape)
    top_regions = [
        TopConsensusRegion(row=int(r), col=int(c), score=float(consensus[r, c]))
        for r, c in zip(top_rows, top_cols)
    ]

    # Per-record agreement scores (cosine vs. consensus).
    consensus_flat = consensus.flatten()
    per_record: list[PerRecordContribution] = []
    for rec, grid, weight in zip(records, aligned, raw_weights):
        trace = rec.get("trace") or {}
        per_record.append(PerRecordContribution(
            analysis_id=str(rec.get("analysis_id") or ""),
            method=str(trace.get("method") or trace.get("explanation_method") or ""),
            backend=str(trace.get("backend") or ""),
            weight=float(weight),
            agreement_score=_cosine(grid.flatten(), consensus_flat),
        ))

    return PosthocConsensusResult(
        consensus_grid=consensus.astype(float).tolist(),
        per_record=per_record,
        top_regions=top_regions,
        weighting=weighting,
        n_records=len(records),
        grid_h=target_h,
        grid_w=target_w,
    )


__all__ = [
    "WeightingMode",
    "PerRecordContribution",
    "TopConsensusRegion",
    "PosthocConsensusResult",
    "build_consensus",
]
