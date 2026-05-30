"""Search recorded explanations by attribution-pattern similarity.

Given a query attribution grid — either supplied directly or pulled
from an existing analysis — find the top-K recorded explanations
whose attention maps are most similar by cosine similarity.

Useful for the "find other inputs the model explained similarly"
workflow: feed in the saliency map of an interesting case, see which
historical analyses share the same focus pattern.

The search is exact (no embedding index, no approximate nearest
neighbour) — fine for the audit-log scale this system targets
(thousands of records).  When that scale grows, swap in faiss-cpu or
hnswlib behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from miru.bench.metrics import bilinear_upsample
from miru.history import HistoryRecord, load_records


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchMatch:
    """One result row in :class:`SearchResult.matches`."""

    analysis_id: str
    ts: str
    method: str
    backend: str
    question: str
    similarity: float    # cosine ∈ [-1, 1]


@dataclass(frozen=True)
class SearchResult:
    """Output of :func:`search_by_pattern`."""

    matches: list[SearchMatch]
    n_candidates: int    # how many records were scored
    n_scanned: int       # how many records were in the source before filtering
    top_k: int
    query_analysis_id: Optional[str]


# ---------------------------------------------------------------------------
# Validation + helpers
# ---------------------------------------------------------------------------


def _validate_grid(grid: Any, label: str) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            f"{label} must be a non-empty 2-D array; got shape {arr.shape}"
        )
    return arr


def _record_grid(record: dict[str, Any]) -> Optional[np.ndarray]:
    """Extract the attention_grid; None if missing/malformed (skipped)."""
    trace = record.get("trace") or {}
    raw = trace.get("attention_grid")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _flatten_aligned(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if grid.shape != (target_h, target_w):
        grid = bilinear_upsample(grid.astype(np.float32), target_h, target_w).astype(np.float64)
    return grid.flatten()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, float(a @ b / denom)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_by_pattern(
    *,
    query_grid: Optional[list[list[float]] | np.ndarray] = None,
    query_analysis_id: Optional[str] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    top_k: int = 10,
    min_similarity: Optional[float] = None,
    max_scan: int = 500,
    directory: Optional[str] = None,
    source: Optional[list[dict[str, Any]]] = None,
) -> SearchResult:
    """Find recorded explanations whose attention map matches the query.

    Args:
        query_grid: A 2-D float saliency map. Mutually exclusive with
            ``query_analysis_id``; exactly one must be provided.
        query_analysis_id: An existing analysis ID; its
            ``attention_grid`` becomes the query.  The matching record
            is excluded from the result set so callers don't get
            "self" with similarity 1.0 cluttering the top.
        method: Optional filter — only score records with this method.
        model: Optional filter — only score records with this backend.
        top_k: Number of matches to return. Clamped ``1..50``.
        min_similarity: Optional lower bound; matches with cosine
            similarity below this are dropped before slicing.
            Must be in ``[-1, 1]`` if set.
        max_scan: Maximum number of recorded records to score.
            Bounded ``1..2000``.  Older records past this cap are
            skipped (newest-first ordering preserved).
        directory: Recorder directory override (test hook).
        source: Optional pre-loaded iterable of records (test hook).
            When supplied, ``directory`` and the live recorder are
            bypassed.

    Returns:
        :class:`SearchResult`.

    Raises:
        ValueError: On bad arguments — neither / both of
            ``query_grid`` and ``query_analysis_id`` supplied,
            out-of-range top_k / min_similarity / max_scan, or
            query_analysis_id not found in the store.
    """
    if (query_grid is None) == (query_analysis_id is None):
        raise ValueError(
            "exactly one of query_grid or query_analysis_id must be supplied"
        )
    if not 1 <= top_k <= 50:
        raise ValueError(f"top_k must be in 1..50, got {top_k}")
    if not 1 <= max_scan <= 2000:
        raise ValueError(f"max_scan must be in 1..2000, got {max_scan}")
    if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
        raise ValueError(
            f"min_similarity must be in [-1, 1], got {min_similarity}"
        )

    # Materialise the source so we can pull the query record from it
    # before scoring (avoids a second filesystem scan).
    if source is not None:
        all_records: list[dict[str, Any]] = list(source)
    else:
        all_records = list(load_records(directory))

    # Resolve the query vector.
    if query_analysis_id is not None:
        match = next(
            (r for r in all_records if r.get("analysis_id") == query_analysis_id),
            None,
        )
        if match is None:
            raise ValueError(
                f"query_analysis_id {query_analysis_id!r} not found in store"
            )
        query_grid_np = _record_grid(match)
        if query_grid_np is None:
            raise ValueError(
                f"record {query_analysis_id!r} has no attention_grid"
            )
    else:
        query_grid_np = _validate_grid(query_grid, "query_grid")

    # Filter candidates by method/model and exclude self.
    candidates: list[dict[str, Any]] = []
    for rec in all_records:
        if query_analysis_id is not None and rec.get("analysis_id") == query_analysis_id:
            continue
        trace = rec.get("trace") or {}
        if method is not None:
            rec_method = trace.get("method") or trace.get("explanation_method")
            if rec_method != method:
                continue
        if model is not None and trace.get("backend") != model:
            continue
        candidates.append(rec)

    n_scanned = len(all_records)
    candidates = candidates[:max_scan]
    n_candidates = len(candidates)

    # Score every candidate. Bilinearly align each candidate grid to
    # the query's shape so a 8×8 query can match against 16×16 records.
    target_h, target_w = query_grid_np.shape
    query_flat = query_grid_np.flatten()

    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in candidates:
        cand_grid = _record_grid(rec)
        if cand_grid is None:
            continue
        cand_flat = _flatten_aligned(cand_grid, target_h, target_w)
        sim = _cosine(query_flat, cand_flat)
        if min_similarity is not None and sim < min_similarity:
            continue
        scored.append((sim, rec))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    page = scored[:top_k]

    matches = [
        SearchMatch(
            analysis_id=str(rec.get("analysis_id") or ""),
            ts=str(rec.get("ts") or ""),
            method=str(
                (rec.get("trace") or {}).get("method")
                or (rec.get("trace") or {}).get("explanation_method")
                or ""
            ),
            backend=str((rec.get("trace") or {}).get("backend") or ""),
            question=str(rec.get("question") or ""),
            similarity=float(sim),
        )
        for sim, rec in page
    ]

    return SearchResult(
        matches=matches,
        n_candidates=n_candidates,
        n_scanned=n_scanned,
        top_k=top_k,
        query_analysis_id=query_analysis_id,
    )


__all__ = [
    "SearchMatch",
    "SearchResult",
    "search_by_pattern",
]
