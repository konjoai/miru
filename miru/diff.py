"""Post-hoc diff of two recorded explanations.

``diff_records(rec_a, rec_b)`` is the inverse direction of
``/explain/compare`` (which re-runs two methods on a fresh image).
Instead, this loads two existing analysis records by ID and reports
how their attribution maps differ — useful for auditing model
versions, validating cache hits, or sanity-checking determinism.

The math is small and deliberately dependency-light: pure NumPy,
nothing fancy.

- ``cosine_similarity`` ∈ ``[-1, 1]`` — the standard dot-product
  measure between the two flattened grids.  ``1`` means identical
  directions, ``0`` means orthogonal, ``-1`` means opposite.
- ``l2_distance`` — Euclidean distance between the flattened grids
  after min-max normalisation to ``[0, 1]`` so the magnitude is
  comparable across methods with different raw scales.
- ``delta_grid`` — the per-cell signed difference ``b - a`` on the
  normalised grids.  Same shape as the larger input (the smaller is
  bilinearly upsampled to match).
- ``top_changed`` — the ``top_n`` cells with the largest absolute
  delta, sorted by ``|delta|`` descending.
- ``summary`` — a short human-readable sentence describing the
  dominant direction of change ("A focused more on the lower-right;
  B shifted toward the centre").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from miru.bench.metrics import bilinear_upsample


@dataclass(frozen=True)
class TopChangedRegion:
    """One cell where attribution moved most between A and B."""

    row: int
    col: int
    value_a: float
    value_b: float
    delta: float          # b - a, sign preserved


@dataclass(frozen=True)
class DiffResult:
    """Structured output of :func:`diff_records`."""

    analysis_id_a: str
    analysis_id_b: str
    method_a: str
    method_b: str
    backend_a: str
    backend_b: str
    cosine_similarity: float
    l2_distance: float
    delta_grid: list[list[float]]
    top_changed: list[TopChangedRegion]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_grid(record: dict[str, Any], side: str) -> np.ndarray:
    """Pull the ``attention_grid`` from a record dict; validate basic shape."""
    trace = record.get("trace") or {}
    raw = trace.get("attention_grid")
    if raw is None:
        raise ValueError(f"record {side} has no attention_grid")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            f"record {side} attention_grid must be a non-empty 2-D array; "
            f"got shape {arr.shape}"
        )
    return arr


def _minmax_normalise(grid: np.ndarray) -> np.ndarray:
    """Min-max normalise to ``[0, 1]``; flat input becomes all-zero."""
    lo = float(grid.min())
    hi = float(grid.max())
    if hi - lo < 1e-12:
        return np.zeros_like(grid, dtype=np.float64)
    return ((grid - lo) / (hi - lo)).astype(np.float64)


def _summarise(delta: np.ndarray, top: list[TopChangedRegion]) -> str:
    """Build a short human-readable description of where attribution moved.

    Looks at the centre-of-mass shift of positive vs. negative delta
    cells.  Says "A focused more on the X; B shifted toward Y" with
    X / Y picked from a 3x3 grid of spatial labels (top-left, …,
    bottom-right).
    """
    if not top:
        return "Explanations are effectively identical (no change above noise)."
    pos = np.where(delta > 0, delta, 0.0)
    neg = np.where(delta < 0, -delta, 0.0)
    pos_label = _grid_label(_centroid(pos)) if pos.sum() > 0 else None
    neg_label = _grid_label(_centroid(neg)) if neg.sum() > 0 else None

    if pos_label and neg_label and pos_label != neg_label:
        return (
            f"A focused more on the {neg_label}; "
            f"B shifted toward the {pos_label}."
        )
    if pos_label and not neg_label:
        return f"B picked up new attention on the {pos_label} that A did not."
    if neg_label and not pos_label:
        return f"A had attention on the {neg_label} that B dropped."
    # Same region for both — magnitude differs but direction is the same.
    return f"Both explanations focus on the {pos_label or neg_label}; magnitudes differ."


def _centroid(weight: np.ndarray) -> tuple[float, float]:
    """Return (row_centroid, col_centroid) of a non-negative weight grid."""
    total = float(weight.sum())
    if total <= 0:
        return (weight.shape[0] / 2.0, weight.shape[1] / 2.0)
    rows = np.arange(weight.shape[0]).reshape(-1, 1)
    cols = np.arange(weight.shape[1]).reshape(1, -1)
    r = float((rows * weight).sum() / total)
    c = float((cols * weight).sum() / total)
    return r, c


def _grid_label(centroid: tuple[float, float]) -> str:
    """Map (row, col) centroid to a 3×3 spatial label."""
    r, c = centroid
    rows = max(1, r + c)  # unused but keeps `r` referenced — see below
    del rows
    # Use the centroid relative to a 3×3 grid: divide the [0, 1)
    # normalised position into thirds.
    # The actual grid bounds aren't known here, so we treat r/c as
    # already in [0, max_dim) and re-normalise by their max.
    # In practice the caller normalises before calling _summarise.
    return _SPATIAL_LABELS[int(min(2, max(0, r * 3 // 1)))][int(min(2, max(0, c * 3 // 1)))]


_SPATIAL_LABELS: list[list[str]] = [
    ["top-left",    "top",    "top-right"],
    ["middle-left", "centre", "middle-right"],
    ["bottom-left", "bottom", "bottom-right"],
]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def diff_records(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
    *,
    top_n: int = 10,
) -> DiffResult:
    """Diff two analysis records on their attribution maps.

    Args:
        rec_a: Recorder dict produced by ``/explain`` (must contain
            ``trace.attention_grid``).
        rec_b: Second record, same shape.
        top_n: How many top-changed cells to surface. Clamped to
            ``1..256``.

    Returns:
        :class:`DiffResult`.

    Raises:
        ValueError: When either record is missing an attention grid
            or carries a malformed one.
    """
    if not 1 <= top_n <= 256:
        raise ValueError(f"top_n must be in 1..256, got {top_n}")

    grid_a = _extract_grid(rec_a, "a")
    grid_b = _extract_grid(rec_b, "b")

    # Align shapes: upsample the smaller grid via bilinear to the larger one.
    target_h = max(grid_a.shape[0], grid_b.shape[0])
    target_w = max(grid_a.shape[1], grid_b.shape[1])
    if grid_a.shape != (target_h, target_w):
        grid_a = bilinear_upsample(grid_a.astype(np.float32), target_h, target_w).astype(np.float64)
    if grid_b.shape != (target_h, target_w):
        grid_b = bilinear_upsample(grid_b.astype(np.float32), target_h, target_w).astype(np.float64)

    norm_a = _minmax_normalise(grid_a)
    norm_b = _minmax_normalise(grid_b)

    # Cosine similarity on the *raw* (pre-normalisation) flattened
    # vectors so a uniform vs. another uniform doesn't collapse to NaN.
    flat_a = grid_a.flatten()
    flat_b = grid_b.flatten()
    denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    cosine = float(flat_a @ flat_b / denom) if denom > 1e-12 else 0.0
    cosine = max(-1.0, min(1.0, cosine))

    l2 = float(np.linalg.norm(norm_b - norm_a))
    delta = norm_b - norm_a

    # Top-N cells by absolute delta.
    abs_delta = np.abs(delta)
    flat_idx = np.argsort(abs_delta, axis=None)[::-1][:top_n]
    top_rows, top_cols = np.unravel_index(flat_idx, delta.shape)
    top_regions = [
        TopChangedRegion(
            row=int(r),
            col=int(c),
            value_a=float(norm_a[r, c]),
            value_b=float(norm_b[r, c]),
            delta=float(delta[r, c]),
        )
        for r, c in zip(top_rows, top_cols)
    ]
    # Drop cells where delta is effectively zero (no change).
    top_regions = [t for t in top_regions if abs(t.delta) > 1e-9]

    # Compute spatial labels in normalised [0, 1) space so _grid_label
    # bins thirds correctly regardless of grid resolution.
    norm_delta = delta.copy()
    pos = np.where(norm_delta > 0, norm_delta, 0.0)
    neg = np.where(norm_delta < 0, -norm_delta, 0.0)

    def _spatial(weight: np.ndarray) -> str | None:
        if weight.sum() <= 1e-12:
            return None
        r, c = _centroid(weight)
        return _SPATIAL_LABELS[
            min(2, max(0, int(r / weight.shape[0] * 3)))
        ][
            min(2, max(0, int(c / weight.shape[1] * 3)))
        ]

    pos_label = _spatial(pos)
    neg_label = _spatial(neg)

    if not top_regions:
        summary = "Explanations are effectively identical (no change above noise)."
    elif pos_label and neg_label and pos_label != neg_label:
        summary = (
            f"A focused more on the {neg_label}; "
            f"B shifted toward the {pos_label}."
        )
    elif pos_label and not neg_label:
        summary = f"B picked up new attention on the {pos_label} that A did not."
    elif neg_label and not pos_label:
        summary = f"A had attention on the {neg_label} that B dropped."
    else:
        summary = (
            f"Both explanations focus on the {pos_label or neg_label}; "
            "magnitudes differ."
        )

    trace_a = rec_a.get("trace") or {}
    trace_b = rec_b.get("trace") or {}

    return DiffResult(
        analysis_id_a=str(rec_a.get("analysis_id") or ""),
        analysis_id_b=str(rec_b.get("analysis_id") or ""),
        method_a=str(trace_a.get("method") or trace_a.get("explanation_method") or ""),
        method_b=str(trace_b.get("method") or trace_b.get("explanation_method") or ""),
        backend_a=str(trace_a.get("backend") or ""),
        backend_b=str(trace_b.get("backend") or ""),
        cosine_similarity=cosine,
        l2_distance=l2,
        delta_grid=delta.astype(float).tolist(),
        top_changed=top_regions,
        summary=summary,
    )


__all__ = ["TopChangedRegion", "DiffResult", "diff_records"]
