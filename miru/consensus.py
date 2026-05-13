"""Multi-method saliency consensus — where do explanations agree?

Given two or more saliency maps from different explainers on the same
input, this module answers two questions:

1. **Agreement (cell-wise).** For each cell of the grid, what fraction
   of methods placed it in their top-``top_pct`` salient region?  The
   ``agreement_grid`` is a normalised ``[0, 1]`` matrix; values near 1
   indicate every method agreed this cell is important.
2. **Disagreement (region-wise).** Which cells are top-``top_pct`` for
   exactly one method?  Those are flagged as ``disagreement_regions``
   — the UI should mark them with a hatched / warning overlay.

The pair-wise ``Jaccard`` over thresholded-top-pct masks gives a
scalar consensus score per pair; the mean across all pairs is the
overall ``consensus_score``.

This module is pure NumPy.  No assumption about the explanation method
producing the saliency map — it operates on the normalised grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

DEFAULT_TOP_PCT = 0.20
DEFAULT_CONSENSUS_RESOLUTION = 16
EPSILON = 1e-9


@dataclass(frozen=True)
class ConsensusResult:
    """Aggregate consensus across two or more saliency maps.

    Attributes:
        agreement_grid: ``(R, R)`` float32 in ``[0, 1]``; cell value is
            the fraction of methods that included that cell in their
            top-``top_pct``.
        consensus_score: Mean pair-wise Jaccard of the top-pct binary
            masks across all unique method pairs.
        pairwise_jaccard: ``{ "<a>|<b>": jaccard, ... }`` for every
            unordered pair of method names.
        disagreement_regions: List of ``(row, col)`` cells flagged in
            exactly one method's top-pct.  Sorted by descending sum of
            saliency across methods (most-disputed-first).
        method_names: Echo of the input order.
        top_pct: Threshold used.
    """

    agreement_grid: np.ndarray
    consensus_score: float
    pairwise_jaccard: dict[str, float]
    disagreement_regions: list[tuple[int, int]]
    method_names: list[str] = field(default_factory=list)
    top_pct: float = DEFAULT_TOP_PCT


def compute_consensus(
    saliency_maps: list[tuple[str, np.ndarray]],
    *,
    top_pct: float = DEFAULT_TOP_PCT,
    resolution: int | None = None,
) -> ConsensusResult:
    """Compute the consensus across a list of named saliency maps.

    Args:
        saliency_maps: ``[(name, grid), ...]``.  All grids are resampled
            (nearest-neighbour) to a common ``resolution`` before
            comparison; the default uses the grid resolution of the
            first map.
        top_pct: Fraction of cells each method keeps as its "top"
            region for the Jaccard / disagreement step.
        resolution: Override the comparison grid size.

    Returns:
        :class:`ConsensusResult`.
    """
    if len(saliency_maps) < 2:
        raise ValueError("consensus requires at least two saliency maps")
    if not 0.0 < top_pct < 1.0:
        raise ValueError(f"top_pct must be in (0, 1), got {top_pct!r}")

    names = [name for name, _ in saliency_maps]
    grids = [grid for _, grid in saliency_maps]
    R = resolution or grids[0].shape[0]

    # Resample every grid to (R, R) using nearest-neighbour — cheap and
    # preserves the top-pct ordering for downstream comparison.
    rs = np.stack([_resample_nn(g, R, R) for g in grids], axis=0)
    # Per-method binary mask of top-pct cells.
    masks = np.stack([_top_pct_mask(g, top_pct) for g in rs], axis=0)

    # Agreement = fraction of methods that flagged each cell.
    agreement = masks.astype(np.float32).mean(axis=0)

    # Pairwise Jaccard.
    pairwise: dict[str, float] = {}
    if len(masks) >= 2:
        for (i, a), (j, b) in combinations(enumerate(masks), 2):
            j_score = _jaccard(a, b)
            key = f"{names[i]}|{names[j]}"
            pairwise[key] = j_score
        consensus_score = float(np.mean(list(pairwise.values())))
    else:
        consensus_score = 1.0

    # Disagreement: cells in exactly one mask.
    method_count = masks.sum(axis=0)
    disagreement_mask = method_count == 1
    # Sort by total saliency across methods so the UI surfaces the
    # most-disputed-but-actually-salient cells first.
    rows, cols = np.where(disagreement_mask)
    saliency_sum = rs.sum(axis=0)
    order = np.argsort(-saliency_sum[rows, cols])
    disagreement_regions = [
        (int(rows[k]), int(cols[k])) for k in order
    ]

    return ConsensusResult(
        agreement_grid=agreement,
        consensus_score=consensus_score,
        pairwise_jaccard=pairwise,
        disagreement_regions=disagreement_regions,
        method_names=names,
        top_pct=float(top_pct),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resample_nn(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resample of a 2-D grid to (target_h, target_w)."""
    h, w = grid.shape
    if (h, w) == (target_h, target_w):
        return grid.astype(np.float32, copy=False)
    ys = ((np.arange(target_h) + 0.5) / target_h * h).astype(np.int32).clip(0, h - 1)
    xs = ((np.arange(target_w) + 0.5) / target_w * w).astype(np.int32).clip(0, w - 1)
    return grid[np.ix_(ys, xs)].astype(np.float32)


def _top_pct_mask(grid: np.ndarray, top_pct: float) -> np.ndarray:
    """Boolean mask of the top ``top_pct`` cells by saliency."""
    flat = grid.flatten()
    n = flat.size
    k = max(1, int(round(n * top_pct)))
    threshold = np.partition(flat, n - k)[n - k]
    return grid >= threshold


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index of two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


__all__ = [
    "DEFAULT_TOP_PCT",
    "DEFAULT_CONSENSUS_RESOLUTION",
    "ConsensusResult",
    "compute_consensus",
]
