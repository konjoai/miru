"""Dataset-level saliency analytics.

Given a collection of saliency grids (one per image), this module produces:

- **Aggregate heatmap** — cell-wise mean of all normalised saliency grids,
  giving a dataset-level picture of where models consistently attend.
- **Spurious-correlation detection** — identifies cells whose mean saliency
  exceeds a threshold *and* whose coefficient of variation is low (i.e. the
  model attends there consistently, regardless of image content).  Such cells
  are candidates for dataset artefacts — watermarks, borders, fixed overlays —
  rather than semantically meaningful regions.

The detection is purely statistical: it flags *candidates* for human review,
not confirmed artefacts.  Callers should inspect flagged regions visually.

All inputs are expected to be 2-D float32 arrays in ``[0, 1]`` at the same
resolution (the standard ``AttentionExtractor`` output grid).  Arrays at
different resolutions are bilinearly resampled to the first grid's shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.bench.metrics import bilinear_upsample

SPURIOUS_MEAN_THRESHOLD: float = 0.5
"""Cells with mean saliency above this value are candidates for spurious-corr."""

SPURIOUS_CV_THRESHOLD: float = 0.5
"""Cells with coefficient of variation below this value are flagged (low variance)."""

MIN_SAMPLES_FOR_SPURIOUS: int = 3
"""Spurious detection requires at least this many samples to be meaningful."""


@dataclass(frozen=True)
class DatasetAnalytics:
    """Aggregate saliency statistics over a dataset of images.

    Attributes:
        mean_grid: Cell-wise mean of all saliency grids, float32 in ``[0, 1]``,
            shape ``(grid_h, grid_w)``.
        std_grid: Cell-wise standard deviation, same shape and dtype.
        cv_grid: Coefficient of variation (std / mean), clamped to ``[0, ∞)``.
            Cells with mean < ``1e-6`` are set to ``0.0``.
        spurious_mask: Boolean array marking cells that are both consistently
            high-saliency and low-variance — spurious-correlation candidates.
        spurious_cells: List of ``(row, col)`` tuples for flagged cells,
            sorted by descending mean saliency.
        n_samples: Number of saliency grids that went into the aggregate.
        grid_h: Height of the output grid.
        grid_w: Width of the output grid.
    """

    mean_grid: np.ndarray
    std_grid: np.ndarray
    cv_grid: np.ndarray
    spurious_mask: np.ndarray
    spurious_cells: list[tuple[int, int]]
    n_samples: int
    grid_h: int
    grid_w: int


def aggregate_saliency(
    grids: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell-wise mean and std over a list of 2-D saliency grids.

    All grids are bilinearly resampled to the shape of the first grid before
    aggregation.

    Args:
        grids: Non-empty list of 2-D float arrays.

    Returns:
        ``(mean_grid, std_grid)`` — both float32, shape matching ``grids[0]``.

    Raises:
        ValueError: If *grids* is empty.
    """
    if not grids:
        raise ValueError("grids must be non-empty")
    target_h, target_w = grids[0].shape
    stack = np.stack(
        [
            bilinear_upsample(g.astype(np.float32), target_h, target_w)
            for g in grids
        ],
        axis=0,
    )
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0, ddof=0).astype(np.float32)
    return mean, std


def detect_spurious(
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
    *,
    mean_threshold: float = SPURIOUS_MEAN_THRESHOLD,
    cv_threshold: float = SPURIOUS_CV_THRESHOLD,
    n_samples: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Flag cells that are consistently high-saliency with low variance.

    A cell is a spurious-correlation candidate when:

    1. Its mean saliency ≥ ``mean_threshold`` — the model consistently
       focuses here.
    2. Its coefficient of variation (std / mean) < ``cv_threshold`` — the
       attention is stable across images, not driven by image content.

    The second condition filters out cells that happen to be high on average
    but vary wildly (e.g. a cell that's salient for some images and not
    others is likely content-driven).

    When ``n_samples < MIN_SAMPLES_FOR_SPURIOUS`` no cells are flagged and
    an empty mask is returned — variance estimates from very small samples
    are not reliable.

    Args:
        mean_grid: Cell-wise mean saliency, float32 in ``[0, 1]``.
        std_grid: Cell-wise standard deviation, same shape.
        mean_threshold: Minimum mean to qualify.
        cv_threshold: Maximum CV to qualify.
        n_samples: Number of samples the statistics were computed from.

    Returns:
        ``(spurious_mask, spurious_cells)`` — boolean array + sorted list of
        ``(row, col)`` tuples.
    """
    if n_samples < MIN_SAMPLES_FOR_SPURIOUS:
        return np.zeros(mean_grid.shape, dtype=bool), []

    safe_mean = np.where(mean_grid < 1e-6, 1e-6, mean_grid)
    cv = (std_grid / safe_mean).astype(np.float32)
    cv = np.where(mean_grid < 1e-6, 0.0, cv)

    mask = (mean_grid >= mean_threshold) & (cv < cv_threshold)
    rows, cols = np.where(mask)
    cells = sorted(
        [(int(r), int(c)) for r, c in zip(rows, cols)],
        key=lambda rc: -float(mean_grid[rc[0], rc[1]]),
    )
    return mask, cells


def analyse_dataset(
    grids: list[np.ndarray],
    *,
    mean_threshold: float = SPURIOUS_MEAN_THRESHOLD,
    cv_threshold: float = SPURIOUS_CV_THRESHOLD,
) -> DatasetAnalytics:
    """Full pipeline: aggregate saliency + detect spurious cells.

    Args:
        grids: Non-empty list of 2-D float32 saliency grids in ``[0, 1]``.
            All grids are resampled to the first grid's shape.
        mean_threshold: Forwarded to :func:`detect_spurious`.
        cv_threshold: Forwarded to :func:`detect_spurious`.

    Returns:
        :class:`DatasetAnalytics` with all aggregated statistics.

    Raises:
        ValueError: If *grids* is empty.
    """
    mean_grid, std_grid = aggregate_saliency(grids)

    safe_mean = np.where(mean_grid < 1e-6, 1e-6, mean_grid)
    cv_grid = (std_grid / safe_mean).astype(np.float32)
    cv_grid = np.where(mean_grid < 1e-6, 0.0, cv_grid)

    n = len(grids)
    spurious_mask, spurious_cells = detect_spurious(
        mean_grid, std_grid,
        mean_threshold=mean_threshold,
        cv_threshold=cv_threshold,
        n_samples=n,
    )
    h, w = mean_grid.shape
    return DatasetAnalytics(
        mean_grid=mean_grid,
        std_grid=std_grid,
        cv_grid=cv_grid.astype(np.float32),
        spurious_mask=spurious_mask,
        spurious_cells=spurious_cells,
        n_samples=n,
        grid_h=h,
        grid_w=w,
    )
