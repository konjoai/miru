"""Expert annotation alignment — compare a saliency map against a human mask.

Given a predicted saliency grid and a binary ground-truth mask supplied by a
human annotator, this module computes three complementary scores:

- **IoU @ top-k%** — binarise the saliency at the top ``top_pct`` threshold,
  then compute intersection-over-union with the mask (uses the existing
  :func:`~miru.bench.metrics.iou_at_topk_pct` helper).
- **AUC-ROC** — pixel-level area under the ROC curve; threshold-free
  (uses :func:`~miru.bench.metrics.auc_roc`).
- **Spearman rank correlation** — rank-correlate the flat saliency scores
  against the flat binary mask.  Pure NumPy; no SciPy dependency.

Additionally a ``misaligned`` flag is set when the model's answer matches an
expected answer but the spatial alignment is below a threshold — the
"right answer, wrong reasoning" diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.bench.metrics import auc_roc, bilinear_upsample, iou_at_topk_pct

MISALIGN_THRESHOLD: float = 0.3
"""IoU below this value triggers the ``misaligned`` flag (when answer is correct)."""

DEFAULT_TOP_PCT: float = 0.20


@dataclass(frozen=True)
class AnnotationAlignment:
    """Alignment scores between a saliency map and a human annotation mask.

    Attributes:
        iou: Intersection-over-union at the ``top_pct`` threshold.
        auc_roc: Pixel-level AUC-ROC (threshold-free); 0.5 = chance.
        spearman_r: Spearman rank correlation in ``[-1, 1]``; 0 = no correlation.
        top_pct: The threshold fraction used for ``iou``.
        misaligned: True when ``answer_correct=True`` and ``iou < MISALIGN_THRESHOLD``.
    """

    iou: float
    auc_roc: float
    spearman_r: float
    top_pct: float
    misaligned: bool


def compare_annotation(
    saliency: np.ndarray,
    mask: np.ndarray,
    *,
    answer_correct: bool = False,
    top_pct: float = DEFAULT_TOP_PCT,
) -> AnnotationAlignment:
    """Score *saliency* against the binary ground-truth *mask*.

    Args:
        saliency: 2-D float32 attention/saliency grid in ``[0, 1]``; any
            resolution — upsampled to ``mask``'s shape before scoring.
        mask: 2-D bool or 0/1 array (any numeric dtype) representing the
            annotated region.  Need not match ``saliency``'s resolution.
        answer_correct: Whether the model's answer matched the expected
            answer.  Used only for the ``misaligned`` flag.
        top_pct: Fraction of pixels used as threshold for the IoU metric.

    Returns:
        :class:`AnnotationAlignment` with all four scores plus the flag.

    Raises:
        ValueError: If ``saliency`` or ``mask`` is not 2-D, or ``mask`` is
            empty, or ``top_pct`` is not in ``(0, 1)``.
    """
    if saliency.ndim != 2:
        raise ValueError(f"saliency must be 2-D, got shape {saliency.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if not 0.0 < top_pct < 1.0:
        raise ValueError(f"top_pct must be in (0, 1), got {top_pct!r}")

    bool_mask = mask.astype(bool)
    iou = iou_at_topk_pct(saliency, bool_mask, top_pct=top_pct)
    auc = auc_roc(saliency, bool_mask)
    rho = _spearman(saliency, bool_mask)
    misaligned = answer_correct and (iou < MISALIGN_THRESHOLD)

    return AnnotationAlignment(
        iou=iou,
        auc_roc=auc,
        spearman_r=rho,
        top_pct=top_pct,
        misaligned=misaligned,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Spearman rank correlation between flattened saliency and mask.

    Returns 0.0 when either series is constant (no rank variation).
    Pure NumPy — no SciPy dependency.
    """
    h, w = mask.shape
    sal_up = bilinear_upsample(saliency, h, w).flatten().astype(np.float64)
    gt = mask.flatten().astype(np.float64)

    rank_sal = _rank(sal_up)
    rank_gt = _rank(gt)

    n = len(rank_sal)
    mean_s = rank_sal.mean()
    mean_g = rank_gt.mean()
    d_s = rank_sal - mean_s
    d_g = rank_gt - mean_g
    num = float((d_s * d_g).sum())
    denom = float(np.sqrt((d_s**2).sum() * (d_g**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


def _rank(x: np.ndarray) -> np.ndarray:
    """Assign average ranks to a 1-D array, handling ties."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks
