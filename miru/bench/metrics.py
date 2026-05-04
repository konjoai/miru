"""Saliency metrics — how well does an attention map track a ground-truth mask.

All metrics treat the attention map as a 2-D float field and the ground
truth as a binary mask.  The attention map is bilinearly upsampled to the
mask's spatial dimensions before scoring, so backends can return any
resolution without affecting the comparison.

Three metrics are reported:

- **iou_at_topk_pct(attn, mask, top_pct)** — binarise the attention map
  by keeping the top ``top_pct`` of pixels by score, then compute
  intersection-over-union with the ground-truth mask.  Useful when the
  benchmark cares whether the model's high-confidence region overlaps
  the truth.

- **auc_roc(attn, mask)** — pixel-level area under the ROC curve.  No
  threshold needed.  ``1.0`` is perfect, ``0.5`` is chance, ``0.0`` is
  perfectly inverted.

- **hit_at_k(attn, mask, k)** — fraction of the top-``k`` attention
  pixels that fall inside the mask.  ``hit_at_k(..., k=1)`` answers
  "did the argmax land inside the ground-truth region?"
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------


def bilinear_upsample(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinearly resample a 2-D float grid to the target shape.

    Pure NumPy.  Coordinates use the convention that the source corners
    map exactly to the target corners (``align_corners=True`` in PyTorch).
    """
    src_h, src_w = grid.shape
    if (src_h, src_w) == (target_h, target_w):
        return grid.astype(np.float32, copy=False)
    if src_h < 2 or src_w < 2:
        # Degenerate — broadcast the single value.
        return np.full((target_h, target_w), float(grid.mean()), dtype=np.float32)

    ys = np.linspace(0, src_h - 1, target_h, dtype=np.float32)
    xs = np.linspace(0, src_w - 1, target_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32); y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.minimum(x0 + 1, src_w - 1)
    dy = (ys - y0).reshape(-1, 1)
    dx = (xs - x0).reshape(1, -1)

    g = grid.astype(np.float32)
    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x1)]
    c = g[np.ix_(y1, x0)]
    d = g[np.ix_(y1, x1)]
    top = a * (1 - dx) + b * dx
    bot = c * (1 - dx) + d * dx
    return (top * (1 - dy) + bot * dy).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def iou_at_topk_pct(
    attn: np.ndarray, mask: np.ndarray, top_pct: float = 0.20
) -> float:
    """Threshold *attn* at the top ``top_pct`` and return IoU vs *mask*.

    Args:
        attn: 2-D float attention map (any resolution, any range).
        mask: 2-D bool ground-truth mask.
        top_pct: Fraction of pixels to keep as the predicted positive
            region (e.g. ``0.20`` → top 20% by score).

    Returns:
        Float in ``[0, 1]``.  Returns ``0.0`` if the union is empty
        (which happens only for a fully-empty ground truth).
    """
    if not 0.0 < top_pct < 1.0:
        raise ValueError(f"top_pct must be in (0, 1), got {top_pct!r}")
    h, w = mask.shape
    up = bilinear_upsample(attn, h, w)
    flat = up.flatten()
    n = flat.size
    k = max(1, int(round(n * top_pct)))
    threshold = np.partition(flat, n - k)[n - k]
    pred = up >= threshold
    return _iou(pred, mask)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def auc_roc(attn: np.ndarray, mask: np.ndarray) -> float:
    """Pixel-level AUC-ROC of attention scores against the binary mask.

    Returns ``0.5`` for a degenerate ground truth (all-positive or
    all-negative) — chance level — rather than raising.
    """
    h, w = mask.shape
    up = bilinear_upsample(attn, h, w).flatten()
    labels = mask.flatten().astype(np.int32)

    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Mann-Whitney U formulation: AUC = U / (n_pos * n_neg).
    # Use rankdata-equivalent: sort once, average ranks for ties.
    order = np.argsort(up, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(up) + 1, dtype=np.float64)

    # Tie correction: average the ranks of equal values.
    sorted_scores = up[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # +1 for 1-based, +1 for inclusive end
            ranks[order[i : j + 1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def hit_at_k(attn: np.ndarray, mask: np.ndarray, k: int = 1) -> float:
    """Fraction of the top-*k* attention pixels that fall inside *mask*.

    Args:
        attn: 2-D attention map.
        mask: 2-D bool ground truth, same logical region but possibly
            different resolution; gets resampled to ``attn``'s shape so
            we don't artificially inflate ``k`` by upsampling first.
        k: Number of top pixels to keep.  Clamped to 1..attn.size.

    Returns:
        Float in ``[0, 1]``.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    h, w = attn.shape
    mh, mw = mask.shape
    # Downsample mask onto attention grid via nearest-neighbour at cell centres.
    if (mh, mw) != (h, w):
        ys = ((np.arange(h) + 0.5) / h * mh).astype(np.int32).clip(0, mh - 1)
        xs = ((np.arange(w) + 0.5) / w * mw).astype(np.int32).clip(0, mw - 1)
        mask_on_attn = mask[np.ix_(ys, xs)]
    else:
        mask_on_attn = mask

    flat = attn.flatten()
    k_clamped = min(k, flat.size)
    idxs = np.argpartition(flat, -k_clamped)[-k_clamped:]
    rows, cols = np.unravel_index(idxs, attn.shape)
    hits = mask_on_attn[rows, cols].sum()
    return float(hits / k_clamped)


__all__ = ["bilinear_upsample", "iou_at_topk_pct", "auc_roc", "hit_at_k"]
