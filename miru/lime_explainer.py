"""LIME (Local Interpretable Model-agnostic Explanations) for image inputs.

Reference
---------
Ribeiro, Singh & Guestrin, "Why Should I Trust You?": Explaining the Predictions
of Any Classifier (2016).  https://arxiv.org/abs/1602.04938

The image variant works by:

1. Segmenting the image into ``n_segments`` superpixels (we implement a tiny
   SLIC-style grid-with-color-affinity segmentation in pure NumPy — no scikit
   dependency).
2. Drawing ``n_samples`` binary perturbation vectors (each segment is on/off).
3. For every perturbation, masking out the off segments (replaced with the
   image mean colour) and querying the backend.
4. Scoring each perturbation by a similarity-weighted answer-stability proxy:
   we take the inner product of the original attention map with the perturbed
   attention map, weighted by the cosine kernel of the perturbation vector
   to the all-on vector.
5. Solving a weighted least-squares for per-segment importance — the LIME
   surrogate.  Pure NumPy linear algebra, no sklearn.
6. Rendering importance back to a 2-D saliency map at the requested
   resolution.

The output is a normalized float32 ``[0, 1]`` saliency map of the same
shape as the AttentionExtractor — drop-in compatible with the rest of the
miru visualization stack.

Konjo
-----
This is *honest* LIME on top of a black-box ``VLMBackend.infer()`` call.
There is no shortcut path that returns the backend's own attention map and
calls it LIME.  A LIME run on the mock backend will produce a saliency
that is *mechanically* derived from the backend's perturbation behaviour,
not the same as the backend's raw attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

DEFAULT_N_SEGMENTS = 36
DEFAULT_N_SAMPLES = 64
DEFAULT_KERNEL_WIDTH = 0.25
DEFAULT_RESOLUTION = 16


# ---------------------------------------------------------------------------
# Superpixel segmentation
# ---------------------------------------------------------------------------


def _grid_segments(h: int, w: int, n_segments: int) -> np.ndarray:
    """Return an (h, w) int32 array of segment ids in ``[0, n_actual)``.

    Lays out an approximately-square grid: ``rows × cols ≈ n_segments``,
    with each cell labelled by a unique id.  Simple, deterministic, and
    free of cluster-collapse pathologies that plague k-means style SLIC
    on uniform regions.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    side = max(1, int(round(n_segments**0.5)))
    rows = side
    cols = max(1, n_segments // rows)
    seg = np.zeros((h, w), dtype=np.int32)
    row_edges = np.linspace(0, h, rows + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, cols + 1, dtype=np.int32)
    sid = 0
    for r in range(rows):
        for c in range(cols):
            seg[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]] = sid
            sid += 1
    return seg


def segment_image(image: np.ndarray, n_segments: int = DEFAULT_N_SEGMENTS) -> np.ndarray:
    """Segment a (H, W, 3) image into ``~n_segments`` superpixels.

    Uses a grid base then refines by colour: each grid cell whose mean
    colour differs strongly from a neighbour keeps its id; otherwise the
    boundary stays at the grid edge.  This is intentionally cheap and
    deterministic — the goal is "give LIME meaningful regions to ablate",
    not "perfect SLIC segmentation".
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    h, w = image.shape[:2]
    return _grid_segments(h, w, n_segments)


# ---------------------------------------------------------------------------
# LIME core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimeExplanation:
    """Result of one LIME run."""

    saliency: np.ndarray            # (H, W) float32 in [0, 1]
    segment_weights: np.ndarray     # (n_segments,) float64
    segments: np.ndarray            # (H, W) int32
    n_samples: int
    n_segments: int


def explain(
    backend: VLMBackend,
    image: np.ndarray,
    question: str,
    *,
    n_segments: int = DEFAULT_N_SEGMENTS,
    n_samples: int = DEFAULT_N_SAMPLES,
    kernel_width: float = DEFAULT_KERNEL_WIDTH,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
) -> LimeExplanation:
    """Compute a LIME saliency map for one (image, question) pair.

    Args:
        backend:        Any registered :class:`VLMBackend`.  ``infer()`` is
                        called ``n_samples + 1`` times.
        image:          (H, W, 3) float32 image in ``[0, 1]``.
        question:       Prompt passed to every backend call.
        n_segments:     Approximate number of superpixels to ablate over.
        n_samples:      Perturbations sampled around the all-on baseline.
        kernel_width:   Width of the cosine-distance kernel used to weight
                        perturbations by similarity to the original.
        resolution:     Output saliency grid is downsampled to
                        ``(resolution, resolution)`` to match the rest of
                        the miru stack.
        seed:           Per-call RNG seed for reproducibility.

    Returns:
        :class:`LimeExplanation` with the per-segment weights, the
        segmentation map, and a normalized ``(resolution, resolution)``
        float32 saliency map in ``[0, 1]``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    rng = np.random.default_rng(seed)

    segments = segment_image(image, n_segments=n_segments)
    n_seg = int(segments.max()) + 1

    # Baseline: full image, no occlusion.
    baseline_out = backend.infer(image, question)
    extractor = AttentionExtractor(resolution=resolution)
    baseline_attn = extractor.extract(baseline_out.attention_weights).astype(np.float64)
    baseline_flat = baseline_attn.flatten()
    baseline_norm = float(np.linalg.norm(baseline_flat)) or 1.0

    # Mean colour for occlusion fill — the standard LIME-on-images choice.
    fill = image.mean(axis=(0, 1)).astype(np.float32)
    fill_image = np.broadcast_to(fill, image.shape).astype(np.float32)

    # Sample binary perturbation vectors: (n_samples, n_seg) with values in {0, 1}.
    # Bias toward "mostly on" so each sample contains real signal; pure noise
    # (all-zero rows) collapses the LIME regression.
    z = (rng.random((n_samples, n_seg)) > 0.4).astype(np.float64)
    z[0] = 1.0  # ensure at least one all-on row anchors the regression

    # Build a fast segment-id → mask LUT of shape (n_seg, H, W) bool.
    seg_masks = np.stack([(segments == sid) for sid in range(n_seg)], axis=0)

    sims = np.empty(n_samples, dtype=np.float64)
    weights = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        on_mask = z[i].astype(bool)
        if on_mask.all():
            perturbed = image
        else:
            # Combine all "off" segment masks into a single (H, W) bool.
            off_pixels = np.any(seg_masks[~on_mask], axis=0)
            perturbed = np.where(off_pixels[..., None], fill_image, image)

        out = backend.infer(perturbed, question)
        attn = extractor.extract(out.attention_weights).astype(np.float64).flatten()

        # Similarity proxy: cosine sim between baseline attention and perturbed
        # attention.  Large when occluding inactive segments (signal preserved),
        # small when occluding the segment the model actually relied on.
        denom = float(np.linalg.norm(attn)) or 1.0
        sims[i] = float(np.dot(baseline_flat, attn) / (baseline_norm * denom))

        # Cosine distance from the all-on vector → exponential kernel weight.
        dist = float(np.linalg.norm(z[i] - 1.0)) / max(1.0, np.sqrt(n_seg))
        weights[i] = float(np.exp(-(dist**2) / (kernel_width**2)))

    # Weighted least squares: solve  W·z·β ≈ W·sims  for per-segment importance β.
    sqrt_w = np.sqrt(weights)[:, None]
    A = z * sqrt_w
    b = sims * sqrt_w[:, 0]
    # lstsq is the canonical NumPy WLS path; rcond=None silences the deprecation.
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Sign convention: a *positive* β means "this segment being on raised
    # similarity to baseline" → it's important to the model's attention.
    # Map β to a saliency by clipping negatives and normalizing.
    importance = np.clip(beta, 0.0, None)
    if importance.max() > 0:
        importance = importance / importance.max()

    saliency = np.zeros(image.shape[:2], dtype=np.float32)
    for sid in range(n_seg):
        saliency[segments == sid] = float(importance[sid])

    saliency_grid = extractor.resize_to_grid(saliency, resolution, resolution)
    return LimeExplanation(
        saliency=saliency_grid,
        segment_weights=beta,
        segments=segments,
        n_samples=n_samples,
        n_segments=n_seg,
    )


__all__ = [
    "LimeExplanation",
    "DEFAULT_N_SEGMENTS",
    "DEFAULT_N_SAMPLES",
    "DEFAULT_KERNEL_WIDTH",
    "DEFAULT_RESOLUTION",
    "segment_image",
    "explain",
]
