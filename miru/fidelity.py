"""Explanation-fidelity scorecard via the deletion test.

The deletion test answers the question: *if I remove the pixels the
explanation says are most important, does the model's confidence
actually drop?*  If yes, the explanation is faithful — the highlighted
region really does drive the prediction.  If not, the explanation is
decorative, not causal.

Method
------

1. Run the backend on the original image; record ``baseline_confidence``.
2. Build a binary "delete mask" from the saliency map: mark the top
   ``k_pct`` of pixels by saliency score as deleted.
3. Replace those pixels with the per-image mean colour (a neutral
   occlusion that doesn't introduce strong gradients of its own).
4. Re-run the backend on the masked image; record ``masked_confidence``.
5. ``fidelity_score = max(0, (baseline - masked) / baseline)``,
   clamped to ``[0, 1]``.  A flat (uniform) saliency map should score
   near zero; a perfectly-targeted saliency map should score near one.

A score below ``LOW_FIDELITY_THRESHOLD`` (default 0.5) flags the
explanation as suspect — the UI surfaces a warning chip.

Konjo
-----
This is a real, citable explanation-evaluation method (Petsiuk, Das &
Saenko, "RISE: Randomized Input Sampling for Explanation of Black-box
Models", 2018; Samek et al. 2017).  No shortcut path that returns the
backend's own confidence verbatim — the test always runs an extra
``backend.infer()`` call on a real masked image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.models.base import VLMBackend

DEFAULT_K_PCT = 0.10
LOW_FIDELITY_THRESHOLD = 0.5
EPSILON = 1e-9


@dataclass(frozen=True)
class FidelityResult:
    """Outcome of one deletion test.

    Attributes:
        fidelity_score: Drop in confidence as a fraction of baseline.
            ``1.0`` means full collapse, ``0.0`` means no change.
        baseline_confidence: Confidence on the original image.
        masked_confidence: Confidence after masking top-K% salient pixels.
        k_pct: Fraction of pixels masked (matches the parameter).
        low_fidelity: True iff ``fidelity_score < LOW_FIDELITY_THRESHOLD``.
    """

    fidelity_score: float
    baseline_confidence: float
    masked_confidence: float
    k_pct: float
    low_fidelity: bool


def deletion_test(
    backend: VLMBackend,
    image: np.ndarray,
    prompt: str,
    saliency_map: np.ndarray,
    *,
    k_pct: float = DEFAULT_K_PCT,
    baseline_confidence: float | None = None,
) -> FidelityResult:
    """Mask top-K% salient pixels, re-run inference, score the drop.

    Args:
        backend: The same ``VLMBackend`` that produced the saliency map.
        image: float32 ``(H, W, 3)`` array in ``[0, 1]`` — the original.
        prompt: The question conditioning the backend.
        saliency_map: 2-D saliency map (any resolution; bilinearly
            upsampled to image space here).
        k_pct: Fraction of pixels to mask, in ``(0, 1)``.
        baseline_confidence: Pre-computed confidence to skip the
            baseline call.  Pass it whenever the caller has already
            run ``backend.infer()`` on the unmasked image so the
            fidelity test costs one extra inference instead of two.

    Returns:
        :class:`FidelityResult`.
    """
    if not 0.0 < k_pct < 1.0:
        raise ValueError(f"k_pct must be in (0, 1), got {k_pct!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be (H, W, 3) float, got shape {image.shape}"
        )

    if baseline_confidence is None:
        baseline_confidence = float(backend.infer(image, prompt).confidence)

    masked = _mask_top_k(image, saliency_map, k_pct)
    masked_out = backend.infer(masked, prompt)
    masked_confidence = float(masked_out.confidence)

    fidelity = (baseline_confidence - masked_confidence) / max(
        EPSILON, baseline_confidence
    )
    fidelity = float(max(0.0, min(1.0, fidelity)))

    return FidelityResult(
        fidelity_score=fidelity,
        baseline_confidence=baseline_confidence,
        masked_confidence=masked_confidence,
        k_pct=float(k_pct),
        low_fidelity=fidelity < LOW_FIDELITY_THRESHOLD,
    )


def _mask_top_k(
    image: np.ndarray, saliency_map: np.ndarray, k_pct: float
) -> np.ndarray:
    """Return a copy of ``image`` with the top-K% salient pixels neutralised.

    The saliency map is bilinearly upsampled to image shape, then a
    binary mask covers the top ``k_pct`` of pixels by score.  Those
    pixels are replaced with the per-image mean colour — a neutral
    occlusion that doesn't introduce strong artificial gradients.
    """
    h, w = image.shape[:2]
    up = _bilinear(saliency_map, h, w)
    flat = up.flatten()
    k = max(1, int(round(flat.size * k_pct)))
    threshold = np.partition(flat, flat.size - k)[flat.size - k]
    mask = up >= threshold

    # Per-image mean colour: a (1, 1, 3) fill that erases local detail
    # without smearing local averages back in.
    mean_colour = image.reshape(-1, image.shape[2]).mean(axis=0)
    masked = image.copy()
    # Broadcast the (3,) colour over the (H, W) mask.
    masked[mask] = mean_colour.astype(image.dtype)
    return masked


def _bilinear(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Tiny bilinear resampler used inline to avoid a SciPy dependency."""
    src_h, src_w = grid.shape
    if (src_h, src_w) == (target_h, target_w):
        return grid.astype(np.float32, copy=False)
    if src_h < 2 or src_w < 2:
        return np.full(
            (target_h, target_w), float(grid.mean()), dtype=np.float32
        )
    ys = np.linspace(0, src_h - 1, target_h, dtype=np.float32)
    xs = np.linspace(0, src_w - 1, target_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32); y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.minimum(x0 + 1, src_w - 1)
    dy = (ys - y0).reshape(-1, 1)
    dx = (xs - x0).reshape(1, -1)
    g = grid.astype(np.float32)
    a = g[np.ix_(y0, x0)]; b = g[np.ix_(y0, x1)]
    c = g[np.ix_(y1, x0)]; d = g[np.ix_(y1, x1)]
    top = a * (1 - dx) + b * dx
    bot = c * (1 - dx) + d * dx
    return (top * (1 - dy) + bot * dy).astype(np.float32)


__all__ = [
    "DEFAULT_K_PCT",
    "LOW_FIDELITY_THRESHOLD",
    "FidelityResult",
    "deletion_test",
]
