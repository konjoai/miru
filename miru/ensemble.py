"""Scale-space attention ensemble.

Runs a VLM backend at multiple image scales and averages the resulting
attention maps.  Single-scale attention captures only 52–75% of the true
saliency signal (Intra-modal Token Interactions, arXiv 2509.22415); a
multi-scale ensemble is more robust and less sensitive to the arbitrary
choice of input resolution.

Algorithm
---------
For each scale factor ``s ∈ scales``:

1. Resize the source image to ``(round(H*s), round(W*s))`` via bilinear
   interpolation (pure NumPy — no PIL required).
2. Run ``backend.infer(scaled_image, question)`` to get raw attention weights.
3. Normalise and resize the attention map to the target grid via
   :class:`~miru.attention.extractor.AttentionExtractor`.

The per-scale grids are averaged with optional per-scale weights (default:
uniform).  The final grid is min-max normalised to ``[0, 1]``.

If inference fails for a scale (e.g. a degenerate 0×0 resize), that scale
is skipped with a warning and excluded from the average.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_SCALES: tuple[float, ...] = (0.5, 1.0, 1.5)
MIN_DIM: int = 4  # minimum spatial dimension after scaling


@dataclass(frozen=True)
class EnsembleResult:
    """Output of a multi-scale attention ensemble.

    Attributes:
        ensemble_grid: Weighted-average attention grid, float32 ``(grid_h, grid_w)``
            in ``[0, 1]``.
        per_scale: List of ``(scale, grid)`` pairs for every scale that
            succeeded.  Each ``grid`` is float32 ``(grid_h, grid_w)`` in
            ``[0, 1]``.
        scales_used: Subset of requested scales for which inference succeeded.
        scales_skipped: Scales that were skipped due to inference errors or
            degenerate image dimensions.
        grid_h: Height of the output grid.
        grid_w: Width of the output grid.
    """

    ensemble_grid: np.ndarray
    per_scale: list[tuple[float, np.ndarray]]
    scales_used: list[float]
    scales_skipped: list[float]
    grid_h: int
    grid_w: int


class AttentionEnsemble:
    """Run a VLM backend at multiple image scales and average the attention.

    Args:
        scales: Iterable of scale factors (relative to the input image size).
            At least one scale must be ≥ 1 pixel on each axis after rounding.
        weights: Optional per-scale weights; must have the same length as
            *scales*.  Weights do not need to be normalised — they are divided
            by their sum internally.  Defaults to uniform.
        extractor: :class:`~miru.attention.extractor.AttentionExtractor` for
            normalisation and grid resizing.  Uses the module default if *None*.
    """

    def __init__(
        self,
        scales: tuple[float, ...] = DEFAULT_SCALES,
        weights: tuple[float, ...] | None = None,
        extractor: AttentionExtractor | None = None,
    ) -> None:
        if not scales:
            raise ValueError("scales must be non-empty")
        if weights is not None and len(weights) != len(scales):
            raise ValueError("weights must have the same length as scales")
        self._scales = tuple(float(s) for s in scales)
        self._weights = tuple(float(w) for w in weights) if weights else None
        self._extractor = extractor or AttentionExtractor()

    def run(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> EnsembleResult:
        """Run multi-scale inference and return the ensembled attention.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question.

        Returns:
            :class:`EnsembleResult` with the averaged grid and per-scale maps.
        """
        resolution = self._extractor.resolution
        grids: list[np.ndarray] = []
        effective_weights: list[float] = []
        per_scale: list[tuple[float, np.ndarray]] = []
        skipped: list[float] = []

        for idx, scale in enumerate(self._scales):
            scaled = _bilinear_resize_image(image_array, scale)
            if scaled is None:
                logger.warning("ensemble: scale %.2f skipped — image too small", scale)
                skipped.append(scale)
                continue
            try:
                out = backend.infer(scaled, question)
            except Exception:
                logger.warning("ensemble: scale %.2f skipped — inference failed", scale)
                skipped.append(scale)
                continue
            grid = self._extractor.extract(out.attention_weights)
            grids.append(grid)
            per_scale.append((scale, grid))
            w = self._weights[idx] if self._weights else 1.0
            effective_weights.append(w)

        if not grids:
            logger.warning("ensemble: all scales failed; returning zeros")
            zero = np.zeros((resolution, resolution), dtype=np.float32)
            return EnsembleResult(
                ensemble_grid=zero,
                per_scale=[],
                scales_used=[],
                scales_skipped=list(self._scales),
                grid_h=resolution,
                grid_w=resolution,
            )

        total_w = sum(effective_weights)
        stack = np.stack(grids, axis=0).astype(np.float32)
        weight_arr = np.array(effective_weights, dtype=np.float32) / total_w
        weighted = (stack * weight_arr[:, None, None]).sum(axis=0)

        # Re-normalise so the ensemble is always in [0, 1].
        lo, hi = float(weighted.min()), float(weighted.max())
        if hi - lo > 1e-8:
            ensemble_grid = ((weighted - lo) / (hi - lo)).astype(np.float32)
        else:
            ensemble_grid = weighted

        h, w = ensemble_grid.shape
        return EnsembleResult(
            ensemble_grid=ensemble_grid,
            per_scale=per_scale,
            scales_used=[s for s, _ in per_scale],
            scales_skipped=skipped,
            grid_h=h,
            grid_w=w,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bilinear_resize_image(
    image: np.ndarray, scale: float
) -> np.ndarray | None:
    """Resize a float32 (H, W, 3) image by *scale* via bilinear interpolation.

    Returns *None* when the resulting dimensions are below ``MIN_DIM``.
    """
    h, w = image.shape[:2]
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    if new_h < MIN_DIM or new_w < MIN_DIM:
        return None

    # Pure NumPy bilinear resize — no PIL dependency.
    ys = np.linspace(0, h - 1, new_h, dtype=np.float32)
    xs = np.linspace(0, w - 1, new_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32).clip(0, h - 2)
    y1 = y0 + 1
    x0 = np.floor(xs).astype(np.int32).clip(0, w - 2)
    x1 = x0 + 1
    dy = (ys - y0).reshape(-1, 1, 1)
    dx = (xs - x0).reshape(1, -1, 1)

    a = image[np.ix_(y0, x0, [0, 1, 2])]
    b = image[np.ix_(y0, x1, [0, 1, 2])]
    c = image[np.ix_(y1, x0, [0, 1, 2])]
    d = image[np.ix_(y1, x1, [0, 1, 2])]
    top = a * (1 - dx) + b * dx
    bot = c * (1 - dx) + d * dx
    return (top * (1 - dy) + bot * dy).astype(np.float32)
