"""Integrated attention — path-integrated saliency along the baseline interpolation.

Analogous to Integrated Gradients (Sundararajan et al. 2017) but for attention-
based explanations that do not expose gradients.

Algorithm
---------
Given an image ``x`` and a baseline ``b`` (default: all-zeros):

1. Construct ``n_steps`` interpolated images:
   ``x_k = b + (k / (n_steps - 1)) * (x - b)``  for k in 0 .. n_steps-1.
2. Run ``backend.infer(x_k, question)`` for each step and extract the
   normalised attention grid via :class:`~miru.attention.extractor.AttentionExtractor`.
3. Integrate (average) the per-step grids:
   ``integrated = mean_k [attention(x_k)]``.
4. Min-max normalise the result to ``[0, 1]``.

The integrated grid highlights regions whose attention score rises most
consistently as the image is revealed from the baseline — a more faithful
attribution than the single-step attention at full image resolution.

References
----------
Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for
Deep Networks*. ICML 2017.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_N_STEPS: int = 20
MIN_N_STEPS: int = 2
MAX_N_STEPS: int = 100


@dataclass(frozen=True)
class IntegratedAttentionResult:
    """Output of the integrated attention explainer.

    Attributes:
        integrated_grid: Path-averaged attention, float32 ``(grid_h, grid_w)``
            in ``[0, 1]``.
        n_steps: Number of interpolation steps actually run.
        grid_h: Height of the output grid.
        grid_w: Width of the output grid.
    """

    integrated_grid: np.ndarray
    n_steps: int
    grid_h: int
    grid_w: int


class IntegratedAttention:
    """Path-integrated attention explainer.

    Args:
        n_steps: Number of interpolation steps from baseline to image.
            Must be in ``[MIN_N_STEPS, MAX_N_STEPS]``.
        baseline: Baseline image strategy.  ``"black"`` (default) uses
            an all-zeros image; ``"mean"`` uses the per-channel mean of
            the input image.
        extractor: :class:`~miru.attention.extractor.AttentionExtractor`
            for normalisation and grid resizing.
    """

    def __init__(
        self,
        n_steps: int = DEFAULT_N_STEPS,
        baseline: str = "black",
        extractor: AttentionExtractor | None = None,
    ) -> None:
        if not MIN_N_STEPS <= n_steps <= MAX_N_STEPS:
            raise ValueError(
                f"n_steps must be in [{MIN_N_STEPS}, {MAX_N_STEPS}], got {n_steps}"
            )
        if baseline not in ("black", "mean"):
            raise ValueError(f"baseline must be 'black' or 'mean', got {baseline!r}")
        self._n_steps = n_steps
        self._baseline = baseline
        self._extractor = extractor or AttentionExtractor()

    def explain(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> IntegratedAttentionResult:
        """Run integrated attention on *image_array*.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question.

        Returns:
            :class:`IntegratedAttentionResult` with the integrated grid.
        """
        if self._baseline == "black":
            base = np.zeros_like(image_array)
        else:
            base = np.full_like(image_array, image_array.mean())

        alphas = np.linspace(0.0, 1.0, self._n_steps, dtype=np.float32)
        grids: list[np.ndarray] = []

        for alpha in alphas:
            interp = (base + alpha * (image_array - base)).astype(np.float32)
            interp = np.clip(interp, 0.0, 1.0)
            try:
                out = backend.infer(interp, question)
                grid = self._extractor.extract(out.attention_weights)
                grids.append(grid)
            except Exception:
                logger.warning(
                    "integrated_attention: step alpha=%.3f failed; skipping", float(alpha)
                )

        resolution = self._extractor.resolution
        if not grids:
            logger.warning("integrated_attention: all steps failed; returning zeros")
            zero = np.zeros((resolution, resolution), dtype=np.float32)
            return IntegratedAttentionResult(
                integrated_grid=zero, n_steps=0, grid_h=resolution, grid_w=resolution
            )

        stacked = np.stack(grids, axis=0).astype(np.float32)
        integrated = stacked.mean(axis=0)

        lo, hi = float(integrated.min()), float(integrated.max())
        if hi - lo > 1e-8:
            integrated = ((integrated - lo) / (hi - lo)).astype(np.float32)

        h, w = integrated.shape
        return IntegratedAttentionResult(
            integrated_grid=integrated,
            n_steps=len(grids),
            grid_h=h,
            grid_w=w,
        )
