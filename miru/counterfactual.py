"""Minimal counterfactual explanation for VLM saliency maps.

Answers the question: *What is the smallest set of image regions that,
when masked, causes the model's confidence to drop by at least
``confidence_drop`` or its answer to change?*

Algorithm
---------
1. Run ``backend.infer(image, question)`` to get the baseline answer and
   confidence, and extract the attention grid via
   :class:`~miru.attention.extractor.AttentionExtractor`.
2. Rank grid cells from most to least salient.
3. Greedily mask cells (replace with ``fill_value``) in saliency rank order,
   calling ``backend.infer`` after each addition until either:

   - ``baseline_confidence − current_confidence ≥ confidence_drop``, or
   - the answer token changes, or
   - ``max_cells`` cells have been masked.

4. Return the resulting binary mask together with the original and
   counterfactual inference outputs.

The result highlights the *minimal sufficient evidence* — the cells whose
removal most destabilises the model's prediction.  This is analogous to
the deletion step in fidelity scoring (see ``miru.fidelity``) but is
goal-directed and stops as early as possible.

References
----------
Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual
Explanations Without Opening the Black Box*.  Harvard Journal of Law &
Technology, 31(2).

Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R.
(2017). *Evaluating the visualization of what a Deep Neural Network has
learned*.  IEEE TNNLS 28(11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_DROP: float = 0.1
DEFAULT_MAX_CELLS: int = 32
DEFAULT_FILL_VALUE: float = 0.0
MIN_CONFIDENCE_DROP: float = 1e-4
MAX_MAX_CELLS: int = 256


@dataclass(frozen=True)
class CounterfactualResult:
    """Output of the minimal counterfactual explainer.

    Attributes:
        counterfactual_mask: Boolean ``(grid_h, grid_w)`` array; ``True``
            where a cell was masked to produce the counterfactual.
        original_answer: Model answer on the unmasked image.
        original_confidence: Model confidence on the unmasked image.
        counterfactual_answer: Model answer after masking.
        counterfactual_confidence: Model confidence after masking.
        delta_confidence: ``original_confidence − counterfactual_confidence``.
        n_cells_masked: Number of cells in the minimal mask.
        grid_h: Height of the attention grid.
        grid_w: Width of the attention grid.
        flipped: Whether the model answer changed after masking.
        goal_reached: Whether the confidence-drop threshold was met or the
            answer flipped before ``max_cells`` was exhausted.
    """

    counterfactual_mask: np.ndarray
    original_answer: str
    original_confidence: float
    counterfactual_answer: str
    counterfactual_confidence: float
    delta_confidence: float
    n_cells_masked: int
    grid_h: int
    grid_w: int
    flipped: bool
    goal_reached: bool


class MinimalCounterfactual:
    """Minimal counterfactual explainer.

    Args:
        confidence_drop: Target drop in confidence to achieve.  Must be
            positive.  A value of ``0.1`` means the mask must reduce
            confidence by at least 10 percentage points from the baseline.
        max_cells: Maximum grid cells to mask before giving up.
        fill_value: Pixel value used to replace masked regions ``[0, 1]``.
        extractor: :class:`~miru.attention.extractor.AttentionExtractor`
            for saliency-based cell ranking.
    """

    def __init__(
        self,
        confidence_drop: float = DEFAULT_CONFIDENCE_DROP,
        max_cells: int = DEFAULT_MAX_CELLS,
        fill_value: float = DEFAULT_FILL_VALUE,
        extractor: AttentionExtractor | None = None,
    ) -> None:
        if confidence_drop < MIN_CONFIDENCE_DROP:
            raise ValueError(
                f"confidence_drop must be ≥ {MIN_CONFIDENCE_DROP}, got {confidence_drop}"
            )
        if not 1 <= max_cells <= MAX_MAX_CELLS:
            raise ValueError(
                f"max_cells must be in [1, {MAX_MAX_CELLS}], got {max_cells}"
            )
        if not 0.0 <= fill_value <= 1.0:
            raise ValueError(
                f"fill_value must be in [0, 1], got {fill_value}"
            )
        self._confidence_drop = confidence_drop
        self._max_cells = max_cells
        self._fill_value = fill_value
        self._extractor = extractor or AttentionExtractor()

    def explain(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> CounterfactualResult:
        """Find the minimal counterfactual mask for *image_array*.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question.

        Returns:
            :class:`CounterfactualResult` with the minimal binary mask.
        """
        baseline = backend.infer(image_array, question)
        saliency = self._extractor.extract(baseline.attention_weights)
        grid_h, grid_w = saliency.shape

        cell_h = image_array.shape[0] / grid_h
        cell_w = image_array.shape[1] / grid_w

        ranked = list(
            zip(*np.unravel_index(np.argsort(saliency.ravel())[::-1], saliency.shape))
        )

        mask = np.zeros((grid_h, grid_w), dtype=bool)
        masked_image = image_array.copy()
        current_answer = baseline.answer
        current_confidence = baseline.confidence
        goal_reached = False

        for step, (row, col) in enumerate(ranked):
            if step >= self._max_cells:
                break

            r0 = int(row * cell_h)
            r1 = int((row + 1) * cell_h)
            c0 = int(col * cell_w)
            c1 = int((col + 1) * cell_w)
            r1 = max(r1, r0 + 1)
            c1 = max(c1, c0 + 1)

            masked_image = masked_image.copy()
            masked_image[r0:r1, c0:c1, :] = self._fill_value
            mask[row, col] = True

            out = backend.infer(masked_image, question)
            current_answer = out.answer
            current_confidence = out.confidence

            delta = baseline.confidence - current_confidence
            flipped_now = current_answer != baseline.answer

            if delta >= self._confidence_drop or flipped_now:
                goal_reached = True
                break

        delta_confidence = float(baseline.confidence - current_confidence)
        flipped = current_answer != baseline.answer
        n_masked = int(mask.sum())

        if not goal_reached:
            logger.warning(
                "counterfactual: goal not reached after %d cells masked "
                "(delta=%.4f, flipped=%s)",
                n_masked,
                delta_confidence,
                flipped,
            )

        return CounterfactualResult(
            counterfactual_mask=mask,
            original_answer=baseline.answer,
            original_confidence=float(baseline.confidence),
            counterfactual_answer=current_answer,
            counterfactual_confidence=float(current_confidence),
            delta_confidence=delta_confidence,
            n_cells_masked=n_masked,
            grid_h=grid_h,
            grid_w=grid_w,
            flipped=flipped,
            goal_reached=goal_reached,
        )
