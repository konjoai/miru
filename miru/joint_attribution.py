"""Joint intra-modal + cross-modal attention attribution.

Combines two complementary attention signals that a ViT-based VLM exposes:

- **Cross-modal attention** (``VLMOutput.attention_weights``): language tokens
  attending to visual patches — the signal already used by the single-step
  attention explainer.
- **Intra-visual attention** (``VLMOutput.intra_visual_weights``): visual
  patches attending to each other — captures local texture / spatial grouping
  information that cross-modal attention under-weights.

The joint saliency map is the convex combination

    joint = α · intra_visual + (1 − α) · cross_modal

then min-max normalised to ``[0, 1]``.  When ``intra_visual_weights`` is
absent the method degrades gracefully to pure cross-modal attention.

References
----------
arXiv:2509.22415 — cross-attention captures 52–75 % of VLM saliency; the
remaining variance lives in the intra-visual pathway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_INTRA_WEIGHT: float = 0.4
MIN_INTRA_WEIGHT: float = 0.0
MAX_INTRA_WEIGHT: float = 1.0


@dataclass(frozen=True)
class JointAttributionResult:
    """Output of the joint attribution explainer.

    Attributes:
        joint_grid: Blended saliency map, float32 ``(grid_h, grid_w)`` in
            ``[0, 1]``.
        intra_weight: Actual weight applied to the intra-visual signal.
        cross_weight: Actual weight applied to the cross-modal signal
            (``1 − intra_weight``).
        used_intra: Whether intra-visual weights were available and used.
        grid_h: Height of the output grid.
        grid_w: Width of the output grid.
    """

    joint_grid: np.ndarray
    intra_weight: float
    cross_weight: float
    used_intra: bool
    grid_h: int
    grid_w: int


class JointAttribution:
    """Joint intra-modal + cross-modal attribution explainer.

    Args:
        intra_weight: Weight for the intra-visual signal ``α ∈ [0, 1]``.
            The cross-modal signal receives weight ``1 − α``.
            Defaults to ``0.4`` (60 % cross-modal, 40 % intra-visual).
        extractor: :class:`~miru.attention.extractor.AttentionExtractor`
            for normalisation and grid resizing.
    """

    def __init__(
        self,
        intra_weight: float = DEFAULT_INTRA_WEIGHT,
        extractor: AttentionExtractor | None = None,
    ) -> None:
        if not MIN_INTRA_WEIGHT <= intra_weight <= MAX_INTRA_WEIGHT:
            raise ValueError(
                f"intra_weight must be in [{MIN_INTRA_WEIGHT}, {MAX_INTRA_WEIGHT}],"
                f" got {intra_weight}"
            )
        self._intra_weight = intra_weight
        self._extractor = extractor or AttentionExtractor()

    def explain(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> JointAttributionResult:
        """Compute joint attribution for *image_array*.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question.

        Returns:
            :class:`JointAttributionResult` with the blended saliency grid.
        """
        out = backend.infer(image_array, question)
        cross_grid = self._extractor.extract(out.attention_weights)

        used_intra = out.intra_visual_weights is not None
        if not used_intra:
            logger.warning(
                "joint_attribution: backend '%s' did not supply "
                "intra_visual_weights; falling back to cross-modal only.",
                backend.name,
            )
            joint = cross_grid
            actual_intra_w = 0.0
        else:
            intra_grid = self._extractor.extract(out.intra_visual_weights)  # type: ignore[arg-type]
            joint = (
                self._intra_weight * intra_grid
                + (1.0 - self._intra_weight) * cross_grid
            ).astype(np.float32)
            actual_intra_w = self._intra_weight

        lo, hi = float(joint.min()), float(joint.max())
        if hi - lo > 1e-8:
            joint = ((joint - lo) / (hi - lo)).astype(np.float32)

        h, w = joint.shape
        return JointAttributionResult(
            joint_grid=joint,
            intra_weight=actual_intra_w,
            cross_weight=1.0 - actual_intra_w,
            used_intra=used_intra,
            grid_h=h,
            grid_w=w,
        )
