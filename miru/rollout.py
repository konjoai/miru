"""Attention rollout for multi-layer Transformer VLMs.

Computes a multi-layer saliency map by propagating attention through all
transformer layers using the geometric-mean ("attention flow") aggregation
from Abnar & Zuidema (2020).

Algorithm
---------
Given *L* per-layer attention maps ``A_1, …, A_L`` (each ``(H, W)`` float32,
non-negative, from the first to the last encoder layer):

1. Min-max normalise each map: ``a_l = normalise(A_l)``.
2. Add a residual identity term (models the skip-connection):
   ``â_l = (1 − residual_weight) * a_l + residual_weight * uniform``,
   where ``uniform = ones(H,W)/(H*W)`` and ``residual_weight`` defaults
   to ``0.5``.
3. Compute the geometric mean across layers in log-space:
   ``rollout = exp(mean_l [log(â_l + ε)])``
4. Min-max normalise the result to ``[0, 1]``.

The geometric mean is preferred over the arithmetic mean because it
preserves sparsity: a cell must receive *consistent* attention across
all layers to score high, matching the rollout intuition that information
must flow through every layer.

When ``layer_attention_weights`` is ``None`` (backend does not expose
per-layer weights), the explainer falls back to the single final-layer
attention map with a warning.

References
----------
Abnar, S., & Zuidema, W. (2020). *Quantifying Attention Flow in
Transformers*.  ACL 2020.  arXiv:2005.00928.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_WEIGHT: float = 0.5
_LOG_EPS: float = 1e-8


@dataclass(frozen=True)
class RolloutResult:
    """Output of the attention rollout explainer.

    Attributes:
        rollout_grid: Multi-layer saliency map, float32
            ``(grid_h, grid_w)`` in ``[0, 1]``.
        n_layers: Number of layers actually used.
        used_layer_weights: Whether per-layer weights were available.
        residual_weight: Residual identity weight applied.
        grid_h: Height of the output grid.
        grid_w: Width of the output grid.
    """

    rollout_grid: np.ndarray
    n_layers: int
    used_layer_weights: bool
    residual_weight: float
    grid_h: int
    grid_w: int


class AttentionRollout:
    """Multi-layer attention rollout explainer.

    Args:
        residual_weight: Weight for the uniform identity term ``∈ [0, 1]``.
            Represents the Transformer skip-connection contribution.
            Defaults to ``0.5``.
        extractor: :class:`~miru.attention.extractor.AttentionExtractor`
            for normalisation and grid resizing.
    """

    def __init__(
        self,
        residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
        extractor: AttentionExtractor | None = None,
    ) -> None:
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                f"residual_weight must be in [0, 1], got {residual_weight}"
            )
        self._residual_weight = residual_weight
        self._extractor = extractor or AttentionExtractor()

    def explain(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> RolloutResult:
        """Compute attention rollout for *image_array*.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question.

        Returns:
            :class:`RolloutResult` with the rolled-out saliency grid.
        """
        out = backend.infer(image_array, question)

        if out.layer_attention_weights is not None and len(out.layer_attention_weights) > 0:
            raw_layers = out.layer_attention_weights
            used_layer_weights = True
        else:
            logger.warning(
                "rollout: backend '%s' did not supply layer_attention_weights;"
                " falling back to single-layer attention.",
                backend.name,
            )
            raw_layers = [out.attention_weights]
            used_layer_weights = False

        resolution = self._extractor.resolution
        n_cells = resolution * resolution
        uniform = np.full((resolution, resolution), 1.0 / n_cells, dtype=np.float32)

        log_sum = np.zeros((resolution, resolution), dtype=np.float64)
        for raw in raw_layers:
            norm = self._extractor.extract(raw)
            blended = (
                (1.0 - self._residual_weight) * norm
                + self._residual_weight * uniform
            ).astype(np.float64)
            log_sum += np.log(blended + _LOG_EPS)

        geom_mean = np.exp(log_sum / len(raw_layers)).astype(np.float32)

        lo, hi = float(geom_mean.min()), float(geom_mean.max())
        if hi - lo > 1e-8:
            geom_mean = ((geom_mean - lo) / (hi - lo)).astype(np.float32)

        h, w = geom_mean.shape
        return RolloutResult(
            rollout_grid=geom_mean,
            n_layers=len(raw_layers),
            used_layer_weights=used_layer_weights,
            residual_weight=self._residual_weight,
            grid_h=h,
            grid_w=w,
        )
