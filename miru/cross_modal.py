"""Cross-modal attention tracer — word → image-region attribution.

For each token in a question, the tracer estimates how much the model's
spatial attention shifts when that token is removed.  The result is a
``(n_words, grid_h * grid_w)`` matrix where each row is the normalised
attention attribution for one word.

Algorithm
---------
1. Tokenise the question on whitespace.
2. Run full inference → ``attn_full`` (H × W float32 in [0, 1]).
3. For each word ``w_i``, form ``q_ablated`` = question without ``w_i``,
   run inference → ``attn_ablated``.
4. Attribution for ``w_i`` = ``max(0, attn_full − attn_ablated)`` — the
   positive shift in attention caused by the word's presence.
5. Min-max normalise each word's row to [0, 1]; uniform rows → all-zero.

This is backend-agnostic (works with any :class:`~miru.models.base.VLMBackend`)
and requires no gradients — it is a pure perturbation approach.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

logger = logging.getLogger(__name__)

_EXTRACTOR = AttentionExtractor()


@dataclass(frozen=True)
class CrossModalTrace:
    """Word-to-image-region attribution matrix.

    Attributes:
        words: Whitespace-tokenised question tokens.
        matrix: float32 array of shape ``(len(words), grid_h * grid_w)``
            with values in ``[0, 1]``.  Row ``i`` is the normalised spatial
            attribution for ``words[i]``.
        grid_h: Height of the spatial grid (number of rows).
        grid_w: Width of the spatial grid (number of columns).
        full_attention: The baseline attention map for the full question,
            shape ``(grid_h, grid_w)`` float32 in ``[0, 1]``.
    """

    words: list[str]
    matrix: np.ndarray
    grid_h: int
    grid_w: int
    full_attention: np.ndarray


class CrossModalTracer:
    """Compute cross-modal word→image-region attribution via ablation.

    Args:
        extractor: :class:`~miru.attention.extractor.AttentionExtractor`
            used to normalise and grid-resize raw attention weights.  If
            *None*, a default extractor (resolution=16) is used.
    """

    def __init__(self, extractor: AttentionExtractor | None = None) -> None:
        self._extractor = extractor or _EXTRACTOR

    def trace(
        self,
        backend: VLMBackend,
        image_array: np.ndarray,
        question: str,
    ) -> CrossModalTrace:
        """Run cross-modal attribution for *question* against *image_array*.

        Args:
            backend: Any registered :class:`~miru.models.base.VLMBackend`.
            image_array: float32 ``(H, W, 3)`` image in ``[0, 1]``.
            question: Natural-language question; tokenised on whitespace.

        Returns:
            :class:`CrossModalTrace` with the word attribution matrix and
            the baseline full-question attention map.
        """
        words = question.split()
        if not words:
            grid = self._extractor.resolution
            empty = np.zeros((0, grid * grid), dtype=np.float32)
            full = np.zeros((grid, grid), dtype=np.float32)
            return CrossModalTrace(
                words=[], matrix=empty, grid_h=grid, grid_w=grid, full_attention=full
            )

        full_out = backend.infer(image_array, question)
        full_attention = self._extractor.extract(full_out.attention_weights)
        grid_h, grid_w = full_attention.shape

        rows: list[np.ndarray] = []
        for i, _word in enumerate(words):
            ablated_tokens = words[:i] + words[i + 1 :]
            ablated_question = " ".join(ablated_tokens) if ablated_tokens else ""
            if ablated_question:
                try:
                    abl_out = backend.infer(image_array, ablated_question)
                    abl_attention = self._extractor.extract(abl_out.attention_weights)
                except Exception:
                    logger.warning(
                        "cross_modal: ablation inference failed for word index %d", i
                    )
                    abl_attention = full_attention.copy()
            else:
                abl_attention = np.zeros_like(full_attention)

            raw_row = np.maximum(0.0, full_attention - abl_attention)
            rows.append(_normalise_row(raw_row.flatten()))

        matrix = np.stack(rows, axis=0).astype(np.float32)
        return CrossModalTrace(
            words=words,
            matrix=matrix,
            grid_h=grid_h,
            grid_w=grid_w,
            full_attention=full_attention,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_row(row: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D float array to [0, 1]; uniform → all-zero."""
    lo, hi = float(row.min()), float(row.max())
    if hi - lo < 1e-8:
        return np.zeros_like(row, dtype=np.float32)
    return ((row - lo) / (hi - lo)).astype(np.float32)
