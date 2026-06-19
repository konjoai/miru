"""Synergistic-faithfulness probe — does an explanation capture genuine
cross-modal interaction, or just visual salience?

The deletion test (:mod:`miru.fidelity`) asks whether the salient pixels
matter to the model.  It cannot, on its own, tell *why* they matter:
a region can be salient because the image alone makes it pop (a bright
border, a watermark) or because it is the region the **question** makes
relevant.  For a vision-language model, only the second case is faithful
cross-modal reasoning — and recent work shows deletion/insertion AUC
systematically over-credits the visual-only case (Cross-Modal Synergy
benchmark, arXiv:2605.22168).

Method
------

We treat the two modalities as the two players of a Shapley *interaction*
and measure the modality-level interaction index — the discrete mixed
second difference of model confidence over the presence/absence of each
modality (Grabisch & Roubens 1999; Janizek et al. 2021).  With

* ``V`` = the salient visual region (present = original pixels, absent =
  top-``k_pct`` salient pixels neutralised with the mean colour), and
* ``Q`` = the question text (present = real question, absent = the
  neutral prompt),

we run four inferences and read the confidence of each::

    f_both    = f(V present,  Q present)   # the real call
    f_lang    = f(V absent,   Q present)   # salient pixels removed
    f_vision  = f(V present,  Q absent)    # question removed
    f_neither = f(V absent,   Q absent)

The interaction is

    interaction = (f_both - f_lang) - (f_vision - f_neither)
                =  f_both - f_lang - f_vision + f_neither

i.e. *the effect of the salient region when the question is present,
minus its effect when the question is absent*.  When the salient region
only matters because the question makes it matter, the first term is
large and the second is ~0 — high synergy, faithful cross-modal
behaviour.  When the region drives confidence regardless of the question
(visual-only salience — exactly what F_syn is designed to catch), the two
terms cancel and synergy ≈ 0.

The score is normalised against the baseline, mirroring the deletion
test::

    synergy_score = clamp((interaction) / max(eps, f_both), 0, 1)

A score below :data:`LOW_SYNERGY_THRESHOLD` flags the explanation as
visual-only — salient, perhaps, but not evidence of cross-modal
reasoning.

Konjo
-----
This is a real, citable evaluation method, not a confidence echo: every
call runs three *extra* ``backend.infer()`` calls (vs. one for the
deletion test) on genuinely ablated inputs.  The salient-region ablation
reuses :func:`miru.fidelity._mask_top_k` rather than re-implementing it.
The mock backend's confidence is image-independent by construction, so it
reports exactly zero synergy — an honest signal that its saliency is
question-hash-driven, not image-driven.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.fidelity import _mask_top_k
from miru.models.base import VLMBackend

DEFAULT_K_PCT = 0.10
LOW_SYNERGY_THRESHOLD = 0.3
NEUTRAL_PROMPT = ""
EPSILON = 1e-9


@dataclass(frozen=True)
class SynergyResult:
    """Outcome of one modality-level synergy test.

    Attributes:
        synergy_score: Interaction as a fraction of baseline confidence,
            clamped to ``[0, 1]``.  Near ``1.0`` means the salient region
            matters *because of* the question (faithful cross-modal
            reasoning); near ``0.0`` means it matters regardless of the
            question (visual-only salience).
        interaction: Raw mixed second difference in confidence units
            (may be negative before clamping).
        f_both: Confidence with both modalities present (the real call).
        f_language_only: Confidence with the salient region removed.
        f_vision_only: Confidence with the question removed.
        f_neither: Confidence with both removed.
        k_pct: Fraction of pixels treated as the salient region.
        low_synergy: True iff ``synergy_score < LOW_SYNERGY_THRESHOLD``.
    """

    synergy_score: float
    interaction: float
    f_both: float
    f_language_only: float
    f_vision_only: float
    f_neither: float
    k_pct: float
    low_synergy: bool


def synergy_test(
    backend: VLMBackend,
    image: np.ndarray,
    prompt: str,
    saliency_map: np.ndarray,
    *,
    k_pct: float = DEFAULT_K_PCT,
    neutral_prompt: str = NEUTRAL_PROMPT,
    baseline_confidence: float | None = None,
) -> SynergyResult:
    """Measure the vision×language interaction behind a saliency map.

    Args:
        backend: The same ``VLMBackend`` that produced the saliency map.
        image: float32 ``(H, W, 3)`` array in ``[0, 1]`` — the original.
        prompt: The question conditioning the backend (the ``Q present``
            arm).
        saliency_map: 2-D saliency map (any resolution; bilinearly
            upsampled to image space by the shared masking helper).
        k_pct: Fraction of pixels forming the salient region, in
            ``(0, 1)``.
        neutral_prompt: The ``Q absent`` ablation — defaults to the empty
            string (language signal removed).
        baseline_confidence: Pre-computed confidence for the ``f_both``
            arm, to skip a redundant call when the caller already ran
            ``backend.infer()`` on the unmasked image with ``prompt``.

    Returns:
        :class:`SynergyResult`.

    Raises:
        ValueError: If ``k_pct`` is outside ``(0, 1)`` or ``image`` is not
            an ``(H, W, 3)`` array.
    """
    if not 0.0 < k_pct < 1.0:
        raise ValueError(f"k_pct must be in (0, 1), got {k_pct!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")

    masked = _mask_top_k(image, saliency_map, k_pct)

    if baseline_confidence is None:
        baseline_confidence = float(backend.infer(image, prompt).confidence)
    f_both = baseline_confidence
    f_language_only = float(backend.infer(masked, prompt).confidence)
    f_vision_only = float(backend.infer(image, neutral_prompt).confidence)
    f_neither = float(backend.infer(masked, neutral_prompt).confidence)

    interaction = f_both - f_language_only - f_vision_only + f_neither
    score = float(max(0.0, min(1.0, interaction / max(EPSILON, f_both))))

    return SynergyResult(
        synergy_score=score,
        interaction=float(interaction),
        f_both=f_both,
        f_language_only=f_language_only,
        f_vision_only=f_vision_only,
        f_neither=f_neither,
        k_pct=float(k_pct),
        low_synergy=score < LOW_SYNERGY_THRESHOLD,
    )


__all__ = [
    "DEFAULT_K_PCT",
    "LOW_SYNERGY_THRESHOLD",
    "NEUTRAL_PROMPT",
    "SynergyResult",
    "synergy_test",
]
