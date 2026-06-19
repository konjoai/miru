"""Unit tests for the modality-level synergy probe (:mod:`miru.synergy`).

The mock backend's confidence is image-independent, so it reports exactly
zero synergy.  To exercise the positive-synergy path we use two tiny
synthetic backends whose confidence is a closed-form function of the
image content and the prompt:

* :class:`_SynergyBackend` adds a vision×language cross term, so removing
  the salient region matters *more* when the question is present —
  genuine cross-modal synergy.
* :class:`_VisualOnlyBackend` makes confidence depend on the image alone,
  so the synergy test must report ~0 (and flag ``low_synergy``) even
  though the saliency map is perfectly targeted.
"""

from __future__ import annotations

import numpy as np
import pytest

from miru.models.base import VLMBackend, VLMOutput
from miru.models.mock import MockVLMBackend
from miru.synergy import (
    DEFAULT_K_PCT,
    LOW_SYNERGY_THRESHOLD,
    SynergyResult,
    synergy_test,
)


def _vision_score(image: np.ndarray) -> float:
    """Mean brightness of the top-left quadrant — high while the bright
    salient patch is intact, lower once it is masked to the mean colour."""
    h, w = image.shape[:2]
    return float(image[: h // 2, : w // 2].mean())


class _SynergyBackend(VLMBackend):
    """Confidence carries a non-separable vision×language interaction."""

    @property
    def name(self) -> str:
        return "synergy-fake"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
        lang = 1.0 if question else 0.0
        vision = _vision_score(image_array)
        # Dominant non-separable cross term → strong, threshold-clearing synergy.
        confidence = 0.05 + 0.05 * lang + 0.9 * vision * lang
        return VLMOutput(
            answer="x",
            confidence=float(confidence),
            attention_weights=np.zeros((4, 4), dtype=np.float32),
            reasoning_steps=[],
        )


class _VisualOnlyBackend(VLMBackend):
    """Confidence depends on the image alone — no cross-modal term."""

    @property
    def name(self) -> str:
        return "visual-only-fake"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:  # noqa: ARG002
        confidence = 0.1 + 0.4 * _vision_score(image_array)
        return VLMOutput(
            answer="x",
            confidence=float(confidence),
            attention_weights=np.zeros((4, 4), dtype=np.float32),
            reasoning_steps=[],
        )


@pytest.fixture
def bright_image() -> np.ndarray:
    """32×32 image with a bright patch in the top-left salient region."""
    img = np.full((32, 32, 3), 0.1, dtype=np.float32)
    img[2:10, 2:10] = 0.9
    return img


@pytest.fixture
def saliency() -> np.ndarray:
    """Saliency map peaked on the top-left patch."""
    grid = np.zeros((16, 16), dtype=np.float32)
    grid[1:5, 1:5] = 1.0
    return grid


def test_returns_synergy_result(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "what is shown?", saliency)
    assert isinstance(res, SynergyResult)


def test_score_in_unit_range(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "what is shown?", saliency)
    assert 0.0 <= res.synergy_score <= 1.0


def test_cross_modal_backend_has_positive_synergy(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "what is shown?", saliency)
    assert res.synergy_score > 0.0
    assert res.interaction > 0.0
    assert not res.low_synergy


def test_visual_only_backend_has_zero_synergy(bright_image, saliency) -> None:
    res = synergy_test(_VisualOnlyBackend(), bright_image, "what is shown?", saliency)
    assert res.synergy_score == pytest.approx(0.0, abs=1e-6)
    assert res.interaction == pytest.approx(0.0, abs=1e-6)
    assert res.low_synergy


def test_mock_backend_reports_zero_synergy(bright_image, saliency) -> None:
    """The mock's confidence is image-independent — honest zero synergy."""
    res = synergy_test(MockVLMBackend(), bright_image, "what is shown?", saliency)
    assert res.synergy_score == pytest.approx(0.0, abs=1e-9)
    assert res.low_synergy


def test_interaction_is_mixed_second_difference(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "describe", saliency)
    expected = res.f_both - res.f_language_only - res.f_vision_only + res.f_neither
    assert res.interaction == pytest.approx(expected, abs=1e-9)


def test_deterministic(bright_image, saliency) -> None:
    a = synergy_test(_SynergyBackend(), bright_image, "q?", saliency)
    b = synergy_test(_SynergyBackend(), bright_image, "q?", saliency)
    assert a == b


def test_baseline_confidence_reused(bright_image, saliency) -> None:
    """Passing baseline_confidence overrides the f_both arm verbatim."""
    res = synergy_test(
        _SynergyBackend(),
        bright_image,
        "q?",
        saliency,
        baseline_confidence=0.5,
    )
    assert res.f_both == pytest.approx(0.5)


def test_default_k_pct_echoed(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "q?", saliency)
    assert res.k_pct == pytest.approx(DEFAULT_K_PCT)


def test_custom_neutral_prompt(bright_image, saliency) -> None:
    """A non-empty neutral prompt is still a valid language-absent arm for
    the cross-modal backend (any prompt → lang=1), collapsing synergy."""
    res = synergy_test(
        _SynergyBackend(),
        bright_image,
        "q?",
        saliency,
        neutral_prompt="a generic caption",
    )
    # With both arms language-present, the interaction cancels to ~0.
    assert res.synergy_score == pytest.approx(0.0, abs=1e-6)


def test_invalid_k_pct_raises(bright_image, saliency) -> None:
    with pytest.raises(ValueError, match="k_pct"):
        synergy_test(_SynergyBackend(), bright_image, "q?", saliency, k_pct=0.0)


def test_invalid_image_shape_raises(saliency) -> None:
    bad = np.zeros((8, 8), dtype=np.float32)  # missing channel axis
    with pytest.raises(ValueError, match="image"):
        synergy_test(_SynergyBackend(), bad, "q?", saliency)


def test_low_synergy_flag_matches_threshold(bright_image, saliency) -> None:
    res = synergy_test(_SynergyBackend(), bright_image, "what is shown?", saliency)
    assert res.low_synergy == (res.synergy_score < LOW_SYNERGY_THRESHOLD)
