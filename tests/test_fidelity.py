"""Tests for the explanation-fidelity scorecard."""
from __future__ import annotations

import numpy as np
import pytest

from miru.fidelity import (
    LOW_FIDELITY_THRESHOLD,
    FidelityResult,
    deletion_test,
)
from miru.models.base import VLMBackend, VLMOutput


class _TrackingMockBackend(VLMBackend):
    """Backend whose confidence drops linearly with image mean.

    Models the deletion test we care about: when salient pixels are
    replaced by the per-image mean, the *visual* mean shifts toward
    that fill colour and the backend reports lower confidence.
    """

    def __init__(self, baseline_image: np.ndarray) -> None:
        self._baseline_mean = float(baseline_image.mean())

    @property
    def name(self) -> str:
        return "tracking_mock"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:  # noqa: ARG002
        # Confidence high when the image is bright, low when the image is dark.
        m = float(image_array.mean())
        confidence = max(0.0, min(1.0, m))
        attn = np.full((4, 4), 0.5, dtype=np.float32)
        return VLMOutput(
            answer="ok", confidence=confidence,
            attention_weights=attn, reasoning_steps=[],
        )


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def test_fidelity_result_low_flag_threshold() -> None:
    """The low_fidelity flag fires below LOW_FIDELITY_THRESHOLD."""
    r = FidelityResult(
        fidelity_score=0.3,
        baseline_confidence=0.8,
        masked_confidence=0.56,
        k_pct=0.1,
        low_fidelity=True,
    )
    assert r.low_fidelity is True
    assert r.fidelity_score < LOW_FIDELITY_THRESHOLD


# ---------------------------------------------------------------------------
# Deletion test — input validation
# ---------------------------------------------------------------------------


def test_deletion_test_rejects_k_pct_out_of_range() -> None:
    img = np.full((4, 4, 3), 0.8, dtype=np.float32)
    sal = np.ones((2, 2), dtype=np.float32)
    backend = _TrackingMockBackend(img)
    with pytest.raises(ValueError):
        deletion_test(backend, img, "q", sal, k_pct=0.0)
    with pytest.raises(ValueError):
        deletion_test(backend, img, "q", sal, k_pct=1.0)


def test_deletion_test_rejects_bad_image_shape() -> None:
    backend = _TrackingMockBackend(np.zeros((4, 4, 3), dtype=np.float32))
    bad = np.zeros((4, 4), dtype=np.float32)  # 2-D, not (H, W, 3)
    sal = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        deletion_test(backend, bad, "q", sal)


# ---------------------------------------------------------------------------
# Deletion test — semantics
# ---------------------------------------------------------------------------


def test_focused_saliency_high_fidelity() -> None:
    """A saliency map that targets the BRIGHT half drops confidence ⇒ high fidelity.

    Image: top half = 1.0 (bright), bottom half = 0.0 (dark).  Mean is
    ~0.5, baseline confidence ≈ 0.5.  A saliency map that picks the
    bright half means deletion replaces 1.0 → 0.5 there, dragging the
    overall mean DOWN ⇒ masked confidence < baseline ⇒ high fidelity.
    """
    h = w = 16
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 2, :, :] = 1.0  # bright top half

    sal = np.zeros((h, w), dtype=np.float32)
    sal[: h // 2, :] = 1.0  # saliency points at bright half

    backend = _TrackingMockBackend(img)
    r = deletion_test(backend, img, "q", sal, k_pct=0.4)
    assert r.baseline_confidence > r.masked_confidence
    assert r.fidelity_score > 0.0
    assert r.k_pct == pytest.approx(0.4)


def test_uniform_saliency_low_fidelity() -> None:
    """A flat saliency map (no localisation) shouldn't drop confidence much."""
    h = w = 16
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    sal = np.full((h, w), 0.5, dtype=np.float32)
    backend = _TrackingMockBackend(img)
    r = deletion_test(backend, img, "q", sal, k_pct=0.2)
    # masking the "top" of a flat map removes ~no signal; confidence stable.
    assert abs(r.baseline_confidence - r.masked_confidence) < 1e-6
    assert r.fidelity_score == 0.0
    assert r.low_fidelity is True


def test_baseline_confidence_can_be_passed_in() -> None:
    """Caller-supplied baseline skips the first inference call."""
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    sal = np.ones((2, 2), dtype=np.float32)
    backend = _TrackingMockBackend(img)
    r = deletion_test(backend, img, "q", sal, k_pct=0.5, baseline_confidence=0.9)
    assert r.baseline_confidence == 0.9


def test_fidelity_score_clamped_to_unit_interval() -> None:
    """A masked_confidence > baseline cannot push the score negative."""
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    sal = np.zeros((2, 2), dtype=np.float32)
    sal[0, 0] = 1.0
    backend = _TrackingMockBackend(img)
    r = deletion_test(
        backend, img, "q", sal, k_pct=0.1, baseline_confidence=0.4
    )
    assert 0.0 <= r.fidelity_score <= 1.0
