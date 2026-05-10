"""Tests for miru.lime_explainer and miru.gradcam_explainer.

These exercise both explainers against the deterministic mock backend so
they run in CI without any model download.
"""
from __future__ import annotations

import numpy as np
import pytest

from miru import gradcam_explainer, lime_explainer
from miru.models.mock import MockVLMBackend


@pytest.fixture
def backend() -> MockVLMBackend:
    return MockVLMBackend()


@pytest.fixture
def small_image() -> np.ndarray:
    """16x16 RGB float32 image with a bright spot — small for fast LIME runs."""
    img = np.full((16, 16, 3), 0.1, dtype=np.float32)
    img[4:8, 4:8] = 0.95
    return img


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def test_lime_segment_image_has_unique_ids(small_image: np.ndarray) -> None:
    seg = lime_explainer.segment_image(small_image, n_segments=16)
    assert seg.shape == small_image.shape[:2]
    # ids dense from 0..max
    unique = np.unique(seg)
    assert unique[0] == 0 and unique[-1] == len(unique) - 1


def test_lime_segment_image_rejects_non_rgb() -> None:
    bad = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        lime_explainer.segment_image(bad, n_segments=4)


def test_lime_explain_returns_normalized_saliency(
    backend: MockVLMBackend, small_image: np.ndarray
) -> None:
    result = lime_explainer.explain(
        backend,
        small_image,
        "where is the bright spot?",
        n_segments=9,
        n_samples=12,
        resolution=8,
        seed=1,
    )
    assert isinstance(result, lime_explainer.LimeExplanation)
    assert result.saliency.shape == (8, 8)
    assert result.saliency.dtype == np.float32
    assert result.saliency.min() >= 0.0
    assert result.saliency.max() <= 1.0
    assert result.n_samples == 12
    assert result.segment_weights.shape == (result.n_segments,)


def test_lime_explain_is_deterministic_under_same_seed(
    backend: MockVLMBackend, small_image: np.ndarray
) -> None:
    a = lime_explainer.explain(backend, small_image, "q", n_samples=8, n_segments=9, seed=7, resolution=8)
    b = lime_explainer.explain(backend, small_image, "q", n_samples=8, n_segments=9, seed=7, resolution=8)
    np.testing.assert_array_equal(a.saliency, b.saliency)


def test_lime_explain_rejects_too_few_samples(
    backend: MockVLMBackend, small_image: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        lime_explainer.explain(backend, small_image, "q", n_samples=1, n_segments=4)


# ---------------------------------------------------------------------------
# GradCAM (occlusion sensitivity)
# ---------------------------------------------------------------------------


def test_gradcam_explain_returns_normalized_saliency(
    backend: MockVLMBackend, small_image: np.ndarray
) -> None:
    result = gradcam_explainer.explain(
        backend, small_image, "q", occlusion_grid=4, resolution=8
    )
    assert isinstance(result, gradcam_explainer.GradCAMExplanation)
    assert result.saliency.shape == (8, 8)
    assert result.saliency.dtype == np.float32
    assert 0.0 <= result.saliency.min() <= result.saliency.max() <= 1.0
    assert result.raw_response.shape == (4, 4)
    assert result.n_calls == 17  # 4*4 + 1 baseline


def test_gradcam_explain_rejects_tiny_grid(
    backend: MockVLMBackend, small_image: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        gradcam_explainer.explain(backend, small_image, "q", occlusion_grid=1)


def test_gradcam_explain_rejects_non_rgb(backend: MockVLMBackend) -> None:
    with pytest.raises(ValueError):
        gradcam_explainer.explain(backend, np.zeros((8, 8), dtype=np.float32), "q")
