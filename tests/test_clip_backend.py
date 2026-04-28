"""Tests for CLIPBackend (miru/models/clip.py).

Tests 1-4 verify structural/interface contracts and run without any model
downloads or MIRU_TEST_REAL_BACKENDS=1.

Tests 5-8 require a real CLIP model and are gated behind the env var
MIRU_TEST_REAL_BACKENDS=1.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from miru.models.base import VLMBackend
from miru.models.clip import CLIPBackend

_REAL_BACKENDS = os.environ.get("MIRU_TEST_REAL_BACKENDS") == "1"
_SKIP_REAL = pytest.mark.skipif(
    not _REAL_BACKENDS,
    reason="requires MIRU_TEST_REAL_BACKENDS=1",
)


# ---------------------------------------------------------------------------
# Structural / interface tests (no model loading)
# ---------------------------------------------------------------------------


def test_clip_backend_name() -> None:
    """.name must equal 'clip' without loading any weights."""
    backend = CLIPBackend()
    assert backend.name == "clip"


def test_clip_backend_instantiates() -> None:
    """CLIPBackend() must succeed without downloading or loading anything."""
    backend = CLIPBackend()
    assert backend is not None


def test_clip_backend_is_vlmbackend_subclass() -> None:
    """CLIPBackend must be a proper subclass of VLMBackend."""
    assert issubclass(CLIPBackend, VLMBackend)
    backend = CLIPBackend()
    assert isinstance(backend, VLMBackend)


def test_clip_backend_lazy_loads() -> None:
    """_model and _processor must be None before the first infer() call."""
    backend = CLIPBackend()
    assert backend._model is None
    assert backend._processor is None


# ---------------------------------------------------------------------------
# Real inference tests (gated)
# ---------------------------------------------------------------------------


@_SKIP_REAL
def test_clip_infer_returns_vlmoutput() -> None:
    """infer() must return a VLMOutput-compatible object."""
    from miru.models.base import VLMOutput

    backend = CLIPBackend()
    image = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "Is there a cat?")
    assert isinstance(result, VLMOutput)


@_SKIP_REAL
def test_clip_confidence_in_range() -> None:
    """confidence must be in [0, 1]."""
    backend = CLIPBackend()
    image = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "Is this an outdoor scene?")
    assert 0.0 <= result.confidence <= 1.0


@_SKIP_REAL
def test_clip_attention_shape_is_square() -> None:
    """attention_weights must be a 2-D square float32 array."""
    backend = CLIPBackend()
    image = np.random.default_rng(2).random((224, 224, 3)).astype(np.float32)
    result = backend.infer(image, "What objects are present?")
    attn = result.attention_weights
    assert attn.ndim == 2
    assert attn.shape[0] == attn.shape[1], "attention map must be square"
    assert attn.dtype == np.float32


@_SKIP_REAL
def test_clip_reasoning_steps_nonempty() -> None:
    """reasoning_steps must contain at least one entry."""
    backend = CLIPBackend()
    image = np.random.default_rng(3).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "Describe the scene.")
    assert len(result.reasoning_steps) >= 1
