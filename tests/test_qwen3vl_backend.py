"""Tests for Qwen3VLBackend (miru/models/qwen3vl.py).

Three tiers:
- Structural/interface tests — no model load, no env var.
- Pure-helper unit tests — the attention-reshaping math, fully offline.
- Real inference tests — gated behind MIRU_TEST_REAL_BACKENDS=1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from miru.models.base import VLMBackend
from miru.models.qwen3vl import (
    Qwen3VLBackend,
    _attention_row_to_grid,
    _select_middle_layer,
)

_REAL_BACKENDS = os.environ.get("MIRU_TEST_REAL_BACKENDS") == "1"
_SKIP_REAL = pytest.mark.skipif(
    not _REAL_BACKENDS,
    reason="requires MIRU_TEST_REAL_BACKENDS=1",
)


# ---------------------------------------------------------------------------
# Structural / interface tests (no model loading)
# ---------------------------------------------------------------------------


def test_qwen3vl_backend_name() -> None:
    assert Qwen3VLBackend().name == "qwen3vl"


def test_qwen3vl_backend_instantiates() -> None:
    assert Qwen3VLBackend() is not None


def test_qwen3vl_backend_is_vlmbackend_subclass() -> None:
    assert issubclass(Qwen3VLBackend, VLMBackend)
    assert isinstance(Qwen3VLBackend(), VLMBackend)


def test_qwen3vl_backend_lazy_loads() -> None:
    backend = Qwen3VLBackend()
    assert backend._model is None
    assert backend._processor is None


def test_qwen3vl_backend_registered_in_defaults() -> None:
    from miru.models import registry

    registry.register_defaults()
    assert "qwen3vl" in registry.available()


# ---------------------------------------------------------------------------
# Pure-helper unit tests (offline)
# ---------------------------------------------------------------------------


def test_select_middle_layer_is_mid_stack() -> None:
    # 36 layers, 0.6 fraction -> round(35*0.6)=21, in the empirical 14-24 band.
    assert _select_middle_layer(36) == 21


def test_select_middle_layer_clamped_single_layer() -> None:
    assert _select_middle_layer(1) == 0


def test_select_middle_layer_within_bounds() -> None:
    for n in range(1, 50):
        idx = _select_middle_layer(n)
        assert 0 <= idx <= n - 1


def test_select_middle_layer_rejects_zero() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        _select_middle_layer(0)


def test_select_middle_layer_fraction_extremes() -> None:
    assert _select_middle_layer(10, fraction=0.0) == 0
    assert _select_middle_layer(10, fraction=1.0) == 9


def test_attention_row_to_grid_square() -> None:
    grid = _attention_row_to_grid(np.arange(16, dtype=np.float32))
    assert grid.shape == (4, 4)
    assert grid.dtype == np.float32


def test_attention_row_to_grid_truncates_non_square() -> None:
    # 17 tokens -> largest square is 4x4 = 16; the 17th is dropped.
    grid = _attention_row_to_grid(np.arange(17, dtype=np.float32))
    assert grid.shape == (4, 4)


def test_attention_row_to_grid_preserves_values() -> None:
    grid = _attention_row_to_grid(np.array([1, 2, 3, 4], dtype=np.float32))
    assert grid.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_attention_row_to_grid_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _attention_row_to_grid(np.array([], dtype=np.float32))


def test_attention_row_to_grid_accepts_2d_input() -> None:
    grid = _attention_row_to_grid(np.ones((1, 9), dtype=np.float32))
    assert grid.shape == (3, 3)


# ---------------------------------------------------------------------------
# Real inference tests (gated)
# ---------------------------------------------------------------------------


@_SKIP_REAL
def test_qwen3vl_infer_returns_vlmoutput() -> None:
    from miru.models.base import VLMOutput

    backend = Qwen3VLBackend()
    image = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "What is in this image?")
    assert isinstance(result, VLMOutput)


@_SKIP_REAL
def test_qwen3vl_confidence_in_range() -> None:
    backend = Qwen3VLBackend()
    image = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "Is this an outdoor scene?")
    assert 0.0 <= result.confidence <= 1.0


@_SKIP_REAL
def test_qwen3vl_attention_shape_is_square() -> None:
    backend = Qwen3VLBackend()
    image = np.random.default_rng(2).random((224, 224, 3)).astype(np.float32)
    result = backend.infer(image, "What objects are present?")
    attn = result.attention_weights
    assert attn.ndim == 2
    assert attn.shape[0] == attn.shape[1]
    assert attn.dtype == np.float32


@_SKIP_REAL
def test_qwen3vl_reasoning_steps_nonempty() -> None:
    backend = Qwen3VLBackend()
    image = np.random.default_rng(3).random((64, 64, 3)).astype(np.float32)
    result = backend.infer(image, "Describe the scene.")
    assert len(result.reasoning_steps) >= 1
