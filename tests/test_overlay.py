"""Unit tests for miru/visualization/overlay.py — 8 tests."""
from __future__ import annotations

import base64

import numpy as np
import pytest

from miru.visualization.overlay import (
    attention_to_heatmap,
    decode_image_b64,
    encode_png_b64,
    generate_overlay,
    overlay_attention_on_image,
)
from miru.schemas import ReasoningTrace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ones_1x1_png_b64() -> str:
    """Base64-encoded PNG of a 1×1 white RGBA pixel (all 255)."""
    arr = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
    return encode_png_b64(arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_heatmap_all_zeros_is_blue() -> None:
    """All-zero attention → blue-ish pixels (jet colormap low-end is blue)."""
    attention = np.zeros((4, 4), dtype=np.float32)
    result = attention_to_heatmap(attention)

    assert result.shape == (4, 4, 4), f"Expected (4,4,4), got {result.shape}"
    # At t=0 the jet colormap gives R=0, G=0, B≈128 (0.5*255).
    # Blue channel should be clearly dominant over red.
    assert int(result[0, 0, 2]) > int(result[0, 0, 0]), (
        f"Expected blue > red for zero attention; got R={result[0,0,0]}, B={result[0,0,2]}"
    )


def test_heatmap_all_ones_is_red() -> None:
    """All-one attention → red-ish pixels (jet colormap high-end is red)."""
    attention = np.ones((4, 4), dtype=np.float32)
    result = attention_to_heatmap(attention)

    assert result.shape == (4, 4, 4), f"Expected (4,4,4), got {result.shape}"
    # At t=1 the jet colormap gives R=0.5, G=0, B=0 → R channel is dominant.
    assert int(result[0, 0, 0]) > int(result[0, 0, 2]), (
        f"Expected red > blue for max attention; got R={result[0,0,0]}, B={result[0,0,2]}"
    )


def test_heatmap_dtype_and_range() -> None:
    """Heatmap output must be uint8 with values in [0, 255]."""
    rng = np.random.default_rng(42)
    attention = rng.uniform(0.0, 1.0, (8, 8)).astype(np.float32)
    result = attention_to_heatmap(attention)

    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_overlay_output_shape_matches_image() -> None:
    """overlay_attention_on_image must return the same spatial shape as the input."""
    rng = np.random.default_rng(7)
    h, w = 32, 48
    image_rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    attention = rng.uniform(0.0, 1.0, (8, 6)).astype(np.float32)

    result = overlay_attention_on_image(image_rgba, attention, alpha=0.4)

    assert result.shape == (h, w, 4), f"Expected ({h},{w},4), got {result.shape}"
    assert result.dtype == np.uint8


def test_encode_png_b64_returns_valid_base64() -> None:
    """encode_png_b64 on a 4×4 RGBA array returns a non-empty valid base64 string."""
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # fully opaque
    result = encode_png_b64(arr)

    assert isinstance(result, str)
    assert len(result) > 0
    # Must be decodable base64.
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_encode_decode_round_trip() -> None:
    """decode_image_b64(encode_png_b64(arr)) round-trip preserves spatial shape."""
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (6, 6, 4), dtype=np.uint8)
    b64 = encode_png_b64(arr)
    recovered = decode_image_b64(b64)

    assert recovered.shape == arr.shape, (
        f"Expected {arr.shape}, got {recovered.shape}"
    )
    assert recovered.dtype == np.uint8


def test_generate_overlay_with_1x1_white_png(ones_1x1_png_b64: str) -> None:
    """generate_overlay with a 1×1 white PNG returns a non-empty base64 string."""
    attention = np.array([[0.5]], dtype=np.float32)
    result = generate_overlay(ones_1x1_png_b64, attention, alpha=0.5, colormap="jet")

    assert isinstance(result, str)
    assert len(result) > 0
    # Must be valid base64.
    raw = base64.b64decode(result)
    assert len(raw) > 0


def test_reasoning_trace_has_overlay_b64_field() -> None:
    """ReasoningTrace schema must have an overlay_b64 field defaulting to None."""
    from miru.schemas import AttentionMap, ReasoningStep

    trace = ReasoningTrace(
        answer="test",
        steps=[ReasoningStep(step=1, description="step one", confidence=0.9)],
        attention_map=AttentionMap(
            width=2,
            height=2,
            data=[[0.1, 0.2], [0.3, 0.4]],
        ),
        backend="mock",
        latency_ms=1.0,
    )
    assert hasattr(trace, "overlay_b64"), "ReasoningTrace must have overlay_b64 attribute"
    assert trace.overlay_b64 is None, f"overlay_b64 must default to None, got {trace.overlay_b64!r}"
