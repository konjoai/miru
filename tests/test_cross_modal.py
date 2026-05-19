"""Tests for miru.cross_modal and the POST /trace endpoint."""
from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _png_b64(width: int = 16, height: int = 16) -> str:
    """Minimal valid PNG encoded as base64."""
    raw_rows = b"".join(
        b"\x00" + b"\xff\xff\xff" * width for _ in range(height)
    )
    def _chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFF_FFFF)
        return length + tag + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(raw_rows))
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(signature + ihdr + idat + iend).decode()


def _image_array(h: int = 8, w: int = 8) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.float32) * 0.5


# ---------------------------------------------------------------------------
# Unit tests — CrossModalTracer
# ---------------------------------------------------------------------------


def test_trace_returns_correct_word_count() -> None:
    """Matrix rows == number of whitespace tokens."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    tracer = CrossModalTracer()
    question = "Where is the cat"
    result = tracer.trace(MockVLMBackend(), _image_array(), question)
    assert result.words == ["Where", "is", "the", "cat"]
    assert result.matrix.shape[0] == 4


def test_trace_matrix_columns_match_grid() -> None:
    """Each row has grid_h * grid_w columns."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    tracer = CrossModalTracer()
    result = tracer.trace(MockVLMBackend(), _image_array(), "What is here")
    h, w = result.grid_h, result.grid_w
    assert result.matrix.shape == (3, h * w)


def test_trace_matrix_dtype_float32() -> None:
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "hello world")
    assert result.matrix.dtype == np.float32


def test_trace_matrix_values_in_unit_range() -> None:
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "one two three")
    assert float(result.matrix.min()) >= 0.0
    assert float(result.matrix.max()) <= 1.0


def test_trace_full_attention_shape_matches_grid() -> None:
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "a b")
    assert result.full_attention.shape == (result.grid_h, result.grid_w)


def test_trace_full_attention_dtype_and_range() -> None:
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "hello")
    assert result.full_attention.dtype == np.float32
    assert float(result.full_attention.min()) >= 0.0
    assert float(result.full_attention.max()) <= 1.0


def test_trace_empty_question_returns_empty_matrix() -> None:
    """Empty question yields zero words and an empty matrix."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "")
    assert result.words == []
    assert result.matrix.shape[0] == 0


def test_trace_single_word_question() -> None:
    """A one-word question ablates to empty string; tracer should not crash."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    result = CrossModalTracer().trace(MockVLMBackend(), _image_array(), "dog")
    assert result.words == ["dog"]
    assert result.matrix.shape == (1, result.grid_h * result.grid_w)


def test_trace_determinism() -> None:
    """Same (image, question, backend) always returns the same matrix."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    tracer = CrossModalTracer()
    img = _image_array()
    r1 = tracer.trace(MockVLMBackend(seed=42), img, "is there a bird")
    r2 = tracer.trace(MockVLMBackend(seed=42), img, "is there a bird")
    np.testing.assert_array_equal(r1.matrix, r2.matrix)


def test_trace_different_questions_differ() -> None:
    """Two different questions should not produce identical matrices."""
    from miru.cross_modal import CrossModalTracer
    from miru.models.mock import MockVLMBackend

    tracer = CrossModalTracer()
    img = _image_array()
    r1 = tracer.trace(MockVLMBackend(), img, "red apple on table")
    r2 = tracer.trace(MockVLMBackend(), img, "blue ocean waves crash")
    # At least one row must differ.
    assert not np.allclose(r1.matrix, r2.matrix, atol=1e-5)


# ---------------------------------------------------------------------------
# Unit tests — _normalise_row helper
# ---------------------------------------------------------------------------


def test_normalise_row_uniform_returns_zeros() -> None:
    from miru.cross_modal import _normalise_row

    row = np.ones(10, dtype=np.float32) * 0.5
    out = _normalise_row(row)
    np.testing.assert_array_equal(out, np.zeros(10, dtype=np.float32))


def test_normalise_row_range() -> None:
    from miru.cross_modal import _normalise_row

    row = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    out = _normalise_row(row)
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# API integration tests — POST /trace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_trace_endpoint_happy_path(api_client) -> None:
    """POST /trace with valid payload returns 200."""
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "Where is the cat sitting",
    }
    resp = api_client.post("/trace", json=payload)
    assert resp.status_code == 200, resp.text


def test_trace_endpoint_response_shape(api_client) -> None:
    """Response has expected keys with correct types."""
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "What color is it",
    }
    body = api_client.post("/trace", json=payload).json()
    assert body["words"] == ["What", "color", "is", "it"]
    assert len(body["matrix"]) == 4
    assert body["grid_h"] > 0 and body["grid_w"] > 0
    assert len(body["matrix"][0]) == body["grid_h"] * body["grid_w"]
    assert body["latency_ms"] >= 0.0


def test_trace_endpoint_matrix_values_in_unit_range(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "big red boat",
    }
    body = api_client.post("/trace", json=payload).json()
    flat = [v for row in body["matrix"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_trace_endpoint_full_attention_shape(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "hello world",
    }
    body = api_client.post("/trace", json=payload).json()
    assert len(body["full_attention"]) == body["grid_h"]
    assert len(body["full_attention"][0]) == body["grid_w"]


def test_trace_endpoint_full_attention_values_in_unit_range(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "a b c",
    }
    body = api_client.post("/trace", json=payload).json()
    flat = [v for row in body["full_attention"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_trace_endpoint_empty_question(api_client) -> None:
    """Empty question returns an empty matrix, not an error."""
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "",
    }
    resp = api_client.post("/trace", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["words"] == []
    assert body["matrix"] == []


def test_trace_endpoint_unknown_model_returns_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nonexistent_backend_xyz",
        "question": "test",
    }
    resp = api_client.post("/trace", json=payload)
    assert resp.status_code == 400
    assert "nonexistent_backend_xyz" in resp.json()["detail"]


def test_trace_endpoint_bad_image_returns_400(api_client) -> None:
    payload = {
        "image_b64": "not_valid_base64!!!",
        "model_name": "mock",
        "question": "anything",
    }
    resp = api_client.post("/trace", json=payload)
    assert resp.status_code == 400


def test_trace_endpoint_model_name_in_response(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "quick test",
    }
    body = api_client.post("/trace", json=payload).json()
    assert body["model_name"] == "mock"
    assert body["question"] == "quick test"


def test_trace_endpoint_health_not_regressed(api_client) -> None:
    """GET /health still returns 200 after /trace is wired in."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
