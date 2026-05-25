"""Tests for miru.integrated_attention and the integrated method in POST /explain."""
from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_b64(width: int = 16, height: int = 16) -> str:
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFF_FFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


def _image(h: int = 16, w: int = 16) -> np.ndarray:
    return np.random.default_rng(7).random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit — IntegratedAttention
# ---------------------------------------------------------------------------


def test_explain_returns_float32_grid() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=4).explain(MockVLMBackend(), _image(), "q")
    assert result.integrated_grid.dtype == np.float32


def test_explain_values_in_unit_range() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=4).explain(MockVLMBackend(), _image(), "q")
    assert float(result.integrated_grid.min()) >= 0.0
    assert float(result.integrated_grid.max()) <= 1.0


def test_explain_grid_shape_matches_extractor_resolution() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    extractor = AttentionExtractor(resolution=8)
    result = IntegratedAttention(n_steps=3, extractor=extractor).explain(
        MockVLMBackend(), _image(), "q"
    )
    assert result.grid_h == 8
    assert result.grid_w == 8
    assert result.integrated_grid.shape == (8, 8)


def test_explain_n_steps_reported() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=5).explain(MockVLMBackend(), _image(), "q")
    assert result.n_steps == 5


def test_explain_black_baseline() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=3, baseline="black").explain(
        MockVLMBackend(), _image(), "q"
    )
    assert result.integrated_grid.shape == (16, 16)


def test_explain_mean_baseline() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=3, baseline="mean").explain(
        MockVLMBackend(), _image(), "q"
    )
    assert result.integrated_grid.shape == (16, 16)


def test_explain_deterministic_for_same_inputs() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = IntegratedAttention(n_steps=4).explain(MockVLMBackend(seed=0), img, "cat?")
    r2 = IntegratedAttention(n_steps=4).explain(MockVLMBackend(seed=0), img, "cat?")
    np.testing.assert_array_equal(r1.integrated_grid, r2.integrated_grid)


def test_explain_different_questions_can_differ() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = IntegratedAttention(n_steps=4).explain(MockVLMBackend(), img, "Where is the dog?")
    r2 = IntegratedAttention(n_steps=4).explain(MockVLMBackend(), img, "What is the color?")
    # Grids may differ because mock attention depends on question hash.
    assert r1.integrated_grid.shape == r2.integrated_grid.shape


def test_invalid_n_steps_raises() -> None:
    from miru.integrated_attention import IntegratedAttention

    with pytest.raises(ValueError, match="n_steps"):
        IntegratedAttention(n_steps=1)


def test_invalid_n_steps_too_large_raises() -> None:
    from miru.integrated_attention import IntegratedAttention

    with pytest.raises(ValueError, match="n_steps"):
        IntegratedAttention(n_steps=101)


def test_invalid_baseline_raises() -> None:
    from miru.integrated_attention import IntegratedAttention

    with pytest.raises(ValueError, match="baseline"):
        IntegratedAttention(baseline="random")


def test_explain_2_steps_minimum() -> None:
    from miru.integrated_attention import IntegratedAttention
    from miru.models.mock import MockVLMBackend

    result = IntegratedAttention(n_steps=2).explain(MockVLMBackend(), _image(), "q")
    assert result.n_steps == 2
    assert float(result.integrated_grid.max()) <= 1.0


# ---------------------------------------------------------------------------
# API integration — POST /explain with method=integrated
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_explain_integrated_returns_200(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "integrated",
        "question": "Where is the subject?",
        "n_steps": 4,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_integrated_response_shape(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "integrated",
        "question": "q",
        "n_steps": 4,
    }
    body = api_client.post("/explain", json=payload).json()
    assert body["method"] == "integrated"
    assert len(body["attention_grid"]) > 0
    flat = [v for row in body["attention_grid"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_explain_integrated_overlay_present(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "integrated",
        "question": "q",
        "n_steps": 3,
    }
    body = api_client.post("/explain", json=payload).json()
    assert isinstance(body["overlay_b64"], str) and len(body["overlay_b64"]) > 0


def test_explain_integrated_mean_baseline(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "integrated",
        "question": "q",
        "n_steps": 3,
        "integrated_baseline": "mean",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_integrated_appears_in_methods(api_client) -> None:
    resp = api_client.get("/methods")
    body = resp.json()
    names = [m["name"] for m in body["methods"]]
    assert "integrated" in names
    statuses = {m["name"]: m["status"] for m in body["methods"]}
    assert statuses["integrated"] == "implemented"


def test_explain_integrated_unknown_model_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope",
        "method": "integrated",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_integrated_bad_image_400(api_client) -> None:
    payload = {
        "image_b64": "!!!bad",
        "model_name": "mock",
        "method": "integrated",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_integrated_n_steps_out_of_range_422(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "integrated",
        "question": "q",
        "n_steps": 1,  # min is 2
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 422


def test_explain_integrated_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
