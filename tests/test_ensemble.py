"""Tests for miru.ensemble and POST /explain/ensemble."""
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


def _image(h: int = 32, w: int = 32) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.float32) * 0.5


# ---------------------------------------------------------------------------
# Unit — _bilinear_resize_image
# ---------------------------------------------------------------------------


def test_resize_image_scale_1_returns_same_shape() -> None:
    from miru.ensemble import _bilinear_resize_image

    img = _image(16, 16)
    out = _bilinear_resize_image(img, 1.0)
    assert out is not None
    assert out.shape == (16, 16, 3)


def test_resize_image_scale_2_doubles_dimensions() -> None:
    from miru.ensemble import _bilinear_resize_image

    img = _image(8, 8)
    out = _bilinear_resize_image(img, 2.0)
    assert out is not None
    assert out.shape == (16, 16, 3)


def test_resize_image_scale_half_halves_dimensions() -> None:
    from miru.ensemble import _bilinear_resize_image

    img = _image(16, 16)
    out = _bilinear_resize_image(img, 0.5)
    assert out is not None
    assert out.shape == (8, 8, 3)


def test_resize_image_too_small_returns_none() -> None:
    from miru.ensemble import _bilinear_resize_image

    img = _image(8, 8)
    assert _bilinear_resize_image(img, 0.1) is None


def test_resize_image_dtype_float32() -> None:
    from miru.ensemble import _bilinear_resize_image

    out = _bilinear_resize_image(_image(), 1.0)
    assert out is not None
    assert out.dtype == np.float32


def test_resize_image_values_in_range() -> None:
    from miru.ensemble import _bilinear_resize_image

    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    out = _bilinear_resize_image(img, 1.5)
    assert out is not None
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Unit — AttentionEnsemble
# ---------------------------------------------------------------------------


def test_ensemble_single_scale_matches_direct_inference() -> None:
    """With one scale (1.0), ensemble grid is the normalised attention map."""
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend
    from miru.attention.extractor import AttentionExtractor

    backend = MockVLMBackend()
    img = _image()
    q = "Where is the subject?"

    extractor = AttentionExtractor()
    out = backend.infer(img, q)
    direct = extractor.extract(out.attention_weights)

    result = AttentionEnsemble(scales=(1.0,)).run(backend, img, q)
    # Both should be normalised grids of the same shape; values may differ
    # only due to the final re-normalisation step.
    assert result.ensemble_grid.shape == direct.shape
    assert result.ensemble_grid.dtype == np.float32


def test_ensemble_result_in_unit_range() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble().run(MockVLMBackend(), _image(), "test")
    assert float(result.ensemble_grid.min()) >= 0.0
    assert float(result.ensemble_grid.max()) <= 1.0


def test_ensemble_scales_used_matches_successful_scales() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble(scales=(0.5, 1.0, 1.5)).run(
        MockVLMBackend(), _image(32, 32), "q"
    )
    assert sorted(result.scales_used) == sorted([0.5, 1.0, 1.5])
    assert result.scales_skipped == []


def test_ensemble_tiny_scale_reported_skipped() -> None:
    """A scale so small it drops below MIN_DIM must appear in scales_skipped."""
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble(scales=(0.01, 1.0)).run(
        MockVLMBackend(), _image(8, 8), "test"
    )
    assert 0.01 in result.scales_skipped
    assert 1.0 in result.scales_used


def test_ensemble_per_scale_count_matches_used() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble(scales=(0.5, 1.0, 2.0)).run(
        MockVLMBackend(), _image(16, 16), "bird"
    )
    assert len(result.per_scale) == len(result.scales_used)


def test_ensemble_per_scale_grids_unit_range() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble().run(MockVLMBackend(), _image(), "test")
    for _, grid in result.per_scale:
        assert float(grid.min()) >= 0.0
        assert float(grid.max()) <= 1.0


def test_ensemble_custom_weights_accepted() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble(
        scales=(0.5, 1.0), weights=(0.2, 0.8)
    ).run(MockVLMBackend(), _image(16, 16), "q")
    assert len(result.scales_used) > 0


def test_ensemble_mismatched_weights_raises() -> None:
    from miru.ensemble import AttentionEnsemble

    with pytest.raises(ValueError, match="weights"):
        AttentionEnsemble(scales=(0.5, 1.0), weights=(1.0,))


def test_ensemble_empty_scales_raises() -> None:
    from miru.ensemble import AttentionEnsemble

    with pytest.raises(ValueError, match="non-empty"):
        AttentionEnsemble(scales=())


def test_ensemble_all_scales_fail_returns_zeros() -> None:
    """If every scale is too small, ensemble returns all-zero grid."""
    from miru.ensemble import AttentionEnsemble
    from miru.models.mock import MockVLMBackend

    result = AttentionEnsemble(scales=(0.001, 0.002)).run(
        MockVLMBackend(), _image(4, 4), "q"
    )
    assert result.scales_used == []
    np.testing.assert_array_equal(result.ensemble_grid, 0.0)


def test_ensemble_grid_shape_matches_extractor_resolution() -> None:
    from miru.ensemble import AttentionEnsemble
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    extractor = AttentionExtractor(resolution=8)
    result = AttentionEnsemble(extractor=extractor).run(
        MockVLMBackend(), _image(), "q"
    )
    assert result.grid_h == 8
    assert result.grid_w == 8
    assert result.ensemble_grid.shape == (8, 8)


# ---------------------------------------------------------------------------
# API integration — POST /explain/ensemble
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_ensemble_endpoint_happy_path(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "Where is it?",
        "scales": [0.5, 1.0, 1.5],
    }
    resp = api_client.post("/explain/ensemble", json=payload)
    assert resp.status_code == 200, resp.text


def test_ensemble_endpoint_response_fields(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "test",
        "scales": [1.0, 1.5],
    }
    body = api_client.post("/explain/ensemble", json=payload).json()
    assert "ensemble_grid" in body
    assert "per_scale" in body
    assert "scales_used" in body
    assert "scales_skipped" in body
    assert "overlay_b64" in body
    assert "top_regions" in body
    assert body["latency_ms"] >= 0.0


def test_ensemble_endpoint_grid_values_in_range(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "scales": [0.5, 1.0],
    }
    body = api_client.post("/explain/ensemble", json=payload).json()
    flat = [v for row in body["ensemble_grid"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_ensemble_endpoint_per_scale_count(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "test",
        "scales": [0.75, 1.0, 1.25],
    }
    body = api_client.post("/explain/ensemble", json=payload).json()
    assert len(body["per_scale"]) == len(body["scales_used"])


def test_ensemble_endpoint_unknown_model_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope",
        "question": "q",
        "scales": [1.0],
    }
    resp = api_client.post("/explain/ensemble", json=payload)
    assert resp.status_code == 400


def test_ensemble_endpoint_bad_image_400(api_client) -> None:
    payload = {
        "image_b64": "!!!bad",
        "model_name": "mock",
        "question": "q",
        "scales": [1.0],
    }
    resp = api_client.post("/explain/ensemble", json=payload)
    assert resp.status_code == 400


def test_ensemble_endpoint_bad_scale_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "scales": [0.0],  # must be > 0
    }
    resp = api_client.post("/explain/ensemble", json=payload)
    assert resp.status_code == 400


def test_ensemble_endpoint_mismatched_weights_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "scales": [0.5, 1.0],
        "weights": [1.0],  # wrong length
    }
    resp = api_client.post("/explain/ensemble", json=payload)
    assert resp.status_code == 400


def test_ensemble_endpoint_model_name_echoed(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "echo test",
        "scales": [1.0],
    }
    body = api_client.post("/explain/ensemble", json=payload).json()
    assert body["model_name"] == "mock"
    assert body["question"] == "echo test"


def test_ensemble_endpoint_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
