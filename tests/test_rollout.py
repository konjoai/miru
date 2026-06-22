"""Tests for miru.rollout and method=rollout in POST /explain."""
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
    return np.random.default_rng(13).random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit — AttentionRollout
# ---------------------------------------------------------------------------


def test_rollout_grid_dtype_float32() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout().explain(MockVLMBackend(), _image(), "q")
    assert result.rollout_grid.dtype == np.float32


def test_rollout_values_in_unit_range() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout().explain(MockVLMBackend(), _image(), "q")
    assert float(result.rollout_grid.min()) >= 0.0
    assert float(result.rollout_grid.max()) <= 1.0


def test_rollout_default_shape() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout().explain(MockVLMBackend(), _image(), "q")
    assert result.rollout_grid.shape == (16, 16)
    assert result.grid_h == 16
    assert result.grid_w == 16


def test_rollout_custom_resolution() -> None:
    from miru.rollout import AttentionRollout
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    extractor = AttentionExtractor(resolution=8)
    result = AttentionRollout(extractor=extractor).explain(MockVLMBackend(), _image(), "q")
    assert result.rollout_grid.shape == (8, 8)
    assert result.grid_h == 8
    assert result.grid_w == 8


def test_rollout_uses_layer_weights_from_mock() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout().explain(MockVLMBackend(), _image(), "q")
    assert result.used_layer_weights is True
    assert result.n_layers == 4


def test_rollout_n_layers_reported() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout().explain(MockVLMBackend(), _image(), "q")
    assert result.n_layers == 4


def test_rollout_residual_weight_reported() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout(residual_weight=0.3).explain(MockVLMBackend(), _image(), "q")
    assert result.residual_weight == pytest.approx(0.3)


def test_rollout_deterministic() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = AttentionRollout().explain(MockVLMBackend(seed=0), img, "cat?")
    r2 = AttentionRollout().explain(MockVLMBackend(seed=0), img, "cat?")
    np.testing.assert_array_equal(r1.rollout_grid, r2.rollout_grid)


def test_rollout_residual_zero_differs_from_residual_one() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    img = _image()
    r0 = AttentionRollout(residual_weight=0.0).explain(MockVLMBackend(seed=0), img, "q")
    r1 = AttentionRollout(residual_weight=1.0).explain(MockVLMBackend(seed=0), img, "q")
    assert not np.array_equal(r0.rollout_grid, r1.rollout_grid)


def test_rollout_fallback_no_layer_weights() -> None:
    """Backend without layer_attention_weights falls back to single-layer."""
    from miru.rollout import AttentionRollout
    from miru.models.base import VLMOutput, VLMBackend

    class _NoLayerBackend(VLMBackend):
        @property
        def name(self) -> str:
            return "no_layer"

        def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
            w = np.ones((4, 4), dtype=np.float32)
            return VLMOutput(
                answer="ok", confidence=0.9, attention_weights=w, reasoning_steps=[]
            )

    result = AttentionRollout().explain(_NoLayerBackend(), _image(), "q")
    assert result.used_layer_weights is False
    assert result.n_layers == 1


def test_rollout_invalid_residual_weight_raises() -> None:
    from miru.rollout import AttentionRollout

    with pytest.raises(ValueError, match="residual_weight"):
        AttentionRollout(residual_weight=-0.1)


def test_rollout_invalid_residual_weight_too_large_raises() -> None:
    from miru.rollout import AttentionRollout

    with pytest.raises(ValueError, match="residual_weight"):
        AttentionRollout(residual_weight=1.1)


def test_rollout_residual_zero_boundary_valid() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout(residual_weight=0.0).explain(MockVLMBackend(), _image(), "q")
    assert result.rollout_grid.shape == (16, 16)


def test_rollout_residual_one_boundary_valid() -> None:
    from miru.rollout import AttentionRollout
    from miru.models.mock import MockVLMBackend

    result = AttentionRollout(residual_weight=1.0).explain(MockVLMBackend(), _image(), "q")
    assert result.rollout_grid.shape == (16, 16)


def test_mock_backend_provides_layer_attention_weights() -> None:
    from miru.models.mock import MockVLMBackend

    out = MockVLMBackend().infer(_image(), "q")
    assert out.layer_attention_weights is not None
    assert len(out.layer_attention_weights) == 4
    for layer in out.layer_attention_weights:
        assert layer.dtype == np.float32
        assert layer.shape == (16, 16)


def test_vlm_output_layer_weights_defaults_to_none() -> None:
    from miru.models.base import VLMOutput

    out = VLMOutput(
        answer="x", confidence=0.5,
        attention_weights=np.ones((4, 4), dtype=np.float32),
        reasoning_steps=[],
    )
    assert out.layer_attention_weights is None


# ---------------------------------------------------------------------------
# API integration — POST /explain with method=rollout
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_explain_rollout_returns_200(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "rollout",
        "question": "Where is the subject?",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_rollout_response_shape(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "rollout",
        "question": "q",
    }
    body = api_client.post("/explain", json=payload).json()
    assert body["method"] == "rollout"
    assert len(body["attention_grid"]) > 0
    flat = [v for row in body["attention_grid"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_explain_rollout_overlay_present(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "rollout",
        "question": "q",
    }
    body = api_client.post("/explain", json=payload).json()
    assert isinstance(body["overlay_b64"], str) and len(body["overlay_b64"]) > 0


def test_explain_rollout_custom_residual_weight(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "rollout",
        "question": "q",
        "residual_weight": 0.3,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_rollout_appears_in_methods(api_client) -> None:
    resp = api_client.get("/methods")
    body = resp.json()
    names = [m["name"] for m in body["methods"]]
    assert "rollout" in names
    statuses = {m["name"]: m["status"] for m in body["methods"]}
    assert statuses["rollout"] == "implemented"


def test_explain_rollout_unknown_model_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope",
        "method": "rollout",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_rollout_bad_image_400(api_client) -> None:
    payload = {
        "image_b64": "!!!bad",
        "model_name": "mock",
        "method": "rollout",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_rollout_residual_weight_out_of_range_422(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "rollout",
        "question": "q",
        "residual_weight": 1.5,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 422


def test_explain_rollout_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
