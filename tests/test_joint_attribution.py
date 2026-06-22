"""Tests for miru.joint_attribution and method=joint in POST /explain."""
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
    return np.random.default_rng(9).random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit — JointAttribution
# ---------------------------------------------------------------------------


def test_result_dtype_float32() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution().explain(MockVLMBackend(), _image(), "q")
    assert result.joint_grid.dtype == np.float32


def test_result_values_in_unit_range() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution().explain(MockVLMBackend(), _image(), "q")
    assert float(result.joint_grid.min()) >= 0.0
    assert float(result.joint_grid.max()) <= 1.0


def test_result_shape_default_resolution() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution().explain(MockVLMBackend(), _image(), "q")
    assert result.joint_grid.shape == (16, 16)
    assert result.grid_h == 16
    assert result.grid_w == 16


def test_result_shape_custom_resolution() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    extractor = AttentionExtractor(resolution=8)
    result = JointAttribution(extractor=extractor).explain(MockVLMBackend(), _image(), "q")
    assert result.joint_grid.shape == (8, 8)
    assert result.grid_h == 8
    assert result.grid_w == 8


def test_used_intra_true_for_mock() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution().explain(MockVLMBackend(), _image(), "q")
    assert result.used_intra is True


def test_intra_weight_reported() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution(intra_weight=0.3).explain(MockVLMBackend(), _image(), "q")
    assert result.intra_weight == pytest.approx(0.3)
    assert result.cross_weight == pytest.approx(0.7)


def test_intra_weight_zero_equals_cross_modal() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    backend = MockVLMBackend(seed=0)
    img = _image()
    result_joint = JointAttribution(intra_weight=0.0).explain(backend, img, "q")

    extractor = AttentionExtractor()
    out = MockVLMBackend(seed=0).infer(img, "q")
    cross_grid = extractor.extract(out.attention_weights)

    np.testing.assert_array_almost_equal(result_joint.joint_grid, cross_grid, decimal=5)


def test_deterministic_same_inputs() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = JointAttribution(intra_weight=0.4).explain(MockVLMBackend(seed=0), img, "cat?")
    r2 = JointAttribution(intra_weight=0.4).explain(MockVLMBackend(seed=0), img, "cat?")
    np.testing.assert_array_equal(r1.joint_grid, r2.joint_grid)


def test_different_intra_weights_differ() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = JointAttribution(intra_weight=0.0).explain(MockVLMBackend(seed=0), img, "q")
    r2 = JointAttribution(intra_weight=1.0).explain(MockVLMBackend(seed=0), img, "q")
    assert not np.array_equal(r1.joint_grid, r2.joint_grid)


def test_fallback_no_intra_weights() -> None:
    """Backend without intra_visual_weights degrades to cross-modal only."""
    from miru.joint_attribution import JointAttribution
    from miru.models.base import VLMOutput, VLMBackend

    class _NoIntraBackend(VLMBackend):
        @property
        def name(self) -> str:
            return "no_intra"

        def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
            w = np.ones((4, 4), dtype=np.float32)
            return VLMOutput(
                answer="ok", confidence=0.9, attention_weights=w, reasoning_steps=[]
            )

    result = JointAttribution().explain(_NoIntraBackend(), _image(), "q")
    assert result.used_intra is False
    assert result.intra_weight == 0.0
    assert result.cross_weight == 1.0


def test_invalid_intra_weight_raises() -> None:
    from miru.joint_attribution import JointAttribution

    with pytest.raises(ValueError, match="intra_weight"):
        JointAttribution(intra_weight=-0.1)


def test_invalid_intra_weight_too_large_raises() -> None:
    from miru.joint_attribution import JointAttribution

    with pytest.raises(ValueError, match="intra_weight"):
        JointAttribution(intra_weight=1.1)


def test_boundary_intra_weight_zero_valid() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution(intra_weight=0.0).explain(MockVLMBackend(), _image(), "q")
    assert result.joint_grid.shape == (16, 16)


def test_boundary_intra_weight_one_valid() -> None:
    from miru.joint_attribution import JointAttribution
    from miru.models.mock import MockVLMBackend

    result = JointAttribution(intra_weight=1.0).explain(MockVLMBackend(), _image(), "q")
    assert result.joint_grid.shape == (16, 16)


# ---------------------------------------------------------------------------
# Unit — VLMOutput intra_visual_weights field
# ---------------------------------------------------------------------------


def test_mock_backend_provides_intra_visual_weights() -> None:
    from miru.models.mock import MockVLMBackend

    out = MockVLMBackend().infer(_image(), "q")
    assert out.intra_visual_weights is not None
    assert out.intra_visual_weights.dtype == np.float32
    assert out.intra_visual_weights.shape == (16, 16)


def test_vlm_output_intra_defaults_to_none() -> None:
    from miru.models.base import VLMOutput

    out = VLMOutput(
        answer="x", confidence=0.5,
        attention_weights=np.ones((4, 4), dtype=np.float32),
        reasoning_steps=[],
    )
    assert out.intra_visual_weights is None


# ---------------------------------------------------------------------------
# API integration — POST /explain with method=joint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_explain_joint_returns_200(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "Where is the subject?",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_joint_response_shape(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "q",
    }
    body = api_client.post("/explain", json=payload).json()
    assert body["method"] == "joint"
    assert len(body["attention_grid"]) > 0
    flat = [v for row in body["attention_grid"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_explain_joint_overlay_present(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "q",
    }
    body = api_client.post("/explain", json=payload).json()
    assert isinstance(body["overlay_b64"], str) and len(body["overlay_b64"]) > 0


def test_explain_joint_custom_intra_weight(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "q",
        "intra_weight": 0.7,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_joint_intra_weight_zero(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "q",
        "intra_weight": 0.0,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text


def test_explain_joint_appears_in_methods(api_client) -> None:
    resp = api_client.get("/methods")
    body = resp.json()
    names = [m["name"] for m in body["methods"]]
    assert "joint" in names
    statuses = {m["name"]: m["status"] for m in body["methods"]}
    assert statuses["joint"] == "implemented"


def test_explain_joint_unknown_model_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope",
        "method": "joint",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_joint_bad_image_400(api_client) -> None:
    payload = {
        "image_b64": "!!!bad",
        "model_name": "mock",
        "method": "joint",
        "question": "q",
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 400


def test_explain_joint_intra_weight_out_of_range_422(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "joint",
        "question": "q",
        "intra_weight": 1.5,
    }
    resp = api_client.post("/explain", json=payload)
    assert resp.status_code == 422


def test_explain_joint_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
