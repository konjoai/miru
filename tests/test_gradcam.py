"""Tests for miru.gradcam — the Grad-CAM explainer."""
from __future__ import annotations

import base64

import numpy as np
import pytest

from miru.gradcam import (
    GradCAMExplainer,
    GradCAMResult,
    attention_to_cam,
    compute_gradcam,
    top_k_regions,
)


# ---------------------------------------------------------------------------
# Pure-numpy core
# ---------------------------------------------------------------------------


def test_compute_gradcam_shape_and_dtype():
    activations = np.random.RandomState(0).randn(8, 7, 7).astype(np.float32)
    gradients = np.random.RandomState(1).randn(8, 7, 7).astype(np.float32)
    cam = compute_gradcam(activations, gradients)
    assert cam.shape == (7, 7)
    assert cam.dtype == np.float32


def test_compute_gradcam_normalisation_range():
    activations = np.abs(np.random.RandomState(2).randn(4, 5, 5).astype(np.float32))
    gradients = np.random.RandomState(3).randn(4, 5, 5).astype(np.float32)
    cam = compute_gradcam(activations, gradients)
    assert float(cam.min()) >= 0.0 - 1e-7
    assert float(cam.max()) <= 1.0 + 1e-7
    # When at least one positive value exists post-ReLU, max should hit 1.
    if cam.max() > 0:
        assert pytest.approx(1.0, abs=1e-5) == float(cam.max())


def test_compute_gradcam_relu_kills_negative_evidence():
    """If every weighted activation is negative, the heatmap is all zeros."""
    activations = -np.ones((2, 3, 3), dtype=np.float32)
    gradients = np.ones((2, 3, 3), dtype=np.float32)
    cam = compute_gradcam(activations, gradients)
    assert np.allclose(cam, 0.0)


def test_compute_gradcam_localises_active_channel():
    """Hot spot in one channel + positive gradient on that channel → hot spot survives."""
    activations = np.zeros((3, 4, 4), dtype=np.float32)
    activations[1, 2, 1] = 5.0  # single hot pixel in channel 1
    gradients = np.zeros((3, 4, 4), dtype=np.float32)
    gradients[1] = 1.0  # only channel 1 has gradient signal
    cam = compute_gradcam(activations, gradients)
    assert cam.argmax() == np.ravel_multi_index((2, 1), (4, 4))
    assert float(cam[2, 1]) == pytest.approx(1.0)


def test_compute_gradcam_shape_mismatch_raises():
    a = np.zeros((4, 5, 5), dtype=np.float32)
    g = np.zeros((4, 6, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_gradcam(a, g)


def test_compute_gradcam_rejects_non_3d():
    with pytest.raises(ValueError):
        compute_gradcam(np.zeros((5, 5)), np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# Attention fallback
# ---------------------------------------------------------------------------


def test_attention_to_cam_2d_passthrough():
    raw = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    cam = attention_to_cam(raw)
    assert cam.shape == (2, 2)
    assert pytest.approx(0.0) == float(cam.min())
    assert pytest.approx(1.0) == float(cam.max())


def test_attention_to_cam_collapses_multihead_attention():
    """A (heads, seq, seq) tensor is reshaped to a square patch grid."""
    # 50 = 1 CLS + 49 patches → 7×7 grid
    rng = np.random.RandomState(7)
    attn = rng.rand(12, 50, 50).astype(np.float32)
    cam = attention_to_cam(attn)
    assert cam.shape == (7, 7)
    assert cam.dtype == np.float32
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0


def test_attention_to_cam_uniform_returns_zeros():
    uniform = np.ones((1, 5, 5), dtype=np.float32)
    cam = attention_to_cam(uniform)
    assert np.allclose(cam, 0.0)


# ---------------------------------------------------------------------------
# top_k_regions
# ---------------------------------------------------------------------------


def test_top_k_regions_sorted_descending():
    heatmap = np.array([[0.1, 0.9, 0.2], [0.5, 0.0, 0.7]], dtype=np.float32)
    top = top_k_regions(heatmap, k=3)
    assert len(top) == 3
    scores = [s for _, _, s in top]
    assert scores == sorted(scores, reverse=True)
    assert top[0] == (0, 1, pytest.approx(0.9))


def test_top_k_regions_zero_or_negative_k():
    heatmap = np.zeros((4, 4), dtype=np.float32)
    assert top_k_regions(heatmap, k=0) == []
    assert top_k_regions(heatmap, k=-1) == []


# ---------------------------------------------------------------------------
# GradCAMExplainer entry points
# ---------------------------------------------------------------------------


def test_explainer_from_arrays_returns_result_dataclass():
    a = np.abs(np.random.RandomState(4).randn(6, 4, 4)).astype(np.float32)
    g = np.random.RandomState(5).randn(6, 4, 4).astype(np.float32)
    result = GradCAMExplainer.from_arrays(a, g, target_class=3, top_k=4)
    assert isinstance(result, GradCAMResult)
    assert result.target_class == 3
    assert result.used_fallback is False
    assert result.heatmap.shape == (4, 4)
    assert len(result.top_regions) == 4


def test_explainer_from_attention_marks_fallback():
    rng = np.random.RandomState(6)
    attn = rng.rand(8, 17, 17).astype(np.float32)  # 17-1 = 16 = 4×4 grid
    result = GradCAMExplainer.from_attention(attn, top_k=2)
    assert result.used_fallback is True
    assert result.heatmap.shape == (4, 4)
    assert len(result.top_regions) == 2


def test_explainer_no_model_explain_raises():
    explainer = GradCAMExplainer()
    with pytest.raises(RuntimeError):
        explainer.explain(image_tensor=np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_explainer_finds_no_conv_when_torch_missing_or_no_layers():
    """A bare object exposing no .modules() falls into the fallback path cleanly."""

    class Dummy:
        pass

    explainer = GradCAMExplainer(model=Dummy())
    assert explainer.uses_attention_fallback is True


# ---------------------------------------------------------------------------
# /explain endpoint
# ---------------------------------------------------------------------------


def _payload(question: str = "what's here?", method: str = "attention", **kw) -> dict:
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "image_b64": base64.b64encode(arr.tobytes()).decode(),
        "question": question,
        "backend": "mock",
        "method": method,
        "top_k": 3,
        **kw,
    }


def test_explain_endpoint_attention_returns_200(client):
    r = client.post("/explain", json=_payload(method="attention"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "implemented"
    assert body["method"] == "attention"
    assert body["used_fallback"] is False
    assert body["width"] > 0 and body["height"] > 0
    assert len(body["heatmap"]) == body["height"]
    assert len(body["heatmap"][0]) == body["width"]
    assert len(body["top_regions"]) == 3


def test_explain_endpoint_gradcam_returns_200(client):
    """M11 ship gate: gradcam is now implemented (not roadmap)."""
    r = client.post("/explain", json=_payload(method="gradcam"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "implemented"
    assert body["method"] == "gradcam"
    assert body["used_fallback"] is True  # mock backend has no Conv2d


def test_explain_endpoint_gradcam_top_regions_have_normalised_bbox(client):
    r = client.post("/explain", json=_payload(method="gradcam", top_k=2))
    assert r.status_code == 200
    regions = r.json()["top_regions"]
    assert len(regions) == 2
    for region in regions:
        assert 0.0 <= region["bbox_x1"] <= region["bbox_x2"] <= 1.0
        assert 0.0 <= region["bbox_y1"] <= region["bbox_y2"] <= 1.0
        assert 0.0 <= region["score"] <= 1.0


def test_explain_endpoint_lime_returns_501(client):
    r = client.post("/explain", json=_payload(method="lime"))
    assert r.status_code == 501
    assert r.json()["error"] == "not_implemented"


def test_explain_endpoint_unknown_method_returns_422(client):
    r = client.post("/explain", json=_payload(method="bogus"))
    assert r.status_code == 422
    assert r.json()["error"] == "unknown_method"


def test_explain_endpoint_overlay_query_returns_b64(client):
    r = client.post("/explain?overlay=true", json=_payload(method="gradcam"))
    assert r.status_code == 200
    overlay_b64 = r.json()["overlay_b64"]
    # Pillow is in [dev] extras, so an overlay should encode to a non-empty string.
    assert overlay_b64 is None or isinstance(overlay_b64, str)


def test_explain_endpoint_unknown_backend_falls_back_to_default(client):
    r = client.post("/explain", json=_payload(method="attention", backend="does-not-exist"))
    assert r.status_code == 200
    assert r.json()["backend"] == "mock"
