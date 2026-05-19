"""Tests for miru.shap_explainer — SHAP perturbation-based explainer.

All tests use MockVLMBackend; no real model is required.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from miru.models.mock import MockVLMBackend
from miru.shap_explainer import SHAPConfig, SHAPExplainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> MockVLMBackend:
    """Deterministic mock backend — safe for any test that touches infer()."""
    return MockVLMBackend(seed=42)


@pytest.fixture
def small_image() -> Image.Image:
    """16×16 RGB PIL image with a bright quadrant — small for fast runs."""
    arr = np.full((16, 16, 3), 32, dtype=np.uint8)
    arr[4:8, 4:8] = 220
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def fast_cfg() -> SHAPConfig:
    """Tiny config so tests finish in under a second."""
    return SHAPConfig(grid_size=3, n_samples=4, seed=0)


def _synthetic_png_b64(h: int = 16, w: int = 16) -> str:
    """Return a base64-encoded PNG fixture for API tests."""
    arr = np.full((h, w, 3), 32, dtype=np.uint8)
    arr[4:8, 4:8] = 200
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# SHAPConfig tests
# ---------------------------------------------------------------------------


def test_shap_config_defaults() -> None:
    """Default SHAPConfig must match the spec values."""
    cfg = SHAPConfig()
    assert cfg.grid_size == 7
    assert cfg.n_samples == 64
    assert cfg.baseline == "mean"
    assert cfg.seed == 42
    assert cfg.batch_size == 8


def test_shap_config_grid_size_7() -> None:
    """Explicit grid_size=7 round-trips correctly."""
    cfg = SHAPConfig(grid_size=7)
    assert cfg.grid_size == 7


def test_shap_config_n_samples_64() -> None:
    """Explicit n_samples=64 round-trips correctly."""
    cfg = SHAPConfig(n_samples=64)
    assert cfg.n_samples == 64


# ---------------------------------------------------------------------------
# SHAPExplainer.explain() tests
# ---------------------------------------------------------------------------


def test_explain_returns_ndarray(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """explain() must return a numpy ndarray."""
    explainer = SHAPExplainer(backend, fast_cfg)
    result = explainer.explain(small_image)
    assert isinstance(result, np.ndarray)


def test_explain_shape_matches_grid(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """Attribution shape must equal (grid_size, grid_size)."""
    explainer = SHAPExplainer(backend, fast_cfg)
    result = explainer.explain(small_image)
    g = fast_cfg.grid_size
    assert result.shape == (g, g)


def test_explain_dtype_float32(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """Attribution dtype must be float32."""
    explainer = SHAPExplainer(backend, fast_cfg)
    result = explainer.explain(small_image)
    assert result.dtype == np.float32


def test_explain_values_in_minus1_to_1(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """All attribution values must lie in [-1, 1]."""
    explainer = SHAPExplainer(backend, fast_cfg)
    result = explainer.explain(small_image)
    assert float(result.min()) >= -1.0
    assert float(result.max()) <= 1.0


def test_explain_seeded_deterministic(backend: MockVLMBackend, small_image: Image.Image) -> None:
    """Same seed → identical attribution map."""
    cfg = SHAPConfig(grid_size=3, n_samples=4, seed=7)
    a = SHAPExplainer(backend, cfg).explain(small_image)
    b = SHAPExplainer(backend, cfg).explain(small_image)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# explain_to_attention_map() tests
# ---------------------------------------------------------------------------


def test_explain_to_attention_map_shape(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """explain_to_attention_map() must return an array shaped (H, W) matching the image."""
    explainer = SHAPExplainer(backend, fast_cfg)
    result = explainer.explain_to_attention_map(small_image)
    w, h = small_image.size
    assert result.shape == (h, w)


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


def test_baseline_mean_is_image_mean(backend: MockVLMBackend, small_image: Image.Image) -> None:
    """baseline='mean' fill must equal the per-channel mean of the image."""
    explainer = SHAPExplainer(backend, SHAPConfig(baseline="mean"))
    fill = explainer._compute_baseline_fill(small_image)
    arr = np.array(small_image, dtype=np.float32)
    expected = arr.mean(axis=(0, 1))
    np.testing.assert_allclose(fill, expected, rtol=1e-5)


def test_baseline_black_is_zeros(backend: MockVLMBackend, small_image: Image.Image) -> None:
    """baseline='black' fill must be all zeros."""
    explainer = SHAPExplainer(backend, SHAPConfig(baseline="black"))
    fill = explainer._compute_baseline_fill(small_image)
    np.testing.assert_array_equal(fill, np.zeros(3, dtype=np.float32))


def test_baseline_white_is_255(backend: MockVLMBackend, small_image: Image.Image) -> None:
    """baseline='white' fill must be all 255."""
    explainer = SHAPExplainer(backend, SHAPConfig(baseline="white"))
    fill = explainer._compute_baseline_fill(small_image)
    np.testing.assert_array_equal(fill, np.full(3, 255.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# _make_masked_image() tests
# ---------------------------------------------------------------------------


def test_make_masked_image_size_preserved(backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig) -> None:
    """Masked image must have the same size as the original."""
    explainer = SHAPExplainer(backend, fast_cfg)
    arr = np.array(small_image, dtype=np.float32)
    fill = explainer._compute_baseline_fill(small_image)
    mask = np.ones((fast_cfg.grid_size, fast_cfg.grid_size), dtype=bool)
    result = explainer._make_masked_image(arr, mask, fill)
    assert result.size == small_image.size


def test_make_masked_image_all_present_equals_original(
    backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig
) -> None:
    """All-present mask must produce an image pixel-equal to the original."""
    explainer = SHAPExplainer(backend, fast_cfg)
    arr = np.array(small_image, dtype=np.float32)
    fill = explainer._compute_baseline_fill(small_image)
    mask = np.ones((fast_cfg.grid_size, fast_cfg.grid_size), dtype=bool)
    result = explainer._make_masked_image(arr, mask, fill)
    np.testing.assert_array_equal(np.array(result), np.array(small_image))


def test_make_masked_image_all_absent_equals_baseline(
    backend: MockVLMBackend, small_image: Image.Image, fast_cfg: SHAPConfig
) -> None:
    """All-absent mask must produce a uniform-colour baseline image."""
    explainer = SHAPExplainer(backend, fast_cfg)
    arr = np.array(small_image, dtype=np.float32)
    fill = explainer._compute_baseline_fill(small_image)
    mask = np.zeros((fast_cfg.grid_size, fast_cfg.grid_size), dtype=bool)
    result = explainer._make_masked_image(arr, mask, fill)
    out_arr = np.array(result, dtype=np.float32)
    fill_uint8 = np.clip(fill, 0.0, 255.0).astype(np.uint8).astype(np.float32)
    expected = np.broadcast_to(fill_uint8, out_arr.shape)
    np.testing.assert_array_equal(out_arr, expected)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_shap_registered_in_explainer_registry() -> None:
    """'shap' must appear in IMPLEMENTED_METHODS after registration."""
    from api.main import IMPLEMENTED_METHODS
    assert "shap" in IMPLEMENTED_METHODS


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------


def test_api_explain_shap_method() -> None:
    """POST /explain?method=shap must return 200."""
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    client = TestClient(app)
    payload = {
        "image_b64": _synthetic_png_b64(),
        "model_name": "mock",
        "method": "shap",
        "question": "Where?",
        "top_k": 3,
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "shap"
    grid = body["attention_grid"]
    flat = [v for row in grid for v in row]
    # API normalises [-1,1] attribution → [0,1] for the overlay pipeline.
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_api_methods_includes_shap() -> None:
    """GET /methods must list 'shap' as implemented."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/methods")
    assert resp.status_code == 200
    statuses = {m["name"]: m["status"] for m in resp.json()["methods"]}
    assert "shap" in statuses
    assert statuses["shap"] == "implemented"
