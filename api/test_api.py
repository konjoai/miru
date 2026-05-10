"""Tests for the deployable Miru explainability API (api/main.py).

These tests are deliberately self-contained — they construct their own
synthetic images and do not rely on the in-package /analyze test fixtures.
Run from the repo root:

    python -m pytest api/test_api.py -v
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import (
    DEFAULT_BENCH_N,
    IMPLEMENTED_METHODS,
    MAX_BENCH_N,
    ROADMAP_METHODS,
    app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def synthetic_png_b64() -> str:
    """A real 16x16 RGB PNG with a bright spot — exercises the image decoder.

    Falls back to encode_png_b64 from miru if Pillow is unavailable.
    """
    h = w = 16
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    img[4:8, 4:8] = (240, 240, 240)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate(
            [img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
        )
        return encode_png_b64(rgba)


@pytest.fixture
def malformed_b64() -> str:
    """Base64 of bytes that are not a real image — used to test the 400 path."""
    return base64.b64encode(b"not-an-image").decode("ascii")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_status_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    assert "mock" in body["backends"]
    assert "attention" in body["methods"]


# ---------------------------------------------------------------------------
# /methods
# ---------------------------------------------------------------------------


def test_methods_lists_implemented_and_roadmap(client: TestClient) -> None:
    resp = client.get("/methods")
    assert resp.status_code == 200
    body = resp.json()

    statuses = {m["name"]: m["status"] for m in body["methods"]}
    for name in IMPLEMENTED_METHODS:
        assert statuses[name] == "implemented"
    for name in ROADMAP_METHODS:
        assert statuses[name] == "roadmap"

    assert "mock" in body["models"]
    assert body["default_model"] == "mock"


# ---------------------------------------------------------------------------
# /explain — happy path + error contracts
# ---------------------------------------------------------------------------


def test_explain_with_png_returns_overlay(client: TestClient, synthetic_png_b64: str) -> None:
    payload = {
        "image_b64": synthetic_png_b64,
        "model_name": "mock",
        "method": "attention",
        "question": "Where is the bright spot?",
        "top_k": 3,
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["model_name"] == "mock"
    assert body["method"] == "attention"
    assert isinstance(body["answer"], str) and body["answer"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["overlay_b64"], str) and len(body["overlay_b64"]) > 100
    assert len(body["top_regions"]) == 3
    for r in body["top_regions"]:
        assert {"row", "col", "score"} <= set(r)
    grid = body["attention_grid"]
    assert isinstance(grid, list) and len(grid) > 0 and len(grid[0]) > 0
    flat = [v for row in grid for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)
    assert body["latency_ms"] >= 0.0


def test_explain_malformed_image_returns_400(client: TestClient, malformed_b64: str) -> None:
    resp = client.post(
        "/explain",
        json={"image_b64": malformed_b64, "model_name": "mock", "method": "attention"},
    )
    assert resp.status_code == 400
    assert "image_b64" in resp.json()["detail"]


def test_explain_unknown_model_returns_400(client: TestClient, synthetic_png_b64: str) -> None:
    resp = client.post(
        "/explain",
        json={"image_b64": synthetic_png_b64, "model_name": "does-not-exist", "method": "attention"},
    )
    assert resp.status_code == 400
    assert "does-not-exist" in resp.json()["detail"]


def test_explain_roadmap_method_returns_400(client: TestClient, synthetic_png_b64: str) -> None:
    """Any method in ROADMAP_METHODS must be rejected with a roadmap message.

    Picks the first roadmap method dynamically so this stays green as
    methods are promoted from roadmap → implemented.
    """
    if not ROADMAP_METHODS:
        pytest.skip("no roadmap methods left to test")
    method = ROADMAP_METHODS[0]
    resp = client.post(
        "/explain",
        json={"image_b64": synthetic_png_b64, "model_name": "mock", "method": method},
    )
    assert resp.status_code == 400
    assert "roadmap" in resp.json()["detail"].lower()


def test_explain_unknown_method_returns_400(client: TestClient, synthetic_png_b64: str) -> None:
    resp = client.post(
        "/explain",
        json={"image_b64": synthetic_png_b64, "model_name": "mock", "method": "totally-bogus"},
    )
    assert resp.status_code == 400
    assert "totally-bogus" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /benchmark
# ---------------------------------------------------------------------------


def test_benchmark_mock_runs_and_aggregates(client: TestClient) -> None:
    resp = client.post(
        "/benchmark",
        json={"model_name": "mock", "n": 6, "seed": 7, "size": 32},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["backend"] == "mock"
    assert body["n"] == 6
    assert body["seed"] == 7
    assert body["size"] == 32
    for metric in ("iou", "auc", "hit1", "latency_ms"):
        stats = body[metric]
        assert stats["n"] == 6
        assert "mean" in stats and "std" in stats and "p50" in stats and "p95" in stats
    assert 0.0 <= body["iou"]["mean"] <= 1.0
    assert 0.0 <= body["auc"]["mean"] <= 1.0
    assert body["latency_ms"]["mean"] >= 0.0
    assert isinstance(body["timestamp_utc"], str)


def test_benchmark_unknown_model_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/benchmark",
        json={"model_name": "missing", "n": 4, "seed": 1, "size": 32},
    )
    assert resp.status_code == 400


def test_benchmark_n_above_cap_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/benchmark",
        json={"model_name": "mock", "n": MAX_BENCH_N + 1, "seed": 1, "size": 32},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /compare
# ---------------------------------------------------------------------------


def test_compare_same_backend_is_tie(client: TestClient) -> None:
    """mock vs mock with the same seed must be a perfect tie (zero delta)."""
    resp = client.post(
        "/compare",
        json={"model_a": "mock", "model_b": "mock", "n": 5, "seed": 11},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["model_a"] == "mock"
    assert body["model_b"] == "mock"
    assert body["winner"] == "tie"
    assert body["paired_iou_delta"] == pytest.approx(0.0, abs=1e-9)
    assert body["a_iou"]["mean"] == pytest.approx(body["b_iou"]["mean"], abs=1e-9)


def test_compare_unknown_model_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/compare",
        json={"model_a": "mock", "model_b": "nope", "n": 4, "seed": 1},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Default-arg sanity — protects against accidental schema regression.
# ---------------------------------------------------------------------------


def test_default_bench_n_is_within_cap() -> None:
    assert 1 <= DEFAULT_BENCH_N <= MAX_BENCH_N


# ---------------------------------------------------------------------------
# /explain — newly-implemented methods (lime, gradcam)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["attention", "lime", "gradcam"])
def test_explain_each_implemented_method(
    client: TestClient, synthetic_png_b64: str, method: str
) -> None:
    payload = {
        "image_b64": synthetic_png_b64,
        "model_name": "mock",
        "method": method,
        "question": "where?",
        "top_k": 3,
        # Keep budgets tiny for fast tests.
        "n_samples": 8,
        "n_segments": 9,
        "occlusion_grid": 4,
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == method
    grid = body["attention_grid"]
    flat = [v for row in grid for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)
    assert len(body["overlay_b64"]) > 100


# ---------------------------------------------------------------------------
# /explain/compare
# ---------------------------------------------------------------------------


def test_explain_compare_returns_two_overlays(client: TestClient, synthetic_png_b64: str) -> None:
    payload = {
        "image_b64": synthetic_png_b64,
        "model_name": "mock",
        "method_a": "attention",
        "method_b": "gradcam",
        "occlusion_grid": 4,
        "top_k": 2,
    }
    resp = client.post("/explain/compare", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method_a"] == "attention"
    assert body["method_b"] == "gradcam"
    assert len(body["a_overlay_b64"]) > 100
    assert len(body["b_overlay_b64"]) > 100
    assert len(body["a_top_regions"]) == 2
    assert len(body["b_top_regions"]) == 2
    a_grid = body["a_attention_grid"]
    b_grid = body["b_attention_grid"]
    assert len(a_grid) == len(b_grid)
    # Different methods should generally produce different heatmaps.
    a_flat = [v for row in a_grid for v in row]
    b_flat = [v for row in b_grid for v in row]
    assert a_flat != b_flat


def test_explain_compare_same_method_returns_400(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = client.post(
        "/explain/compare",
        json={
            "image_b64": synthetic_png_b64,
            "model_name": "mock",
            "method_a": "attention",
            "method_b": "attention",
        },
    )
    assert resp.status_code == 400
    assert "differ" in resp.json()["detail"]


def test_explain_compare_unknown_method_returns_400(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = client.post(
        "/explain/compare",
        json={
            "image_b64": synthetic_png_b64,
            "model_name": "mock",
            "method_a": "attention",
            "method_b": "shap",
        },
    )
    assert resp.status_code == 400


def test_methods_lists_lime_and_gradcam_as_implemented(client: TestClient) -> None:
    resp = client.get("/methods")
    assert resp.status_code == 200
    statuses = {m["name"]: m["status"] for m in resp.json()["methods"]}
    assert statuses["attention"] == "implemented"
    assert statuses["lime"] == "implemented"
    assert statuses["gradcam"] == "implemented"
    assert statuses["shap"] == "roadmap"
