"""HTTP tests for ROI-targeted explanation (POST /explain with roi field).

Phase 24: bounding-box parameter restricts saliency computation to the
sub-region; cells outside the ROI are zero; the VLM answer comes from
the full image.

Run from the repo root:

    python -m pytest api/test_roi.py -v
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def png_b64() -> str:
    """32×32 RGB PNG with a bright 8×8 patch."""
    h = w = 32
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    img[4:12, 4:12] = (240, 200, 100)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return encode_png_b64(rgba)


def _explain(client: TestClient, image_b64: str, **kwargs) -> object:
    body: dict = {"image_b64": image_b64, "model_name": "mock"}
    body.update(kwargs)
    return client.post("/explain", json=body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_roi_returns_200(client: TestClient, png_b64: str) -> None:
    resp = _explain(client, png_b64, roi={"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "attention_grid" in body
    assert "analysis_id" in body
    assert body["model_name"] == "mock"


def test_roi_zeros_outside_region(client: TestClient, png_b64: str) -> None:
    """Bottom-right quadrant roi → top-left half of grid must be zero."""
    resp = _explain(
        client,
        png_b64,
        method="attention",
        roi={"x1": 0.5, "y1": 0.5, "x2": 1.0, "y2": 1.0},
    )
    assert resp.status_code == 200
    grid = np.array(resp.json()["attention_grid"])
    res = grid.shape[0]
    mid = res // 2
    assert grid[:mid, :mid].max() == pytest.approx(0.0), (
        "top-left cells must be zero when roi is bottom-right"
    )


def test_roi_nonzero_inside_region(client: TestClient, png_b64: str) -> None:
    """The ROI sub-region must contain at least one nonzero cell."""
    resp = _explain(
        client,
        png_b64,
        method="attention",
        roi={"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5},
    )
    assert resp.status_code == 200
    grid = np.array(resp.json()["attention_grid"])
    res = grid.shape[0]
    mid = res // 2
    assert grid[:mid, :mid].max() > 0.0, "roi region must have nonzero saliency"


def test_no_roi_still_works(client: TestClient, png_b64: str) -> None:
    """Omitting roi (default=None) preserves existing behaviour."""
    resp = _explain(client, png_b64, method="attention")
    assert resp.status_code == 200
    grid = np.array(resp.json()["attention_grid"])
    assert grid.max() > 0.0


def test_roi_with_gradcam(client: TestClient, png_b64: str) -> None:
    resp = _explain(
        client,
        png_b64,
        method="gradcam",
        occlusion_grid=4,
        roi={"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "gradcam"
    grid = np.array(body["attention_grid"])
    # Grid shape and dtype are the only invariants we can assert for gradcam
    # with the mock backend on a small crop — the mock may produce uniform
    # occlusion sensitivity (all zeros after normalisation), which is valid.
    assert grid.ndim == 2
    assert grid.shape[0] == grid.shape[1]
    assert float(grid.min()) >= 0.0
    assert float(grid.max()) <= 1.0


def test_roi_with_lime(client: TestClient, png_b64: str) -> None:
    resp = _explain(
        client,
        png_b64,
        method="lime",
        n_samples=4,
        n_segments=4,
        roi={"x1": 0.0, "y1": 0.0, "x2": 0.6, "y2": 0.6},
    )
    assert resp.status_code == 200
    assert resp.json()["method"] == "lime"


def test_roi_with_shap(client: TestClient, png_b64: str) -> None:
    resp = _explain(
        client,
        png_b64,
        method="shap",
        shap_grid=3,
        shap_samples=4,
        roi={"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
    )
    assert resp.status_code == 200
    assert resp.json()["method"] == "shap"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_roi_x1_gte_x2_rejected(client: TestClient, png_b64: str) -> None:
    resp = _explain(client, png_b64, roi={"x1": 0.7, "y1": 0.0, "x2": 0.3, "y2": 1.0})
    assert resp.status_code == 422


def test_roi_y1_gte_y2_rejected(client: TestClient, png_b64: str) -> None:
    resp = _explain(client, png_b64, roi={"x1": 0.0, "y1": 0.8, "x2": 1.0, "y2": 0.2})
    assert resp.status_code == 422


def test_roi_out_of_range_rejected(client: TestClient, png_b64: str) -> None:
    resp = _explain(client, png_b64, roi={"x1": -0.1, "y1": 0.0, "x2": 0.5, "y2": 1.0})
    assert resp.status_code == 422


def test_roi_too_small_rejected(client: TestClient, png_b64: str) -> None:
    """A roi that maps to < 4 pixels should return 400."""
    resp = _explain(
        client,
        png_b64,
        roi={"x1": 0.0, "y1": 0.0, "x2": 0.001, "y2": 0.001},
    )
    assert resp.status_code == 400
    assert "4×4" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Cache key isolation
# ---------------------------------------------------------------------------


def test_roi_affects_cache_key(client: TestClient, png_b64: str) -> None:
    """Two requests with different ROIs should return different grids."""
    resp_a = _explain(
        client,
        png_b64,
        method="attention",
        roi={"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5},
    )
    resp_b = _explain(
        client,
        png_b64,
        method="attention",
        roi={"x1": 0.5, "y1": 0.5, "x2": 1.0, "y2": 1.0},
    )
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    grid_a = np.array(resp_a.json()["attention_grid"])
    grid_b = np.array(resp_b.json()["attention_grid"])
    assert not np.allclose(grid_a, grid_b), (
        "different rois must produce different grids"
    )
