"""Unit tests for BoundingBox validation and ROI grid-embedding math."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from api.main import BoundingBox


# ---------------------------------------------------------------------------
# BoundingBox model validation
# ---------------------------------------------------------------------------


def test_bounding_box_valid() -> None:
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert bbox.x1 == pytest.approx(0.1)
    assert bbox.y2 == pytest.approx(0.9)


def test_bounding_box_rejects_x1_equal_x2() -> None:
    with pytest.raises(ValidationError, match="x2"):
        BoundingBox(x1=0.5, y1=0.0, x2=0.5, y2=1.0)


def test_bounding_box_rejects_x2_less_than_x1() -> None:
    with pytest.raises(ValidationError, match="x2"):
        BoundingBox(x1=0.7, y1=0.0, x2=0.3, y2=1.0)


def test_bounding_box_rejects_y1_equal_y2() -> None:
    with pytest.raises(ValidationError, match="y2"):
        BoundingBox(x1=0.0, y1=0.5, x2=1.0, y2=0.5)


def test_bounding_box_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=-0.1, y1=0.0, x2=0.5, y2=1.0)
    with pytest.raises(ValidationError):
        BoundingBox(x1=0.0, y1=0.0, x2=1.1, y2=1.0)


def test_bounding_box_full_image() -> None:
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    assert bbox.x2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Grid embedding math — verified via _apply_roi_saliency
# ---------------------------------------------------------------------------


def _make_png_b64(h: int = 32, w: int = 32) -> str:
    """Make a real PNG base64 string for testing."""
    import base64
    import io

    img = np.full((h, w, 3), 32, dtype=np.uint8)
    img[4:12, 4:12] = 240

    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return encode_png_b64(rgba)


def test_roi_grid_zeros_outside_bbox() -> None:
    """Cells outside the ROI bbox must be zero in the returned grid."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    png = _make_png_b64()

    resp = client.post(
        "/explain",
        json={
            "image_b64": png,
            "model_name": "mock",
            "method": "attention",
            "roi": {"x1": 0.5, "y1": 0.5, "x2": 1.0, "y2": 1.0},
        },
    )
    assert resp.status_code == 200, resp.text
    grid = resp.json()["attention_grid"]
    arr = np.array(grid)
    resolution = arr.shape[0]
    mid = resolution // 2
    # Top-left quadrant (rows 0..mid-1, cols 0..mid-1) must be all zero
    assert arr[:mid, :mid].max() == pytest.approx(0.0), "top-left must be zero"


def test_roi_full_image_matches_no_roi_structure() -> None:
    """roi=[0,0,1,1] should produce a non-zero grid (covers whole image)."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    png = _make_png_b64()

    resp = client.post(
        "/explain",
        json={
            "image_b64": png,
            "model_name": "mock",
            "method": "attention",
            "roi": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        },
    )
    assert resp.status_code == 200
    grid = np.array(resp.json()["attention_grid"])
    assert grid.max() > 0.0, "full-image roi must have nonzero saliency"
