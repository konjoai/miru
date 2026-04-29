"""Integration tests for overlay query parameter on POST /analyze — 4 tests."""
from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miru.visualization.overlay import encode_png_b64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def png_image_b64() -> str:
    """Base64-encoded 4×4 white RGBA PNG — valid Pillow-decodable image."""
    arr = np.full((4, 4, 4), 255, dtype=np.uint8)
    return encode_png_b64(arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_without_overlay_returns_none(client: TestClient, mock_image_b64: str) -> None:
    """POST /analyze without overlay=true must return overlay_b64 == null."""
    payload = {"image_b64": mock_image_b64, "question": "What is shown?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "overlay_b64" in data, "overlay_b64 field must be present in response"
    assert data["overlay_b64"] is None, (
        f"overlay_b64 should be null when overlay=false, got {data['overlay_b64']!r}"
    )


def test_analyze_with_overlay_returns_nonempty_string(
    client: TestClient, png_image_b64: str
) -> None:
    """POST /analyze?overlay=true with a valid PNG image returns a non-empty overlay_b64."""
    payload = {
        "image_b64": png_image_b64,
        "question": "Describe the image.",
        "backend": "mock",
    }
    resp = client.post("/analyze?overlay=true", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "overlay_b64" in data
    assert data["overlay_b64"] is not None, "overlay_b64 should not be null when overlay=true"
    assert isinstance(data["overlay_b64"], str)
    assert len(data["overlay_b64"]) > 0, "overlay_b64 must not be an empty string"


def test_analyze_with_overlay_invalid_image_does_not_crash(
    client: TestClient,
) -> None:
    """POST /analyze?overlay=true with an invalid image returns valid trace without crashing."""
    # Raw bytes that are not a valid PNG/JPEG — overlay generation should silently fall back.
    raw = b"\x00\x01\x02\x03" * 12  # not a valid image format
    bad_b64 = base64.b64encode(raw).decode()
    payload = {"image_b64": bad_b64, "question": "Test?", "backend": "mock"}
    resp = client.post("/analyze?overlay=true", json=payload)
    # Must not crash; the trace itself is still valid.
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "attention_map" in data
    # overlay_b64 may be null on failure — that is acceptable.
    assert "overlay_b64" in data


def test_health_still_returns_200_after_overlay_import(client: TestClient) -> None:
    """Regression guard: GET /health still returns 200 after visualization module import."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
