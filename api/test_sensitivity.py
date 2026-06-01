"""HTTP tests for POST /explain/sensitivity (api/main.py).

Self-contained: builds its own real PNG so the image decoder is exercised.
Run from the repo root:

    python -m pytest api/test_sensitivity.py -v
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
def synthetic_png_b64() -> str:
    """A real 16×16 RGB PNG with a bright spot."""
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

        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return encode_png_b64(rgba)


def _post(client: TestClient, image_b64: str, **overrides) -> object:
    body = {"image_b64": image_b64, "model_name": "mock", "question": "what is here"}
    body.update(overrides)
    return client.post("/explain/sensitivity", json=body)


# ---------------------------------------------------------------------------
# Happy path + response contract
# ---------------------------------------------------------------------------


def test_sensitivity_happy_path(client: TestClient, synthetic_png_b64: str) -> None:
    resp = _post(client, synthetic_png_b64)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_name"] == "mock"
    assert body["method"] == "attention"
    assert 0.0 <= body["stability_score"] <= 1.0
    assert isinstance(body["is_stable"], bool)
    assert len(body["per_sigma"]) == 3
    assert body["latency_ms"] >= 0.0


def test_sensitivity_per_sigma_fields(
    client: TestClient, synthetic_png_b64: str
) -> None:
    body = _post(client, synthetic_png_b64, sigmas=[0.02, 0.08]).json()
    assert [p["sigma"] for p in body["per_sigma"]] == [0.02, 0.08]
    for p in body["per_sigma"]:
        assert p["mean_drift"] >= 0.0
        assert p["max_drift"] >= p["mean_drift"]


def test_sensitivity_mock_attention_is_stable(
    client: TestClient, synthetic_png_b64: str
) -> None:
    """Mock attention is question-driven, not image-driven → can't drift."""
    body = _post(client, synthetic_png_b64, method="attention").json()
    assert body["is_stable"] is True
    assert body["stability_score"] == pytest.approx(1.0)


def test_sensitivity_is_deterministic(
    client: TestClient, synthetic_png_b64: str
) -> None:
    a = _post(client, synthetic_png_b64, method="gradcam", seed=7).json()
    b = _post(client, synthetic_png_b64, method="gradcam", seed=7).json()
    assert a["stability_score"] == b["stability_score"]
    assert [p["mean_drift"] for p in a["per_sigma"]] == [
        p["mean_drift"] for p in b["per_sigma"]
    ]


def test_sensitivity_gradcam_runs(client: TestClient, synthetic_png_b64: str) -> None:
    body = _post(client, synthetic_png_b64, method="gradcam", n_trials=2).json()
    assert 0.0 <= body["stability_score"] <= 1.0
    assert body["worst_sigma"] in [p["sigma"] for p in body["per_sigma"]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_sensitivity_rejects_out_of_range_sigma(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = _post(client, synthetic_png_b64, sigmas=[1.5])
    assert resp.status_code == 400


def test_sensitivity_rejects_empty_sigmas(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = _post(client, synthetic_png_b64, sigmas=[])
    assert resp.status_code == 400


def test_sensitivity_rejects_too_many_sigmas(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = _post(client, synthetic_png_b64, sigmas=[0.05] * 9)
    assert resp.status_code == 400


def test_sensitivity_rejects_unknown_method(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = _post(client, synthetic_png_b64, method="bogus")
    assert resp.status_code == 400


def test_sensitivity_rejects_unknown_model(
    client: TestClient, synthetic_png_b64: str
) -> None:
    resp = _post(client, synthetic_png_b64, model_name="ghost")
    assert resp.status_code == 400


def test_sensitivity_rejects_bad_image(client: TestClient) -> None:
    resp = client.post(
        "/explain/sensitivity",
        json={"image_b64": "not-a-real-image", "model_name": "mock"},
    )
    assert resp.status_code == 400


def test_sensitivity_rejects_out_of_range_trials(
    client: TestClient, synthetic_png_b64: str
) -> None:
    # n_trials > MAX_SENSITIVITY_TRIALS (8) is rejected by pydantic (422).
    resp = _post(client, synthetic_png_b64, n_trials=99)
    assert resp.status_code == 422
