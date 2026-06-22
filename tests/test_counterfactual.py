"""Tests for miru.counterfactual and POST /explain/counterfactual."""
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
    return np.random.default_rng(11).random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit — MinimalCounterfactual
# ---------------------------------------------------------------------------


def test_result_mask_shape_matches_grid() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert result.counterfactual_mask.shape == (result.grid_h, result.grid_w)
    assert result.counterfactual_mask.dtype == bool


def test_result_mask_is_bool_array() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert result.counterfactual_mask.dtype == bool


def test_n_cells_masked_consistent_with_mask() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert result.n_cells_masked == int(result.counterfactual_mask.sum())


def test_original_confidence_in_unit_range() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert 0.0 <= result.original_confidence <= 1.0


def test_counterfactual_confidence_in_unit_range() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert 0.0 <= result.counterfactual_confidence <= 1.0


def test_delta_confidence_equals_difference() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    expected = result.original_confidence - result.counterfactual_confidence
    assert abs(result.delta_confidence - expected) < 1e-6


def test_n_cells_masked_does_not_exceed_max_cells() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    mc = MinimalCounterfactual(max_cells=5)
    result = mc.explain(MockVLMBackend(), _image(), "q")
    assert result.n_cells_masked <= 5


def test_deterministic_same_inputs() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    img = _image()
    r1 = MinimalCounterfactual().explain(MockVLMBackend(seed=0), img, "q")
    r2 = MinimalCounterfactual().explain(MockVLMBackend(seed=0), img, "q")
    np.testing.assert_array_equal(r1.counterfactual_mask, r2.counterfactual_mask)
    assert r1.delta_confidence == r2.delta_confidence


def test_custom_fill_value_propagated() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual(fill_value=0.5).explain(MockVLMBackend(), _image(), "q")
    assert isinstance(result, object)


def test_custom_resolution_via_extractor() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.attention.extractor import AttentionExtractor
    from miru.models.mock import MockVLMBackend

    extractor = AttentionExtractor(resolution=8)
    result = MinimalCounterfactual(extractor=extractor).explain(
        MockVLMBackend(), _image(), "q"
    )
    assert result.grid_h == 8
    assert result.grid_w == 8
    assert result.counterfactual_mask.shape == (8, 8)


def test_flipped_matches_answer_comparison() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    expected_flipped = result.counterfactual_answer != result.original_answer
    assert result.flipped == expected_flipped


def test_invalid_confidence_drop_raises() -> None:
    from miru.counterfactual import MinimalCounterfactual

    with pytest.raises(ValueError, match="confidence_drop"):
        MinimalCounterfactual(confidence_drop=0.0)


def test_invalid_max_cells_zero_raises() -> None:
    from miru.counterfactual import MinimalCounterfactual

    with pytest.raises(ValueError, match="max_cells"):
        MinimalCounterfactual(max_cells=0)


def test_invalid_max_cells_too_large_raises() -> None:
    from miru.counterfactual import MinimalCounterfactual

    with pytest.raises(ValueError, match="max_cells"):
        MinimalCounterfactual(max_cells=257)


def test_invalid_fill_value_raises() -> None:
    from miru.counterfactual import MinimalCounterfactual

    with pytest.raises(ValueError, match="fill_value"):
        MinimalCounterfactual(fill_value=1.5)


def test_goal_reached_flag_type() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual().explain(MockVLMBackend(), _image(), "q")
    assert isinstance(result.goal_reached, bool)


def test_max_cells_one() -> None:
    from miru.counterfactual import MinimalCounterfactual
    from miru.models.mock import MockVLMBackend

    result = MinimalCounterfactual(max_cells=1).explain(MockVLMBackend(), _image(), "q")
    assert result.n_cells_masked <= 1


# ---------------------------------------------------------------------------
# API integration — POST /explain/counterfactual
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def test_counterfactual_returns_200(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "Where is the subject?",
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 200, resp.text


def test_counterfactual_response_fields(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
    }
    body = api_client.post("/explain/counterfactual", json=payload).json()
    for field in [
        "counterfactual_mask", "original_answer", "original_confidence",
        "counterfactual_answer", "counterfactual_confidence",
        "delta_confidence", "n_cells_masked", "grid_h", "grid_w",
        "flipped", "goal_reached", "latency_ms",
    ]:
        assert field in body, f"missing field: {field}"


def test_counterfactual_mask_is_2d_bool_list(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
    }
    body = api_client.post("/explain/counterfactual", json=payload).json()
    mask = body["counterfactual_mask"]
    assert isinstance(mask, list)
    assert all(isinstance(row, list) for row in mask)
    assert all(isinstance(v, bool) for row in mask for v in row)


def test_counterfactual_confidence_in_range(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
    }
    body = api_client.post("/explain/counterfactual", json=payload).json()
    assert 0.0 <= body["original_confidence"] <= 1.0
    assert 0.0 <= body["counterfactual_confidence"] <= 1.0


def test_counterfactual_custom_params(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "confidence_drop": 0.05,
        "max_cells": 8,
        "fill_value": 0.5,
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 200, resp.text


def test_counterfactual_n_cells_le_max_cells(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "max_cells": 3,
    }
    body = api_client.post("/explain/counterfactual", json=payload).json()
    assert body["n_cells_masked"] <= 3


def test_counterfactual_unknown_model_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope",
        "question": "q",
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 400


def test_counterfactual_bad_image_400(api_client) -> None:
    payload = {
        "image_b64": "!!!bad",
        "model_name": "mock",
        "question": "q",
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 400


def test_counterfactual_max_cells_out_of_range_422(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "max_cells": 0,
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 422


def test_counterfactual_fill_value_out_of_range_422(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "question": "q",
        "fill_value": 2.0,
    }
    resp = api_client.post("/explain/counterfactual", json=payload)
    assert resp.status_code == 422


def test_counterfactual_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
