"""Tests for miru.dataset_analytics and POST /analyze/batch."""
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


def _grid(size: int = 8, fill: float = 0.5) -> np.ndarray:
    return np.full((size, size), fill, dtype=np.float32)


def _hot_top_left(size: int = 8) -> np.ndarray:
    g = np.zeros((size, size), dtype=np.float32)
    g[: size // 2, : size // 2] = 1.0
    return g


# ---------------------------------------------------------------------------
# Unit — aggregate_saliency
# ---------------------------------------------------------------------------


def test_aggregate_single_grid_mean_equals_grid() -> None:
    from miru.dataset_analytics import aggregate_saliency

    g = _hot_top_left()
    mean, std = aggregate_saliency([g])
    np.testing.assert_allclose(mean, g, atol=1e-6)


def test_aggregate_single_grid_std_is_zero() -> None:
    from miru.dataset_analytics import aggregate_saliency

    g = _hot_top_left()
    _, std = aggregate_saliency([g])
    assert float(std.max()) == pytest.approx(0.0, abs=1e-6)


def test_aggregate_mean_dtype_float32() -> None:
    from miru.dataset_analytics import aggregate_saliency

    mean, _ = aggregate_saliency([_grid()])
    assert mean.dtype == np.float32


def test_aggregate_shape_matches_first_grid() -> None:
    from miru.dataset_analytics import aggregate_saliency

    g1 = np.zeros((8, 8), dtype=np.float32)
    g2 = np.ones((16, 16), dtype=np.float32)
    mean, _ = aggregate_saliency([g1, g2])
    assert mean.shape == (8, 8)


def test_aggregate_two_grids_mean_correct() -> None:
    from miru.dataset_analytics import aggregate_saliency

    g1 = _grid(fill=0.2)
    g2 = _grid(fill=0.8)
    mean, _ = aggregate_saliency([g1, g2])
    np.testing.assert_allclose(mean, _grid(fill=0.5), atol=1e-6)


def test_aggregate_empty_list_raises() -> None:
    from miru.dataset_analytics import aggregate_saliency

    with pytest.raises(ValueError, match="non-empty"):
        aggregate_saliency([])


# ---------------------------------------------------------------------------
# Unit — detect_spurious
# ---------------------------------------------------------------------------


def test_detect_spurious_below_min_samples_returns_empty() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = _grid(fill=0.9)
    std = _grid(fill=0.01)
    mask, cells = detect_spurious(mean, std, n_samples=2)
    assert not mask.any()
    assert cells == []


def test_detect_spurious_high_mean_low_cv_flagged() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = _grid(fill=0.9)  # all > 0.5
    std = _grid(fill=0.01)  # CV = 0.01/0.9 ≈ 0.011 < 0.5
    mask, cells = detect_spurious(mean, std, n_samples=5)
    assert mask.all()
    assert len(cells) == mean.size


def test_detect_spurious_high_cv_not_flagged() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = _grid(fill=0.9)
    std = _grid(fill=0.9)  # CV = 1.0 > 0.5
    mask, cells = detect_spurious(mean, std, n_samples=5)
    assert not mask.any()
    assert cells == []


def test_detect_spurious_low_mean_not_flagged() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = _grid(fill=0.1)  # below threshold 0.5
    std = _grid(fill=0.01)
    mask, cells = detect_spurious(mean, std, n_samples=5)
    assert not mask.any()
    assert cells == []


def test_detect_spurious_cells_sorted_by_mean_desc() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = np.zeros((4, 4), dtype=np.float32)
    mean[0, 0] = 0.9
    mean[0, 1] = 0.7
    std = np.zeros((4, 4), dtype=np.float32)
    _, cells = detect_spurious(mean, std, n_samples=5)
    assert len(cells) == 2
    assert cells[0] == (0, 0)  # highest mean first
    assert cells[1] == (0, 1)


def test_detect_spurious_zero_mean_cells_not_flagged() -> None:
    from miru.dataset_analytics import detect_spurious

    mean = np.zeros((4, 4), dtype=np.float32)
    std = np.zeros((4, 4), dtype=np.float32)
    _, cells = detect_spurious(mean, std, n_samples=5)
    assert cells == []


# ---------------------------------------------------------------------------
# Unit — analyse_dataset
# ---------------------------------------------------------------------------


def test_analyse_dataset_shape_contract() -> None:
    from miru.dataset_analytics import analyse_dataset

    grids = [_hot_top_left() for _ in range(3)]
    result = analyse_dataset(grids)
    assert result.mean_grid.shape == (8, 8)
    assert result.std_grid.shape == (8, 8)
    assert result.cv_grid.shape == (8, 8)
    assert result.spurious_mask.shape == (8, 8)
    assert result.n_samples == 3
    assert result.grid_h == 8
    assert result.grid_w == 8


def test_analyse_dataset_mean_in_unit_range() -> None:
    from miru.dataset_analytics import analyse_dataset

    grids = [np.random.default_rng(i).random((8, 8)).astype(np.float32) for i in range(4)]
    result = analyse_dataset(grids)
    assert float(result.mean_grid.min()) >= 0.0
    assert float(result.mean_grid.max()) <= 1.0


def test_analyse_dataset_identical_grids_std_zero() -> None:
    from miru.dataset_analytics import analyse_dataset

    grids = [_grid(fill=0.8) for _ in range(5)]
    result = analyse_dataset(grids)
    np.testing.assert_allclose(result.std_grid, 0.0, atol=1e-6)


def test_analyse_dataset_identical_high_grids_all_spurious() -> None:
    from miru.dataset_analytics import analyse_dataset

    grids = [_grid(fill=0.9) for _ in range(5)]
    result = analyse_dataset(grids, mean_threshold=0.5, cv_threshold=0.5)
    assert result.spurious_mask.all()
    assert len(result.spurious_cells) == 64


def test_analyse_dataset_empty_raises() -> None:
    from miru.dataset_analytics import analyse_dataset

    with pytest.raises(ValueError):
        analyse_dataset([])


# ---------------------------------------------------------------------------
# API integration — POST /analyze/batch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def _item(question: str = "What is here?") -> dict:
    return {"image_b64": _png_b64(), "question": question}


def test_analyze_batch_happy_path(api_client) -> None:
    payload = {
        "images": [_item("Where?"), _item("What color?"), _item("Is there a dog?")],
        "model_name": "mock",
        "method": "attention",
    }
    resp = api_client.post("/analyze/batch", json=payload)
    assert resp.status_code == 200, resp.text


def test_analyze_batch_response_shape(api_client) -> None:
    payload = {
        "images": [_item("a"), _item("b"), _item("c")],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    assert body["n_images"] == 3
    assert len(body["per_image"]) == 3
    assert len(body["mean_grid"]) > 0
    assert len(body["std_grid"]) > 0
    assert len(body["cv_grid"]) > 0
    assert body["latency_ms"] >= 0.0


def test_analyze_batch_per_image_indices(api_client) -> None:
    payload = {
        "images": [_item("q1"), _item("q2")],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    indices = [item["index"] for item in body["per_image"]]
    assert indices == [0, 1]


def test_analyze_batch_mean_grid_values_in_range(api_client) -> None:
    payload = {
        "images": [_item() for _ in range(4)],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    flat = [v for row in body["mean_grid"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_analyze_batch_spurious_cells_list_present(api_client) -> None:
    payload = {
        "images": [_item() for _ in range(3)],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    assert "spurious_cells" in body
    assert isinstance(body["spurious_cells"], list)


def test_analyze_batch_single_image_no_spurious(api_client) -> None:
    """Spurious detection requires ≥ 3 samples — single image returns empty list."""
    payload = {
        "images": [_item()],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    assert body["spurious_cells"] == []


def test_analyze_batch_model_name_echoed(api_client) -> None:
    payload = {
        "images": [_item()],
        "model_name": "mock",
        "method": "attention",
    }
    body = api_client.post("/analyze/batch", json=payload).json()
    assert body["model_name"] == "mock"
    assert body["method"] == "attention"


def test_analyze_batch_unknown_model_returns_400(api_client) -> None:
    payload = {
        "images": [_item()],
        "model_name": "nope_xyz",
        "method": "attention",
    }
    resp = api_client.post("/analyze/batch", json=payload)
    assert resp.status_code == 400
    assert "nope_xyz" in resp.json()["detail"]


def test_analyze_batch_bad_image_returns_400(api_client) -> None:
    payload = {
        "images": [{"image_b64": "!!!bad", "question": "test"}],
        "model_name": "mock",
        "method": "attention",
    }
    resp = api_client.post("/analyze/batch", json=payload)
    assert resp.status_code == 400


def test_analyze_batch_unknown_method_returns_400(api_client) -> None:
    payload = {
        "images": [_item()],
        "model_name": "mock",
        "method": "telepathy",
    }
    resp = api_client.post("/analyze/batch", json=payload)
    assert resp.status_code == 400


def test_analyze_batch_empty_images_returns_422(api_client) -> None:
    payload = {
        "images": [],
        "model_name": "mock",
        "method": "attention",
    }
    resp = api_client.post("/analyze/batch", json=payload)
    assert resp.status_code == 422


def test_analyze_batch_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert "mock" in resp.json()["backends"]
