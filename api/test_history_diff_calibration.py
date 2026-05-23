"""HTTP tests for /explain/history, /explain/calibration, /explain/diff."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from miru.explain_cache import (
    CACHE_ENABLED_ENV,
    CACHE_PATH_ENV,
    reset_cache,
)
from miru.recorder import reset_recorder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def record_dir(tmp_path: Path, monkeypatch):
    """Recorder writing to tmp + a per-test cache file."""
    monkeypatch.setenv("MIRU_RECORD", "1")
    monkeypatch.setenv("MIRU_RECORD_PATH", str(tmp_path / "traces"))
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "cache.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "1")
    reset_recorder()
    reset_cache()
    yield tmp_path / "traces"
    reset_recorder()
    reset_cache()


@pytest.fixture
def client(record_dir) -> TestClient:
    return TestClient(app)


def _png_b64(seed: int = 0, side: int = 16) -> str:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (side, side, 3)).astype(np.uint8)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate(
            [img, np.full((side, side, 1), 255, dtype=np.uint8)], axis=2,
        )
        return encode_png_b64(rgba)


def _post_explain(
    client: TestClient,
    *,
    seed: int = 0,
    method: str = "attention",
    fidelity: bool = False,
    question: str = "where?",
) -> str:
    """Hit /explain, return analysis_id."""
    qs = "?fidelity=true" if fidelity else ""
    resp = client.post(f"/explain{qs}", json={
        "image_b64": _png_b64(seed=seed),
        "model_name": "mock",
        "method": method,
        "question": question,
        "occlusion_grid": 4,
        "n_segments": 9,
        "n_samples": 16,
        "shap_grid": 4,
        "shap_samples": 8,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()["analysis_id"]


# ===========================================================================
# /explain/history
# ===========================================================================


def test_history_empty_when_no_records(client: TestClient) -> None:
    resp = client.get("/explain/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 50


def test_history_returns_recorded_explanations(client: TestClient) -> None:
    ids = [_post_explain(client, seed=i) for i in range(3)]
    resp = client.get("/explain/history")
    body = resp.json()
    assert body["total"] == 3
    returned = {item["analysis_id"] for item in body["items"]}
    assert returned == set(ids)
    # All items have the expected shape.
    for item in body["items"]:
        assert "ts" in item and item["ts"]
        assert "method" in item
        assert "backend" in item
        assert item["fidelity_score"] is None  # we didn't pass fidelity=true


def test_history_filter_by_method(client: TestClient) -> None:
    _post_explain(client, seed=0, method="attention")
    _post_explain(client, seed=1, method="gradcam")
    _post_explain(client, seed=2, method="lime")
    resp = client.get("/explain/history?method=gradcam")
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["method"] == "gradcam"


def test_history_pagination(client: TestClient) -> None:
    for i in range(5):
        _post_explain(client, seed=i)
    resp = client.get("/explain/history?limit=2&offset=1")
    body = resp.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_history_filter_by_model(client: TestClient) -> None:
    _post_explain(client, seed=0)
    resp = client.get("/explain/history?model=mock")
    assert resp.json()["total"] == 1
    resp = client.get("/explain/history?model=nonexistent")
    assert resp.json()["total"] == 0


def test_history_filter_by_min_confidence(client: TestClient) -> None:
    _post_explain(client, seed=0)
    # Mock confidence is deterministic per question and < 1.0, so a
    # threshold of exactly 1.0 strips everything.
    resp = client.get("/explain/history?min_confidence=1.0")
    assert resp.json()["total"] == 0
    resp = client.get("/explain/history?min_confidence=0.0")
    assert resp.json()["total"] == 1


def test_history_rejects_oversized_limit(client: TestClient) -> None:
    resp = client.get("/explain/history?limit=201")
    assert resp.status_code == 422


# ===========================================================================
# /explain/calibration
# ===========================================================================


def test_calibration_empty_population_returns_zero(client: TestClient) -> None:
    resp = client.get("/explain/calibration")
    body = resp.json()
    assert resp.status_code == 200
    assert body["n"] == 0
    assert body["ece"] == 0.0
    assert len(body["bins"]) == 10


def test_calibration_skips_records_without_fidelity(client: TestClient) -> None:
    _post_explain(client, seed=0, fidelity=False)
    resp = client.get("/explain/calibration")
    body = resp.json()
    assert body["n"] == 0


def test_calibration_counts_records_with_fidelity(client: TestClient) -> None:
    for i in range(3):
        _post_explain(client, seed=i, fidelity=True, question=f"q{i}")
    resp = client.get("/explain/calibration?n_bins=5")
    body = resp.json()
    assert body["n"] == 3
    assert body["n_bins"] == 5
    assert len(body["bins"]) == 5
    # ECE in valid range
    assert 0.0 <= body["ece"] <= 1.0


def test_calibration_filter_by_method(client: TestClient) -> None:
    _post_explain(client, seed=0, method="attention", fidelity=True)
    _post_explain(client, seed=1, method="gradcam",  fidelity=True)
    resp = client.get("/explain/calibration?method=gradcam")
    body = resp.json()
    assert body["filter_method"] == "gradcam"
    assert body["n"] == 1


def test_calibration_rejects_invalid_n_bins(client: TestClient) -> None:
    resp = client.get("/explain/calibration?n_bins=1")
    assert resp.status_code == 422
    resp = client.get("/explain/calibration?n_bins=51")
    assert resp.status_code == 422


# ===========================================================================
# /explain/diff
# ===========================================================================


def test_diff_two_distinct_analyses(client: TestClient) -> None:
    id_a = _post_explain(client, seed=1, method="attention", question="qA")
    id_b = _post_explain(client, seed=2, method="gradcam",   question="qB")
    resp = client.post("/explain/diff", json={
        "analysis_id_a": id_a, "analysis_id_b": id_b, "top_n": 5,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id_a"] == id_a
    assert body["analysis_id_b"] == id_b
    assert body["method_a"] == "attention"
    assert body["method_b"] == "gradcam"
    assert -1.0 <= body["cosine_similarity"] <= 1.0
    assert body["l2_distance"] >= 0.0
    assert len(body["top_changed"]) <= 5
    assert body["summary"]


def test_diff_same_analysis_rejected(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    resp = client.post("/explain/diff", json={
        "analysis_id_a": id_a, "analysis_id_b": id_a,
    })
    assert resp.status_code == 400


def test_diff_unknown_id_returns_404(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    resp = client.post("/explain/diff", json={
        "analysis_id_a": id_a, "analysis_id_b": "nonexistent-id-1234",
    })
    assert resp.status_code == 404
    assert "nonexistent-id-1234" in resp.json()["detail"]


def test_diff_top_n_bounded(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    id_b = _post_explain(client, seed=1)
    resp = client.post("/explain/diff", json={
        "analysis_id_a": id_a, "analysis_id_b": id_b, "top_n": 0,
    })
    assert resp.status_code == 422


def test_diff_delta_grid_shape(client: TestClient) -> None:
    """The delta_grid round-trips through JSON intact."""
    id_a = _post_explain(client, seed=10)
    id_b = _post_explain(client, seed=20)
    body = client.post("/explain/diff", json={
        "analysis_id_a": id_a, "analysis_id_b": id_b,
    }).json()
    grid = body["delta_grid"]
    # Mock backend returns 16x16 attention grids.
    assert len(grid) == 16
    assert all(len(row) == 16 for row in grid)
