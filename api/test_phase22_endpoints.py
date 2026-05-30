"""HTTP tests for /explain/models/compare, /explain/consensus/by_ids, /explain/search."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import MAX_COMPARE_MODELS, app
from miru.explain_cache import CACHE_ENABLED_ENV, CACHE_PATH_ENV, reset_cache
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
# /explain/models/compare
# ===========================================================================


def test_models_compare_empty_store(client: TestClient) -> None:
    resp = client.get("/explain/models/compare?models=mock")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == ["mock"]
    assert body["stats"]["mock"]["n_records"] == 0
    assert body["winner_by_confidence"] is None
    assert body["winner_by_fidelity"] is None
    assert body["winner_by_ece"] is None


def test_models_compare_with_data(client: TestClient) -> None:
    for i in range(3):
        _post_explain(client, seed=i, question=f"q{i}")
    resp = client.get("/explain/models/compare?models=mock&limit=10")
    body = resp.json()
    assert body["stats"]["mock"]["n_records"] == 3
    assert body["stats"]["mock"]["mean_confidence"] is not None
    assert 0.0 <= body["stats"]["mock"]["mean_confidence"] <= 1.0
    assert body["winner_by_confidence"] == "mock"
    # No fidelity records yet.
    assert body["stats"]["mock"]["n_with_fidelity"] == 0
    assert body["winner_by_fidelity"] is None


def test_models_compare_method_filter(client: TestClient) -> None:
    _post_explain(client, seed=0, method="attention")
    _post_explain(client, seed=1, method="gradcam")
    resp = client.get("/explain/models/compare?models=mock&method=gradcam")
    body = resp.json()
    assert body["stats"]["mock"]["n_records"] == 1
    assert body["filter_method"] == "gradcam"


def test_models_compare_method_distribution(client: TestClient) -> None:
    _post_explain(client, seed=0, method="attention")
    _post_explain(client, seed=1, method="attention")
    _post_explain(client, seed=2, method="gradcam")
    resp = client.get("/explain/models/compare?models=mock")
    body = resp.json()
    dist = body["stats"]["mock"]["method_distribution"]
    assert dist == {"attention": 2, "gradcam": 1}


def test_models_compare_rejects_empty(client: TestClient) -> None:
    resp = client.get("/explain/models/compare?models=")
    assert resp.status_code == 400


def test_models_compare_rejects_oversized_list(client: TestClient) -> None:
    too_many = ",".join(f"m{i}" for i in range(MAX_COMPARE_MODELS + 1))
    resp = client.get(f"/explain/models/compare?models={too_many}")
    assert resp.status_code == 400


def test_models_compare_rejects_duplicates(client: TestClient) -> None:
    resp = client.get("/explain/models/compare?models=mock,mock")
    assert resp.status_code == 400


# ===========================================================================
# /explain/consensus/by_ids
# ===========================================================================


def test_consensus_by_ids_combines_records(client: TestClient) -> None:
    id_a = _post_explain(client, seed=1)
    id_b = _post_explain(client, seed=2)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a, id_b],
        "weighting": "uniform",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_records"] == 2
    assert body["weighting"] == "uniform"
    assert len(body["per_record"]) == 2
    assert len(body["top_regions"]) <= 5
    # Each contributor agrees with the consensus (cosine in [-1, 1]).
    for contrib in body["per_record"]:
        assert -1.0 <= contrib["agreement_score"] <= 1.0


def test_consensus_by_ids_rejects_too_few(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a],
    })
    assert resp.status_code == 422  # min_length=2


def test_consensus_by_ids_rejects_duplicates(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a, id_a],
    })
    assert resp.status_code == 400


def test_consensus_by_ids_404_on_missing(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a, "definitely-not-real"],
    })
    assert resp.status_code == 404
    assert "definitely-not-real" in resp.json()["detail"]


def test_consensus_by_ids_fidelity_weighting(client: TestClient) -> None:
    id_a = _post_explain(client, seed=1, fidelity=True)
    id_b = _post_explain(client, seed=2, fidelity=True)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a, id_b],
        "weighting": "fidelity",
    })
    body = resp.json()
    assert body["weighting"] == "fidelity"
    # Per-record weight is populated; mock fidelity_score is real.
    for contrib in body["per_record"]:
        assert contrib["weight"] >= 0.0


def test_consensus_by_ids_rejects_invalid_weighting(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    id_b = _post_explain(client, seed=1)
    resp = client.post("/explain/consensus/by_ids", json={
        "analysis_ids": [id_a, id_b],
        "weighting": "bogus",
    })
    assert resp.status_code == 400


# ===========================================================================
# /explain/search
# ===========================================================================


def test_search_by_grid_returns_matches(client: TestClient) -> None:
    # Populate with a few records first.
    for i in range(3):
        _post_explain(client, seed=i, question=f"q{i}")

    # Build a query grid; mock returns 16x16 attention.
    query = [[0.5 for _ in range(16)] for _ in range(16)]
    resp = client.post("/explain/search", json={
        "query_grid": query, "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_k"] == 5
    assert body["n_scanned"] == 3
    assert body["n_candidates"] == 3
    assert len(body["matches"]) <= 5
    for m in body["matches"]:
        assert -1.0 <= m["similarity"] <= 1.0


def test_search_by_analysis_id_excludes_self(client: TestClient) -> None:
    id_a = _post_explain(client, seed=0)
    _post_explain(client, seed=1)
    resp = client.post("/explain/search", json={
        "query_analysis_id": id_a, "top_k": 5,
    })
    body = resp.json()
    ids = {m["analysis_id"] for m in body["matches"]}
    assert id_a not in ids
    assert body["query_analysis_id"] == id_a


def test_search_404_on_missing_query_id(client: TestClient) -> None:
    resp = client.post("/explain/search", json={
        "query_analysis_id": "not-a-real-id",
    })
    assert resp.status_code == 404


def test_search_rejects_both_query_modes(client: TestClient) -> None:
    resp = client.post("/explain/search", json={
        "query_grid": [[0.5]], "query_analysis_id": "x",
    })
    assert resp.status_code == 400


def test_search_rejects_neither_query_mode(client: TestClient) -> None:
    resp = client.post("/explain/search", json={})
    assert resp.status_code == 400


def test_search_filter_by_method(client: TestClient) -> None:
    _post_explain(client, seed=0, method="attention")
    _post_explain(client, seed=1, method="gradcam")
    query = [[0.5 for _ in range(16)] for _ in range(16)]
    resp = client.post("/explain/search", json={
        "query_grid": query, "method": "gradcam",
    })
    body = resp.json()
    assert all(m["method"] == "gradcam" for m in body["matches"])


def test_search_min_similarity_drops_low(client: TestClient) -> None:
    for i in range(3):
        _post_explain(client, seed=i)
    query = [[0.5 for _ in range(16)] for _ in range(16)]
    # Set a threshold so high that nothing passes.
    resp = client.post("/explain/search", json={
        "query_grid": query, "min_similarity": 0.9999999,
    })
    body = resp.json()
    # Some or none may pass; the body shape is still correct.
    for m in body["matches"]:
        assert m["similarity"] >= 0.9999999
