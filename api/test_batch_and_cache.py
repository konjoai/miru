"""HTTP-level tests for /explain/batch and the cache surface."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import MAX_BATCH_ITEMS, app
from miru.explain_cache import CACHE_ENABLED_ENV, CACHE_PATH_ENV, reset_cache


# ---------------------------------------------------------------------------
# Fixtures — fresh cache per test
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "test_cache.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "1")
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client(isolated_cache) -> TestClient:
    return TestClient(app)


def _png_b64(seed: int = 0, side: int = 16) -> str:
    rng = np.random.default_rng(seed)
    img = (rng.integers(0, 256, (side, side, 3))).astype(np.uint8)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate(
            [img, np.full((side, side, 1), 255, dtype=np.uint8)], axis=2
        )
        return encode_png_b64(rgba)


@pytest.fixture
def png() -> str:
    return _png_b64(seed=1)


# ---------------------------------------------------------------------------
# /explain — cache hit/miss observable via X-Miru-Cache header
# ---------------------------------------------------------------------------


def test_explain_first_call_is_miss(client: TestClient, png: str) -> None:
    resp = client.post("/explain", json={
        "image_b64": png, "model_name": "mock", "method": "attention",
    })
    assert resp.status_code == 200
    assert resp.headers.get("X-Miru-Cache") == "miss"


def test_explain_second_call_is_hit(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock", "method": "attention"}
    a = client.post("/explain", json=body)
    b = client.post("/explain", json=body)
    assert a.status_code == 200 and b.status_code == 200
    assert a.headers.get("X-Miru-Cache") == "miss"
    assert b.headers.get("X-Miru-Cache") == "hit"
    # Response payloads are identical — same analysis_id served back.
    assert a.json()["analysis_id"] == b.json()["analysis_id"]
    assert a.json()["attention_grid"] == b.json()["attention_grid"]


def test_cache_partitions_by_method(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock"}
    r_att = client.post("/explain", json={**body, "method": "attention"})
    r_grad = client.post("/explain", json={**body, "method": "gradcam",
                                          "occlusion_grid": 4})
    assert r_att.headers["X-Miru-Cache"] == "miss"
    assert r_grad.headers["X-Miru-Cache"] == "miss"
    # Hit each method again
    assert client.post("/explain", json={**body, "method": "attention"}) \
        .headers["X-Miru-Cache"] == "hit"
    assert client.post("/explain", json={**body, "method": "gradcam",
                                         "occlusion_grid": 4}) \
        .headers["X-Miru-Cache"] == "hit"


def test_cache_partitions_by_param(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock", "method": "attention"}
    client.post("/explain", json={**body, "alpha": 0.4})
    # Different alpha — different cache entry, so this is a miss.
    resp = client.post("/explain", json={**body, "alpha": 0.6})
    assert resp.headers["X-Miru-Cache"] == "miss"


def test_explain_use_cache_false_bypasses(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock", "method": "attention"}
    client.post("/explain", json=body)  # populate
    resp = client.post("/explain?use_cache=false", json=body)
    assert resp.headers["X-Miru-Cache"] == "bypass"


def test_cache_disabled_via_env(tmp_path: Path, monkeypatch, png: str) -> None:
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "x.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "0")
    reset_cache()
    try:
        c = TestClient(app)
        body = {"image_b64": png, "model_name": "mock", "method": "attention"}
        a = c.post("/explain", json=body)
        b = c.post("/explain", json=body)
        # No caching → both calls bypass via the use_cache check
        assert a.headers["X-Miru-Cache"] == "bypass"
        assert b.headers["X-Miru-Cache"] == "bypass"
    finally:
        reset_cache()


# ---------------------------------------------------------------------------
# /explain/cache_stats + /explain/cache_clear
# ---------------------------------------------------------------------------


def test_cache_stats_reflects_traffic(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock", "method": "attention"}
    client.post("/explain", json=body)       # miss
    client.post("/explain", json=body)       # hit
    client.post("/explain", json=body)       # hit
    resp = client.get("/explain/cache_stats")
    assert resp.status_code == 200
    s = resp.json()
    assert s["enabled"] is True
    assert s["total_entries"] == 1
    assert s["total_hits"] == 2
    assert s["total_misses"] == 1
    assert abs(s["hit_rate"] - 2 / 3) < 1e-9
    assert s["per_method"] == {"attention": 1}
    assert s["size_bytes"] > 0


def test_cache_stats_when_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "x.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "0")
    reset_cache()
    try:
        c = TestClient(app)
        resp = c.get("/explain/cache_stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["total_entries"] == 0
    finally:
        reset_cache()


def test_cache_clear_drops_entries(client: TestClient, png: str) -> None:
    body = {"image_b64": png, "model_name": "mock", "method": "attention"}
    client.post("/explain", json=body)
    client.post("/explain", json=body)
    assert client.get("/explain/cache_stats").json()["total_entries"] == 1
    resp = client.post("/explain/cache_clear")
    assert resp.status_code == 200
    assert resp.json()["cleared"] == 1
    s = client.get("/explain/cache_stats").json()
    assert s["total_entries"] == 0
    assert s["total_hits"] == 0
    assert s["total_misses"] == 0


# ---------------------------------------------------------------------------
# /explain/batch — happy path + aggregate + cache + error handling
# ---------------------------------------------------------------------------


def test_batch_three_items_all_succeed(client: TestClient) -> None:
    items = [
        {"image_b64": _png_b64(seed=s), "model_name": "mock", "method": "attention"}
        for s in range(3)
    ]
    resp = client.post("/explain/batch", json={"items": items})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 3
    assert all(it["success"] is True for it in body["items"])
    # First batch — every item is a cache miss.
    assert all(it["cached"] is False for it in body["items"])
    agg = body["aggregate"]
    assert agg["total"] == 3
    assert agg["success_count"] == 3
    assert agg["failure_count"] == 0
    assert agg["cache_hits"] == 0
    assert agg["cache_misses"] == 3
    assert agg["mean_confidence"] is not None
    assert agg["mean_fidelity"] is None  # fidelity disabled
    assert agg["total_latency_ms"] > 0


def test_batch_warm_cache_reports_hits(client: TestClient) -> None:
    items = [
        {"image_b64": _png_b64(seed=s), "model_name": "mock", "method": "attention"}
        for s in range(3)
    ]
    client.post("/explain/batch", json={"items": items})       # populate
    resp = client.post("/explain/batch", json={"items": items}) # warm
    body = resp.json()
    assert body["aggregate"]["cache_hits"] == 3
    assert body["aggregate"]["cache_misses"] == 0
    assert all(it["cached"] for it in body["items"])


def test_batch_preserves_order(client: TestClient) -> None:
    items = [
        {"image_b64": _png_b64(seed=s), "model_name": "mock", "method": "attention",
         "question": f"q{s}"}
        for s in range(4)
    ]
    resp = client.post("/explain/batch", json={"items": items})
    body = resp.json()
    indices = [it["index"] for it in body["items"]]
    assert indices == [0, 1, 2, 3]


def test_batch_fidelity_flag_propagates(client: TestClient) -> None:
    items = [
        {"image_b64": _png_b64(seed=s), "model_name": "mock", "method": "attention"}
        for s in range(2)
    ]
    resp = client.post("/explain/batch", json={"items": items, "fidelity": True})
    body = resp.json()
    assert all(it["response"]["fidelity"] is not None for it in body["items"])
    assert body["aggregate"]["mean_fidelity"] is not None


def test_batch_mixed_methods(client: TestClient, png: str) -> None:
    items = [
        {"image_b64": png, "model_name": "mock", "method": "attention"},
        {"image_b64": png, "model_name": "mock", "method": "gradcam",
         "occlusion_grid": 4},
        {"image_b64": png, "model_name": "mock", "method": "lime",
         "n_segments": 9, "n_samples": 16},
    ]
    resp = client.post("/explain/batch", json={"items": items})
    body = resp.json()
    methods = [it["response"]["method"] for it in body["items"]]
    assert methods == ["attention", "gradcam", "lime"]


def test_batch_one_bad_item_does_not_fail_others(client: TestClient, png: str) -> None:
    items = [
        {"image_b64": png, "model_name": "mock", "method": "attention"},
        # Empty image → boundary validation either rejects with 400
        # (caught and turned into an item failure) or fails decode.
        {"image_b64": "!!!", "model_name": "mock", "method": "attention"},
        {"image_b64": png, "model_name": "mock", "method": "attention"},
    ]
    resp = client.post("/explain/batch", json={"items": items})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["success"] is True
    assert body["items"][1]["success"] is False
    assert body["items"][1]["error"]
    assert body["items"][2]["success"] is True
    assert body["aggregate"]["success_count"] == 2
    assert body["aggregate"]["failure_count"] == 1


def test_batch_stop_on_error_aborts_remainder(client: TestClient, png: str) -> None:
    items = [
        {"image_b64": png, "model_name": "mock", "method": "attention"},
        {"image_b64": "!!!", "model_name": "mock", "method": "attention"},
        {"image_b64": png, "model_name": "mock", "method": "attention"},
    ]
    resp = client.post("/explain/batch",
                       json={"items": items, "stop_on_error": True})
    body = resp.json()
    assert body["items"][0]["success"] is True
    assert body["items"][1]["success"] is False
    # Item 2 is reported but marked skipped.
    assert body["items"][2]["success"] is False
    assert "skipped" in body["items"][2]["error"]


def test_batch_empty_items_rejected(client: TestClient) -> None:
    resp = client.post("/explain/batch", json={"items": []})
    assert resp.status_code == 422


def test_batch_oversized_rejected(client: TestClient, png: str) -> None:
    one = {"image_b64": png, "model_name": "mock", "method": "attention"}
    items = [one] * (MAX_BATCH_ITEMS + 1)
    resp = client.post("/explain/batch", json={"items": items})
    assert resp.status_code == 422


def test_batch_single_item(client: TestClient, png: str) -> None:
    resp = client.post("/explain/batch", json={
        "items": [{"image_b64": png, "model_name": "mock", "method": "attention"}],
    })
    body = resp.json()
    assert body["aggregate"]["total"] == 1
    assert body["aggregate"]["success_count"] == 1
