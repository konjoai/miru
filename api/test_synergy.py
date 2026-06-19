"""HTTP tests for the ``synergy`` flag on ``POST /explain``.

The mock backend is image-independent, so its synergy_score is exactly
0.0 and ``low_synergy`` is always True — a deterministic contract to
assert against.  The synergy *math* (positive-synergy path) is covered by
``tests/test_synergy.py`` with synthetic backends.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ``png_b64`` is provided by api/conftest.py.


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _explain(client: TestClient, image_b64: str, **params) -> object:
    body = {"image_b64": image_b64, "model_name": "mock"}
    return client.post("/explain", params=params, json=body)


def test_no_synergy_flag_omits_block(client, png_b64) -> None:
    resp = _explain(client, png_b64)
    assert resp.status_code == 200
    assert resp.json()["synergy"] is None


def test_synergy_flag_attaches_block(client, png_b64) -> None:
    resp = _explain(client, png_b64, synergy=True)
    assert resp.status_code == 200
    block = resp.json()["synergy"]
    assert block is not None


def test_synergy_block_fields(client, png_b64) -> None:
    block = _explain(client, png_b64, synergy=True).json()["synergy"]
    expected = {
        "synergy_score",
        "interaction",
        "f_both",
        "f_language_only",
        "f_vision_only",
        "f_neither",
        "k_pct",
        "low_synergy",
    }
    assert expected <= set(block)


def test_synergy_score_in_unit_range(client, png_b64) -> None:
    block = _explain(client, png_b64, synergy=True).json()["synergy"]
    assert 0.0 <= block["synergy_score"] <= 1.0


def test_mock_reports_zero_synergy(client, png_b64) -> None:
    """Image-independent mock → exactly zero synergy, flagged low."""
    block = _explain(client, png_b64, synergy=True).json()["synergy"]
    assert block["synergy_score"] == pytest.approx(0.0, abs=1e-9)
    assert block["low_synergy"] is True


def test_f_both_matches_confidence(client, png_b64) -> None:
    body = _explain(client, png_b64, synergy=True).json()
    assert body["synergy"]["f_both"] == pytest.approx(body["confidence"])


def test_synergy_and_fidelity_coexist(client, png_b64) -> None:
    body = _explain(client, png_b64, synergy=True, fidelity=True).json()
    assert body["synergy"] is not None
    assert body["fidelity"] is not None


def test_synergy_cache_key_isolated(client, png_b64) -> None:
    """A synergy=true call must not be served a synergy-less cached body."""
    plain = _explain(client, png_b64, synergy=False, use_cache=True).json()
    assert plain["synergy"] is None
    withsyn = _explain(client, png_b64, synergy=True, use_cache=True).json()
    assert withsyn["synergy"] is not None


def test_synergy_with_lime_method(client, png_b64) -> None:
    body = {"image_b64": png_b64, "model_name": "mock", "method": "lime"}
    resp = client.post("/explain", params={"synergy": True}, json=body)
    assert resp.status_code == 200
    assert resp.json()["synergy"] is not None


def test_batch_synergy_propagates(client, png_b64) -> None:
    body = {
        "items": [
            {"image_b64": png_b64, "model_name": "mock"},
            {"image_b64": png_b64, "model_name": "mock", "method": "lime"},
        ],
        "synergy": True,
        "record": False,
    }
    resp = client.post("/explain/batch", json=body)
    assert resp.status_code == 200
    for item in resp.json()["items"]:
        assert item["response"]["synergy"] is not None
