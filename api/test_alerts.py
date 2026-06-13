"""HTTP tests for alert rules and history endpoints.

Phase 25: POST/GET/DELETE /explain/alerts/rules, GET /explain/alerts/history.

Run from the repo root:

    python -m pytest api/test_alerts.py -v

The module fixture injects a fresh in-memory AlertStore directly into the
singleton slot so the store is isolated regardless of what other test modules
have done to the singleton before this module runs.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import miru.alerts as _alerts_mod
from miru.alerts import AlertStore
from api.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _fresh_alert_store():
    """Inject a fresh in-memory AlertStore directly into the singleton slot."""
    fresh = AlertStore(":memory:")
    _alerts_mod._store = fresh
    yield fresh
    _alerts_mod._store = None


@pytest.fixture(scope="module")
def client(_fresh_alert_store) -> TestClient:  # noqa: ARG001
    return TestClient(app)


@pytest.fixture(scope="module")
def png_b64() -> str:
    h = w = 16
    img = np.full((h, w, 3), 64, dtype=np.uint8)
    img[4:8, 4:8] = 200
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return encode_png_b64(rgba)


_WEBHOOK = "http://example.com/hook"


def _create_rule(client: TestClient, name: str, **overrides) -> object:
    body = {
        "name": name,
        "field": "confidence",
        "op": "<",
        "threshold": 0.4,
        "webhook_url": _WEBHOOK,
    }
    body.update(overrides)
    return client.post("/explain/alerts/rules", json=body)


# ---------------------------------------------------------------------------
# POST /explain/alerts/rules
# ---------------------------------------------------------------------------


def test_create_rule_returns_201(client: TestClient) -> None:
    resp = _create_rule(client, "test-create-201")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["rule_id"].startswith("rule_")
    assert body["name"] == "test-create-201"
    assert body["field"] == "confidence"
    assert body["op"] == "<"
    assert body["threshold"] == pytest.approx(0.4)
    assert body["enabled"] is True


def test_create_rule_unknown_field_rejected(client: TestClient) -> None:
    resp = _create_rule(client, "bad-field", field="latency_ms")
    assert resp.status_code == 400
    assert "field" in resp.json()["detail"]


def test_create_rule_unknown_op_rejected(client: TestClient) -> None:
    resp = _create_rule(client, "bad-op", op="!=")
    assert resp.status_code == 400
    assert "op" in resp.json()["detail"]


def test_create_rule_bad_webhook_rejected(client: TestClient) -> None:
    resp = _create_rule(client, "bad-webhook", webhook_url="ftp://example.com")
    assert resp.status_code == 400
    assert "http" in resp.json()["detail"].lower()


def test_create_rule_fidelity_field(client: TestClient) -> None:
    resp = _create_rule(
        client,
        "fid-rule",
        field="fidelity_score",
        op="<",
        threshold=0.5,
    )
    assert resp.status_code == 201
    assert resp.json()["field"] == "fidelity_score"


# ---------------------------------------------------------------------------
# GET /explain/alerts/rules
# ---------------------------------------------------------------------------


def test_list_rules_returns_200(client: TestClient) -> None:
    resp = client.get("/explain/alerts/rules")
    assert resp.status_code == 200
    body = resp.json()
    assert "rules" in body
    assert "total" in body
    assert body["total"] == len(body["rules"])


def test_list_rules_enabled_only_filter(client: TestClient) -> None:
    resp_all = client.get("/explain/alerts/rules")
    resp_enabled = client.get("/explain/alerts/rules?enabled_only=true")
    assert resp_all.status_code == 200
    assert resp_enabled.status_code == 200
    assert resp_enabled.json()["total"] <= resp_all.json()["total"]


# ---------------------------------------------------------------------------
# DELETE /explain/alerts/rules/{rule_id}
# ---------------------------------------------------------------------------


def test_delete_rule_ok(client: TestClient) -> None:
    create_resp = _create_rule(client, "to-delete")
    rule_id = create_resp.json()["rule_id"]

    del_resp = client.delete(f"/explain/alerts/rules/{rule_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["deleted"] is True
    assert body["rule_id"] == rule_id


def test_delete_rule_not_found(client: TestClient) -> None:
    resp = client.delete("/explain/alerts/rules/rule_doesnotexist000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /explain/alerts/history
# ---------------------------------------------------------------------------


def test_alert_history_returns_200(client: TestClient) -> None:
    resp = client.get("/explain/alerts/history")
    assert resp.status_code == 200
    body = resp.json()
    assert "alerts" in body
    assert "total" in body
    assert "limit" in body


def test_alert_history_limit_param(client: TestClient) -> None:
    resp = client.get("/explain/alerts/history?limit=5")
    assert resp.status_code == 200
    assert resp.json()["limit"] == 5


def test_alert_history_limit_out_of_range(client: TestClient) -> None:
    resp = client.get("/explain/alerts/history?limit=999")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration: /explain fires alert rules
# ---------------------------------------------------------------------------


def test_explain_fires_alert_rule(client: TestClient, png_b64: str) -> None:
    """A confidence<1.0 rule fires on every mock /explain call (mock conf < 1)."""
    _create_rule(client, "always-fires", op="<", threshold=1.0)

    before = client.get("/explain/alerts/history").json()["total"]
    client.post(
        "/explain",
        json={
            "image_b64": png_b64,
            "model_name": "mock",
            "method": "attention",
        },
    )
    after = client.get("/explain/alerts/history").json()["total"]
    assert after > before, "at least one alert should have fired"


def test_explain_no_alert_when_condition_false(
    client: TestClient, png_b64: str
) -> None:
    """A confidence>0.9999 rule should never fire on the mock backend."""
    _create_rule(client, "never-fires", op=">", threshold=0.9999)

    client.post(
        "/explain",
        json={
            "image_b64": png_b64,
            "model_name": "mock",
            "method": "attention",
        },
    )
    # Count of *new* alerts attributable to the never-fires rule should be 0.
    after_alerts = client.get("/explain/alerts/history").json()["alerts"]
    never_fired = [a for a in after_alerts if a["rule_name"] == "never-fires"]
    assert never_fired == []
