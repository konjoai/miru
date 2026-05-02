"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_version(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert isinstance(data["version"], str)
    assert data["version"] == "0.4.0"


def test_health_backends(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    backends = resp.json()["backends"]
    assert isinstance(backends, list)
    assert "mock" in backends


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


def test_analyze_success(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "What is shown?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200


def test_analyze_response_structure(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "Describe the scene.", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    data = resp.json()
    assert "answer" in data
    assert "steps" in data
    assert "attention_map" in data
    assert "backend" in data
    assert "latency_ms" in data


def test_analyze_default_backend(client: TestClient, mock_image_b64: str) -> None:
    """Omitting backend should default to mock."""
    payload = {"image_b64": mock_image_b64, "question": "Any objects present?"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    assert resp.json()["backend"] == "mock"


def test_analyze_attention_map_dimensions(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "What color dominates?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    attn = resp.json()["attention_map"]
    assert attn["width"] == 16
    assert attn["height"] == 16


def test_analyze_latency_positive(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "Is there movement?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert resp.json()["latency_ms"] > 0.0


def test_analyze_answer_nonempty(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "Describe the image.", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert len(resp.json()["answer"]) > 0


def test_analyze_steps_nonempty(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "What is the focus?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert len(resp.json()["steps"]) >= 1


def test_analyze_bad_image_graceful(client: TestClient) -> None:
    """Invalid base64 must not raise a 5xx — endpoint handles it gracefully."""
    payload = {"image_b64": "not-valid-base64!!!", "question": "Test?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    # The service must return a usable response, not a 500.
    assert resp.status_code == 200


def test_analyze_unknown_backend_falls_back(client: TestClient, mock_image_b64: str) -> None:
    """An unknown backend name should fall back to mock without crashing."""
    payload = {
        "image_b64": mock_image_b64,
        "question": "Test fallback.",
        "backend": "nonexistent_backend",
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    assert resp.json()["backend"] == "mock"
