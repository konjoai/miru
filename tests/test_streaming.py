"""Tests for /analyze/stream SSE endpoint and the stream_analyze generator."""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miru.models.base import VLMBackend, VLMOutput, VLMStreamChunk


@pytest.fixture(autouse=True)
def _ensure_default_registry():
    """test_registry tears down the global registry; refill it for every streaming test."""
    from miru.models import registry as reg

    reg.register_defaults()
    yield


def _parse_sse(stream_bytes: bytes) -> list[tuple[str, dict]]:
    """Parse an SSE byte stream into a list of (event, data-dict) tuples.

    Comments (``: keepalive``) are ignored.  Frames lacking an ``event:``
    line use the SSE default of ``message``.
    """
    text = stream_bytes.decode("utf-8")
    events: list[tuple[str, dict]] = []
    for raw_frame in text.split("\n\n"):
        frame = raw_frame.strip("\n")
        if not frame:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith(":"):  # comment / keepalive
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            events.append((event_name, payload))
    return events


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_stream_endpoint_emits_steps(client: TestClient, mock_image_b64: str) -> None:
    payload = {"image_b64": mock_image_b64, "question": "What's here?", "backend": "mock"}
    resp = client.post("/analyze/stream", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.content)
    kinds = [e[0] for e in events]
    assert kinds.count("step") >= 1
    assert "trace" in kinds
    assert kinds[-1] == "done"


def test_stream_step_event_shape(client: TestClient, mock_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "Q", "backend": "mock"},
    )
    events = _parse_sse(resp.content)
    step_events = [data for kind, data in events if kind == "step"]
    assert step_events, "expected at least one step event"
    for s in step_events:
        assert set(s.keys()) == {"step", "description", "confidence"}
        assert isinstance(s["step"], int)
        assert isinstance(s["description"], str) and s["description"]
        assert 0.0 <= s["confidence"] <= 1.0


def test_stream_trace_matches_analyze(client: TestClient, mock_image_b64: str) -> None:
    """The trailing ``trace`` event payload must be schema-equivalent to /analyze."""
    body = {"image_b64": mock_image_b64, "question": "match", "backend": "mock"}
    sync_resp = client.post("/analyze", json=body).json()
    stream_resp = client.post("/analyze/stream", json=body)
    events = _parse_sse(stream_resp.content)
    trace_events = [data for kind, data in events if kind == "trace"]
    assert len(trace_events) == 1
    streamed = trace_events[0]

    # Latency differs between calls — compare everything else.
    assert streamed["answer"] == sync_resp["answer"]
    assert streamed["backend"] == sync_resp["backend"]
    assert streamed["attention_map"] == sync_resp["attention_map"]
    assert streamed["steps"] == sync_resp["steps"]


@pytest.fixture
def png_image_b64() -> str:
    from miru.visualization.overlay import encode_png_b64

    arr = np.full((4, 4, 4), 255, dtype=np.uint8)
    return encode_png_b64(arr)


def test_stream_overlay_query_param(client: TestClient, png_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream?overlay=true",
        json={"image_b64": png_image_b64, "question": "with overlay", "backend": "mock"},
    )
    events = _parse_sse(resp.content)
    trace = next(data for kind, data in events if kind == "trace")
    assert isinstance(trace["overlay_b64"], str)
    assert len(trace["overlay_b64"]) > 0


def test_stream_no_overlay_default_null(client: TestClient, mock_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "no overlay", "backend": "mock"},
    )
    events = _parse_sse(resp.content)
    trace = next(data for kind, data in events if kind == "trace")
    assert trace["overlay_b64"] is None


def test_stream_unknown_backend_falls_back(client: TestClient, mock_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "fallback", "backend": "no_such"},
    )
    events = _parse_sse(resp.content)
    trace = next(data for kind, data in events if kind == "trace")
    assert trace["backend"] == "mock"


def test_stream_done_is_last_event(client: TestClient, mock_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "ordering", "backend": "mock"},
    )
    events = _parse_sse(resp.content)
    assert events[-1] == ("done", {})


def test_stream_step_indices_are_sequential(client: TestClient, mock_image_b64: str) -> None:
    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "ordering", "backend": "mock"},
    )
    events = _parse_sse(resp.content)
    step_indices = [data["step"] for kind, data in events if kind == "step"]
    assert step_indices == list(range(1, len(step_indices) + 1))


# ---------------------------------------------------------------------------
# Backend stream_infer default behaviour
# ---------------------------------------------------------------------------


def test_default_stream_infer_replays_steps_then_final() -> None:
    from miru.models.mock import MockVLMBackend

    backend = MockVLMBackend()
    chunks = list(backend.stream_infer(np.zeros((2, 2, 3), dtype=np.float32), "hi"))
    step_chunks = [c for c in chunks if c.kind == "step"]
    final_chunks = [c for c in chunks if c.kind == "final"]
    assert len(step_chunks) >= 1
    assert len(final_chunks) == 1
    assert final_chunks[0].output is not None
    # Indices start at 0 and are contiguous
    assert [c.step_index for c in step_chunks] == list(range(len(step_chunks)))


# ---------------------------------------------------------------------------
# Error path: backend that raises during streaming
# ---------------------------------------------------------------------------


class _ExplodingBackend(VLMBackend):
    @property
    def name(self) -> str:
        return "explode"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:  # noqa: ARG002
        raise RuntimeError("boom")

    def stream_infer(self, image_array: np.ndarray, question: str) -> Iterator[VLMStreamChunk]:  # noqa: ARG002
        raise RuntimeError("boom")


def test_stream_emits_error_event_on_backend_failure(client: TestClient, mock_image_b64: str) -> None:
    from miru.models import registry

    registry.register("explode", lambda: _ExplodingBackend())
    try:
        resp = client.post(
            "/analyze/stream",
            json={"image_b64": mock_image_b64, "question": "boom", "backend": "explode"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        kinds = [k for k, _ in events]
        assert "error" in kinds
        err = next(data for kind, data in events if kind == "error")
        assert "boom" in err["detail"]
    finally:
        registry._REGISTRY.pop("explode", None)  # type: ignore[attr-defined]
