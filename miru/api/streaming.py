"""Server-Sent Events (SSE) streaming for /analyze/stream.

Wire format
-----------
Each event is a UTF-8 string of the form::

    event: <event-name>\\n
    data: <json>\\n
    \\n

Event names emitted, in order:

- ``step`` — one reasoning step is now available.  Payload::

      {"step": <int>, "description": <str>, "confidence": <float>}

- ``trace`` — full :class:`miru.schemas.ReasoningTrace` JSON.
- ``done``  — empty payload sentinel; clients may close the connection.
- ``error`` — emitted in place of ``trace``/``done`` if inference fails.

A ``: keepalive`` SSE comment is emitted every ``keepalive_seconds`` while
waiting on the upstream backend so intermediate proxies do not idle-close
the connection.  An overall ``timeout_seconds`` budget is enforced on the
inference path; on timeout an ``error`` event is emitted and the stream
closes cleanly.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend, VLMStreamChunk
from miru.reasoning.tracer import ReasoningTracer, step_confidence
from miru.schemas import AttentionMap, ReasoningStep, ReasoningTrace
from miru.visualization.overlay import generate_overlay as _gen_overlay


def _format_sse(event: str, data: dict) -> bytes:
    """Encode one SSE frame as UTF-8 bytes."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _keepalive() -> bytes:
    return b": keepalive\n\n"


async def stream_analyze(
    backend: VLMBackend,
    image_array: np.ndarray,
    question: str,
    *,
    image_b64: Optional[str] = None,
    overlay: bool = False,
    extractor: Optional[AttentionExtractor] = None,
    timeout_seconds: float = 30.0,
    keepalive_seconds: float = 5.0,
) -> AsyncIterator[bytes]:
    """Async generator yielding SSE-framed bytes for /analyze/stream.

    Runs ``backend.stream_infer`` in a worker thread so synchronous backends
    cooperate with the event loop.  Per-chunk arrival is monitored against a
    keepalive interval; overall completion is bounded by ``timeout_seconds``.
    """
    extractor = extractor or AttentionExtractor()
    loop = asyncio.get_running_loop()

    # Run the synchronous generator in a thread, marshaling chunks via a queue.
    # Sentinels: ``_DONE`` for clean completion, ``_ERROR`` carrier for failure.
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    _DONE = object()

    def _producer() -> None:
        try:
            for chunk in backend.stream_infer(image_array, question):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as exc:  # noqa: BLE001 — propagate as error event
            asyncio.run_coroutine_threadsafe(queue.put(("__error__", repr(exc))), loop).result()
            return
        asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    producer_task = loop.run_in_executor(None, _producer)

    t0 = time.perf_counter()
    final_output = None
    step_index = 1
    deadline = time.perf_counter() + timeout_seconds

    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                yield _format_sse(
                    "error",
                    {"error": "timeout", "detail": f"exceeded {timeout_seconds}s budget"},
                )
                return

            wait = min(keepalive_seconds, remaining)
            try:
                item = await asyncio.wait_for(queue.get(), timeout=wait)
            except asyncio.TimeoutError:
                yield _keepalive()
                continue

            if item is _DONE:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                yield _format_sse("error", {"error": "inference_failed", "detail": item[1]})
                return

            chunk: VLMStreamChunk = item  # type: ignore[assignment]
            if chunk.kind == "step":
                # Confidence per step decays from a placeholder base of 1.0
                # until the final output arrives; clients receive a refined
                # confidence in the trailing ``trace`` event.
                yield _format_sse(
                    "step",
                    {
                        "step": step_index,
                        "description": chunk.step_description or "",
                        "confidence": 1.0,
                    },
                )
                step_index += 1
            elif chunk.kind == "final":
                final_output = chunk.output
            else:  # pragma: no cover — defensive
                continue
    finally:
        await producer_task

    if final_output is None:
        yield _format_sse("error", {"error": "no_output", "detail": "backend produced no final chunk"})
        return

    latency_ms = (time.perf_counter() - t0) * 1_000.0

    # Build full trace using the same logic as ReasoningTracer so the streamed
    # ``trace`` event matches /analyze byte-for-byte for identical input.
    attention_grid = extractor.extract(final_output.attention_weights)
    h, w = attention_grid.shape
    attn_map = AttentionMap(width=w, height=h, data=attention_grid.tolist())
    steps = [
        ReasoningStep(
            step=i + 1,
            description=desc,
            confidence=step_confidence(final_output.confidence, i),
        )
        for i, desc in enumerate(final_output.reasoning_steps)
    ]
    overlay_b64: Optional[str] = None
    if overlay and image_b64 is not None:
        try:
            overlay_b64 = _gen_overlay(image_b64, attention_grid)
        except Exception:  # noqa: BLE001 — never fail the stream on overlay
            overlay_b64 = None

    trace = ReasoningTrace(
        answer=final_output.answer,
        steps=steps,
        attention_map=attn_map,
        backend=backend.name,
        latency_ms=latency_ms,
        overlay_b64=overlay_b64,
    )

    yield _format_sse("trace", trace.model_dump())
    yield _format_sse("done", {})


__all__ = ["stream_analyze"]
