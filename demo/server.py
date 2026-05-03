"""Miru demo server — runs real Miru code behind a friendly /api surface.

What it does
------------

- Serves ``demo/index.html`` at ``/``
- Wraps Miru's real FastAPI internals under ``/api/*``:
  - ``GET  /api/health``          — service status + backends + recording flag
  - ``POST /api/analyze``         — synchronous analysis; returns the result,
                                    confidence, attention grid and latency
  - ``POST /api/analyze/stream``  — Server-Sent Events stream of step events
                                    followed by a final ``trace`` and ``done``
  - ``GET  /api/traces``          — recent trace records read back from the
                                    real on-disk recorder
- Sets ``MIRU_RECORD=1`` and points ``MIRU_RECORD_PATH`` at ``demo/_traces``
  so every analyse call lands as a real privacy-stripped JSONL record

Run::

    python demo/server.py

Then open ``http://localhost:8000``.

Implementation note: the spec asked for ``GET /api/analyze/stream``, but the
request payload includes a base64 image — it does not fit a GET query string
in any practical way.  POST + ``text/event-stream`` is the canonical pattern
for streaming long-lived responses with a non-trivial body, and the browser
``fetch`` + ``reader.read()`` API consumes it cleanly.  This is the same
Konjo pushback Miru applied when the upstream plan called for GET on its own
``/analyze/stream``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: repo root on sys.path, recorder env vars set BEFORE miru imports.
# ---------------------------------------------------------------------------
_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RECORD_DIR = _DEMO_DIR / "_traces"
_RECORD_DIR.mkdir(exist_ok=True)
os.environ["MIRU_RECORD"] = "1"
os.environ["MIRU_RECORD_PATH"] = str(_RECORD_DIR)

# ---------------------------------------------------------------------------
# Miru imports — happen AFTER the env is set so the recorder picks it up.
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from miru import __version__
from miru.api.routes import _decode_image
from miru.api.streaming import stream_analyze
from miru.config import settings
from miru.models import registry
from miru.reasoning.tracer import ReasoningTracer
from miru.recorder import (
    _list_files,
    _read_lines,
    get_recorder,
    maybe_record,
)

# Initialise registry & shared tracer once at import time.
registry.register_defaults()
_tracer = ReasoningTracer()

# ---------------------------------------------------------------------------
# Wire-format models — friendly shapes per the demo brief.
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 of raw RGB pixel bytes.")
    prompt: str = Field(..., description="Question to ask about the image.")
    backend: str = Field(default="mock", description="Registered backend name.")


class StepRecord(BaseModel):
    step: int
    description: str
    confidence: float


class AnalyzeResponse(BaseModel):
    result: str
    confidence: float
    attention_grid: list[list[float]]
    latency_ms: float
    backend: str
    steps: list[StepRecord]


class HealthResponse(BaseModel):
    status: str
    version: str
    backends: list[str]
    recording: bool
    record_path: str
    record_count: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Miru demo",
    description="Real Miru code, served alongside the demo HTML.",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_DEMO_DIR / "index.html")


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def _record_count() -> int:
    """Total recorded trace lines on disk — best-effort."""
    rec = get_recorder()
    rec.flush()
    total = 0
    for path in _list_files(rec.directory):
        try:
            for _ in _read_lines(path):
                total += 1
        except Exception:  # noqa: BLE001
            continue
    return total


@app.get("/api/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    """Live status — used by the page banner to flip green."""
    return HealthResponse(
        status="ok",
        version=__version__,
        backends=registry.available(),
        recording=os.environ.get("MIRU_RECORD", "").lower() in {"1", "true", "yes", "on"},
        record_path=str(_RECORD_DIR),
        record_count=_record_count(),
    )


# ---------------------------------------------------------------------------
# /api/analyze (sync)
# ---------------------------------------------------------------------------


def _resolve_backend(name: str):
    try:
        return registry.get(name)
    except KeyError:
        return registry.get(settings.default_backend)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def api_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run a single inference and return a friendly response shape."""
    image_array = _decode_image(req.image_b64)
    backend = _resolve_backend(req.backend)

    t0 = time.perf_counter()
    vlm_output = backend.infer(image_array, req.prompt)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    trace = _tracer.trace(vlm_output, backend.name, latency_ms)
    maybe_record(trace.model_dump(), image_b64=req.image_b64, question=req.prompt)

    confidence = max((s.confidence for s in trace.steps), default=0.0)
    return AnalyzeResponse(
        result=trace.answer,
        confidence=confidence,
        attention_grid=trace.attention_map.data,
        latency_ms=trace.latency_ms,
        backend=trace.backend,
        steps=[
            StepRecord(step=s.step, description=s.description, confidence=s.confidence)
            for s in trace.steps
        ],
    )


# ---------------------------------------------------------------------------
# /api/analyze/stream (SSE)
# ---------------------------------------------------------------------------


@app.post("/api/analyze/stream")
def api_analyze_stream(
    req: AnalyzeRequest,
    timeout_seconds: float = Query(default=30.0, ge=1.0, le=300.0),
) -> StreamingResponse:
    """Stream the reasoning trace as Server-Sent Events.

    Frame grammar (handled directly by ``miru.api.streaming.stream_analyze``):

    - ``event: step``  — one reasoning step now available
    - ``event: trace`` — full ReasoningTrace once inference completes
    - ``event: done``  — sentinel
    - ``event: error`` — emitted on backend failure or timeout
    """
    image_array = _decode_image(req.image_b64)
    backend = _resolve_backend(req.backend)

    generator = stream_analyze(
        backend,
        image_array,
        req.prompt,
        image_b64=req.image_b64,
        overlay=False,
        timeout_seconds=timeout_seconds,
        record=True,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# /api/traces — read the recorder back
# ---------------------------------------------------------------------------


@app.get("/api/traces")
def api_traces(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """Drain recently recorded JSONL records — newest first."""
    rec = get_recorder()
    rec.flush()
    records: list[dict[str, Any]] = []
    for path in _list_files(rec.directory):
        for line in _read_lines(path):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "count": len(records),
        "record_path": str(_RECORD_DIR),
        "traces": records[-limit:][::-1],
    }


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def _banner(host: str, port: int) -> None:
    bar = "─" * 62
    print()
    print(f"┌{bar}┐")
    print(f"│  Miru demo · v{__version__}".ljust(63) + "│")
    print(f"│{' ' * 62}│")
    print(f"│  backends:    {', '.join(registry.available())}".ljust(63) + "│")
    print(f"│  recorder:    {_RECORD_DIR}".ljust(63) + "│")
    print(f"│  page:        http://{host}:{port}/".ljust(63) + "│")
    print(f"│  health:      http://{host}:{port}/api/health".ljust(63) + "│")
    print(f"└{bar}┘")
    print()


def main() -> int:
    host = os.environ.get("MIRU_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("MIRU_DEMO_PORT", "8000"))
    _banner(host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
