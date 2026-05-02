"""REST endpoints for Miru."""
from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from miru.api.streaming import stream_analyze
from miru.config import settings
from miru.models import registry
from miru.reasoning.tracer import ReasoningTracer
from miru.recorder import maybe_record
from miru.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageInput,
    ReasoningTrace,
)

# ---------------------------------------------------------------------------
# Initialise backend registry once at module import time.
# ---------------------------------------------------------------------------
registry.register_defaults()

# Shared tracer instance (stateless — safe for concurrent use).
_tracer = ReasoningTracer()

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and available backend list."""
    return HealthResponse(
        status="ok",
        version=settings.version,
        backends=registry.available(),
    )


@router.post(
    "/analyze",
    response_model=ReasoningTrace,
    responses={422: {"model": ErrorResponse}},
)
def analyze(
    payload: ImageInput,
    overlay: bool = Query(default=False, description="When true, include a base64 PNG attention overlay in the response."),
) -> ReasoningTrace:
    """Run VLM inference and return a structured reasoning trace.

    Image decoding is best-effort: if the provided base64 payload cannot be
    decoded to a valid array, a 1×1 black RGB image is used as a fallback so
    the endpoint never raises a 5xx on bad image data.

    When ``overlay=true`` is passed as a query parameter, a base64-encoded PNG
    showing the attention heatmap blended over the source image is included in
    the ``overlay_b64`` field of the response.  If overlay generation fails for
    any reason the field is ``null`` and the rest of the trace is unaffected.
    """
    image_array = _decode_image(payload.image_b64)

    try:
        backend = registry.get(payload.backend)
    except KeyError:
        backend = registry.get(settings.default_backend)

    t0 = time.perf_counter()
    vlm_output = backend.infer(image_array, payload.question)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    trace = _tracer.trace(
        vlm_output,
        backend.name,
        latency_ms,
        image_b64=payload.image_b64 if overlay else None,
        generate_overlay=overlay,
    )
    maybe_record(trace.model_dump(), image_b64=payload.image_b64, question=payload.question)
    return trace


@router.post(
    "/analyze/stream",
    responses={422: {"model": ErrorResponse}},
)
def analyze_stream(
    payload: ImageInput,
    overlay: bool = Query(default=False, description="Include base64 PNG overlay in the trailing trace event."),
    timeout_seconds: float = Query(default=30.0, ge=1.0, le=300.0, description="Overall inference budget."),
) -> StreamingResponse:
    """Stream a reasoning trace via Server-Sent Events.

    Emits ``step`` events as each reasoning step becomes available, then a
    final ``trace`` event with the complete :class:`ReasoningTrace` JSON,
    followed by a ``done`` sentinel.  See ``miru.api.streaming`` for the
    full event grammar.

    Implementation note: the upstream plan named this ``GET /analyze/stream``
    but the request payload includes a base64 image which does not fit a GET
    query string in any practical way.  POST + ``text/event-stream`` is the
    canonical pattern for streaming long-lived responses with a non-trivial
    request body.
    """
    image_array = _decode_image(payload.image_b64)

    try:
        backend = registry.get(payload.backend)
    except KeyError:
        backend = registry.get(settings.default_backend)

    generator = stream_analyze(
        backend,
        image_array,
        payload.question,
        image_b64=payload.image_b64,
        overlay=overlay,
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
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64 string to a float32 (H, W, 3) array in [0, 1].

    On any failure, returns a 1×1 black RGB pixel.
    """
    _BLACK_PIXEL = np.zeros((1, 1, 3), dtype=np.float32)
    try:
        raw_bytes = base64.b64decode(image_b64, validate=True)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        # Attempt to infer H×W from byte count assuming 3 channels.
        n_pixels = len(arr) // 3
        if n_pixels == 0:
            return _BLACK_PIXEL
        side = int(n_pixels**0.5)
        if side * side * 3 == len(arr):
            image = arr[: side * side * 3].reshape(side, side, 3).astype(np.float32) / 255.0
        else:
            # Fall back to a flat 1×(n_pixels)×3 row image.
            usable = (len(arr) // 3) * 3
            image = arr[:usable].reshape(1, len(arr) // 3, 3).astype(np.float32) / 255.0
        return image
    except Exception:  # noqa: BLE001
        return _BLACK_PIXEL
