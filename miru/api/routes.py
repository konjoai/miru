"""REST endpoints for Miru."""
from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter

from miru.config import settings
from miru.models.base import VLMBackend
from miru.models.mock import MockVLMBackend
from miru.reasoning.tracer import ReasoningTracer
from miru.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageInput,
    ReasoningTrace,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
_backends: dict[str, VLMBackend] = {
    "mock": MockVLMBackend(),
}

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
        backends=list(_backends.keys()),
    )


@router.post(
    "/analyze",
    response_model=ReasoningTrace,
    responses={422: {"model": ErrorResponse}},
)
def analyze(payload: ImageInput) -> ReasoningTrace:
    """Run VLM inference and return a structured reasoning trace.

    Image decoding is best-effort: if the provided base64 payload cannot be
    decoded to a valid array, a 1×1 black RGB image is used as a fallback so
    the endpoint never raises a 5xx on bad image data.
    """
    image_array = _decode_image(payload.image_b64)

    backend = _backends.get(payload.backend, _backends[settings.default_backend])

    t0 = time.perf_counter()
    vlm_output = backend.infer(image_array, payload.question)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    return _tracer.trace(vlm_output, backend.name, latency_ms)


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
