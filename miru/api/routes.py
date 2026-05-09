"""REST endpoints for Miru."""
from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import Response, StreamingResponse

from miru.api.streaming import stream_analyze
from miru.config import settings
from miru.gradcam import GradCAMExplainer, attention_to_cam
from miru.models import registry
from miru.reasoning.tracer import ReasoningTracer
from miru.recorder import maybe_record
from miru.schemas import (
    ErrorResponse,
    ExplainRegion,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ImageInput,
    ReasoningTrace,
)

# Explainability method registry: maps method name → implementation status.
EXPLAIN_METHODS: dict[str, str] = {
    "attention": "implemented",
    "gradcam": "implemented",
    "lime": "roadmap",
    "shap": "roadmap",
}

try:
    from miru.metrics import MiruMetrics

    _metrics = MiruMetrics()
except ImportError:
    _metrics = None  # type: ignore[assignment]

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


@router.get("/metrics")
def metrics() -> Response:
    """Return Prometheus metrics in text exposition format.

    Returns a 404 if prometheus-client is not installed.
    """
    if _metrics is None:
        return Response("Metrics not available: prometheus-client not installed", status_code=404)

    return Response(
        _metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
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
    latency_s = time.perf_counter() - t0
    latency_ms = latency_s * 1_000.0

    if _metrics is not None:
        _metrics.observe_analyze(backend.name, latency_s)

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

    async def _timed_generator():
        """Wrapper that records stream latency after streaming completes."""
        t0 = time.perf_counter()
        async for chunk in stream_analyze(
            backend,
            image_array,
            payload.question,
            image_b64=payload.image_b64,
            overlay=overlay,
            timeout_seconds=timeout_seconds,
            record=True,
        ):
            yield chunk
        latency_s = time.perf_counter() - t0
        if _metrics is not None:
            _metrics.observe_stream(backend.name, latency_s)

    return StreamingResponse(
        _timed_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/explain",
    responses={
        200: {"model": ExplainResponse},
        422: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)
def explain(
    payload: ExplainRequest,
    overlay: bool = Query(default=False, description="Include base64 PNG overlay in response."),
) -> Response:
    """Run an explainability method against the input image.

    Supported ``method`` values: ``attention`` (raw VLM attention),
    ``gradcam`` (Grad-CAM with attention fallback for ViT backbones).
    ``lime`` / ``shap`` are on the roadmap and return 501.
    """
    status = EXPLAIN_METHODS.get(payload.method)
    if status is None:
        return _json_error(422, "unknown_method", f"method '{payload.method}' is not recognised")
    if status == "roadmap":
        return _json_error(501, "not_implemented", f"method '{payload.method}' is on the roadmap")

    image_array = _decode_image(payload.image_b64)
    try:
        backend = registry.get(payload.backend)
    except KeyError:
        backend = registry.get(settings.default_backend)

    t0 = time.perf_counter()
    vlm_output = backend.infer(image_array, payload.question)

    if payload.method == "attention":
        heatmap = attention_to_cam(vlm_output.attention_weights)
        used_fallback = False
    else:  # "gradcam"
        # Backends here expose only post-hoc attention weights, not torch hooks.
        # Use the Grad-CAM attention fallback path: same algorithm as a pure
        # ViT (no Conv2d) would invoke.
        result = GradCAMExplainer.from_attention(
            vlm_output.attention_weights, target_class=payload.target_class, top_k=payload.top_k
        )
        heatmap = result.heatmap
        used_fallback = result.used_fallback

    latency_ms = (time.perf_counter() - t0) * 1_000.0

    h, w = heatmap.shape
    top = _heatmap_top_regions(heatmap, payload.top_k)
    overlay_b64: str | None = None
    if overlay:
        try:
            from miru.visualization.overlay import generate_overlay

            overlay_b64 = generate_overlay(payload.image_b64, heatmap)
        except Exception:  # noqa: BLE001 — overlay is best-effort
            overlay_b64 = None

    return Response(
        content=ExplainResponse(
            method=payload.method,
            status=status,
            backend=backend.name,
            answer=vlm_output.answer,
            width=w,
            height=h,
            heatmap=heatmap.tolist(),
            top_regions=top,
            used_fallback=used_fallback,
            overlay_b64=overlay_b64,
            latency_ms=latency_ms,
        ).model_dump_json(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _json_error(status_code: int, error: str, detail: str) -> Response:
    """Render a structured ErrorResponse with a non-200 status code."""
    body = ErrorResponse(error=error, detail=detail).model_dump_json()
    return Response(content=body, status_code=status_code, media_type="application/json")


def _heatmap_top_regions(heatmap: np.ndarray, k: int) -> list[ExplainRegion]:
    """Build :class:`ExplainRegion` list with normalised bounding boxes.

    Each region's bbox covers exactly one grid cell.  Coordinates are
    image-relative (``[0, 1]``) so demo callers can scale them against the
    rendered image without knowing the heatmap resolution.
    """
    if k <= 0 or heatmap.size == 0:
        return []
    h, w = heatmap.shape
    flat = heatmap.flatten()
    k_clamped = min(k, flat.size)
    idx = np.argpartition(flat, -k_clamped)[-k_clamped:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    rows, cols = np.unravel_index(idx, heatmap.shape)
    regions: list[ExplainRegion] = []
    for r, c in zip(rows, cols):
        regions.append(
            ExplainRegion(
                row=int(r),
                col=int(c),
                score=float(heatmap[r, c]),
                bbox_x1=float(c) / w,
                bbox_y1=float(r) / h,
                bbox_x2=float(c + 1) / w,
                bbox_y2=float(r + 1) / h,
            )
        )
    return regions


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
