"""Pydantic request/response schemas for Miru."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class ImageInput(BaseModel):
    """Request payload for /analyze."""

    model_config = ConfigDict(frozen=True)

    image_b64: str
    question: str
    backend: str = "mock"


class AttentionMap(BaseModel):
    """Normalized attention grid (H × W, values in [0, 1])."""

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    data: list[list[float]]

    @field_validator("data")
    @classmethod
    def _validate_data_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("data must be non-empty")
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError("All rows in data must have equal length")
        return v


class ReasoningStep(BaseModel):
    """Single step in a structured reasoning trace."""

    model_config = ConfigDict(frozen=True)

    step: int
    description: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class ReasoningTrace(BaseModel):
    """Full inference result with reasoning trace and attention map."""

    model_config = ConfigDict(frozen=True)

    answer: str
    steps: list[ReasoningStep]
    attention_map: AttentionMap
    backend: str
    latency_ms: float
    overlay_b64: str | None = None  # base64 PNG overlay; present when overlay=true


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    backends: list[str]


class ErrorResponse(BaseModel):
    """Structured error payload."""

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str


class ExplainRequest(BaseModel):
    """Request payload for /explain.

    ``method`` selects the explainability technique:
      * ``attention`` — collapse the backend's raw attention map.
      * ``gradcam`` — Grad-CAM (falls back to attention for ViT backbones).
      * ``lime`` / ``shap`` — roadmap; returns 501.
    """

    model_config = ConfigDict(frozen=True)

    image_b64: str
    question: str = ""
    backend: str = "mock"
    method: str = "attention"
    target_class: int | None = None
    top_k: int = 5


class ExplainRegion(BaseModel):
    """One top-attended region in the heatmap grid."""

    model_config = ConfigDict(frozen=True)

    row: int
    col: int
    score: float
    bbox_x1: float  # normalised image coords in [0, 1]
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


class ExplainResponse(BaseModel):
    """Response payload for /explain."""

    model_config = ConfigDict(frozen=True)

    method: str
    status: str  # "implemented" | "roadmap"
    backend: str
    answer: str
    width: int
    height: int
    heatmap: list[list[float]]
    top_regions: list[ExplainRegion]
    used_fallback: bool
    overlay_b64: str | None = None
    latency_ms: float
