"""Pydantic request/response schemas for Miru."""
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
