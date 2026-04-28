"""Unit tests for ReasoningTracer."""
from __future__ import annotations

import numpy as np
import pytest

from miru.models.base import VLMOutput
from miru.reasoning.tracer import ReasoningTracer
from miru.schemas import ReasoningTrace


def _make_output(
    answer: str = "Test answer",
    confidence: float = 0.85,
    steps: list[str] | None = None,
) -> VLMOutput:
    if steps is None:
        steps = ["Step A", "Step B", "Step C"]
    attention = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    return VLMOutput(
        answer=answer,
        confidence=confidence,
        attention_weights=attention,
        reasoning_steps=steps,
    )


@pytest.fixture
def tracer() -> ReasoningTracer:
    return ReasoningTracer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trace_returns_reasoning_trace(tracer: ReasoningTracer) -> None:
    result = tracer.trace(_make_output(), backend_name="mock", latency_ms=12.5)
    assert isinstance(result, ReasoningTrace)


def test_trace_answer_preserved(tracer: ReasoningTracer) -> None:
    output = _make_output(answer="Unique answer 42")
    result = tracer.trace(output, backend_name="mock", latency_ms=5.0)
    assert result.answer == "Unique answer 42"


def test_trace_steps_count(tracer: ReasoningTracer) -> None:
    output = _make_output(steps=["A", "B", "C"])
    result = tracer.trace(output, backend_name="mock", latency_ms=5.0)
    assert len(result.steps) == 3


def test_trace_step_confidence_decreasing(tracer: ReasoningTracer) -> None:
    """Confidence must be monotonically non-increasing across steps."""
    output = _make_output(confidence=0.90, steps=["X", "Y", "Z"])
    result = tracer.trace(output, backend_name="mock", latency_ms=5.0)
    confidences = [s.confidence for s in result.steps]
    assert confidences == sorted(confidences, reverse=True)


def test_trace_attention_map_shape(tracer: ReasoningTracer) -> None:
    result = tracer.trace(_make_output(), backend_name="mock", latency_ms=5.0)
    attn = result.attention_map
    assert attn.height == 16
    assert attn.width == 16
    assert len(attn.data) == 16
    assert all(len(row) == 16 for row in attn.data)


def test_trace_latency_preserved(tracer: ReasoningTracer) -> None:
    result = tracer.trace(_make_output(), backend_name="mock", latency_ms=99.9)
    assert result.latency_ms == pytest.approx(99.9)


def test_trace_backend_preserved(tracer: ReasoningTracer) -> None:
    result = tracer.trace(_make_output(), backend_name="my_backend", latency_ms=1.0)
    assert result.backend == "my_backend"


def test_trace_step_numbers_sequential(tracer: ReasoningTracer) -> None:
    output = _make_output(steps=["A", "B", "C"])
    result = tracer.trace(output, backend_name="mock", latency_ms=1.0)
    assert [s.step for s in result.steps] == [1, 2, 3]


def test_trace_attention_values_in_range(tracer: ReasoningTracer) -> None:
    result = tracer.trace(_make_output(), backend_name="mock", latency_ms=1.0)
    for row in result.attention_map.data:
        for val in row:
            assert 0.0 <= val <= 1.0 + 1e-6
