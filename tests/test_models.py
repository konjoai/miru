"""Unit tests for VLM backend implementations."""
import numpy as np
import pytest

from miru.models.base import VLMOutput
from miru.models.mock import MockVLMBackend

_DUMMY_IMAGE = np.zeros((4, 4, 3), dtype=np.float32)
_QUESTION = "What is in this image?"


@pytest.fixture
def backend() -> MockVLMBackend:
    return MockVLMBackend(seed=42)


def test_mock_name(backend: MockVLMBackend) -> None:
    assert backend.name == "mock"


def test_mock_infer_returns_valid_output(backend: MockVLMBackend) -> None:
    output = backend.infer(_DUMMY_IMAGE, _QUESTION)
    assert isinstance(output, VLMOutput)
    assert isinstance(output.answer, str)
    assert isinstance(output.confidence, float)
    assert isinstance(output.attention_weights, np.ndarray)
    assert isinstance(output.reasoning_steps, list)


def test_mock_confidence_range(backend: MockVLMBackend) -> None:
    output = backend.infer(_DUMMY_IMAGE, _QUESTION)
    assert 0.0 <= output.confidence <= 1.0


def test_mock_attention_shape(backend: MockVLMBackend) -> None:
    output = backend.infer(_DUMMY_IMAGE, _QUESTION)
    assert output.attention_weights.shape == (16, 16)


def test_mock_attention_normalized(backend: MockVLMBackend) -> None:
    output = backend.infer(_DUMMY_IMAGE, _QUESTION)
    w = output.attention_weights
    # Raw Gaussian values are already in (0, 1] by construction.
    assert float(w.min()) >= 0.0
    assert float(w.max()) <= 1.0


def test_mock_reasoning_steps_nonempty(backend: MockVLMBackend) -> None:
    output = backend.infer(_DUMMY_IMAGE, _QUESTION)
    assert len(output.reasoning_steps) >= 1
    assert all(isinstance(s, str) and s for s in output.reasoning_steps)


def test_mock_deterministic(backend: MockVLMBackend) -> None:
    """Same question must always produce the same answer from the same seed."""
    out1 = backend.infer(_DUMMY_IMAGE, _QUESTION)
    out2 = backend.infer(_DUMMY_IMAGE, _QUESTION)
    assert out1.answer == out2.answer
    assert out1.confidence == pytest.approx(out2.confidence)
    np.testing.assert_array_equal(out1.attention_weights, out2.attention_weights)
    assert out1.reasoning_steps == out2.reasoning_steps


def test_mock_different_questions_may_differ(backend: MockVLMBackend) -> None:
    """Different questions should produce different attention maps."""
    out1 = backend.infer(_DUMMY_IMAGE, "What color is the sky?")
    out2 = backend.infer(_DUMMY_IMAGE, "How many people are in the photo?")
    # Answers or attention maps should differ (hash collision possible but
    # extremely unlikely for these specific strings).
    differ = (out1.answer != out2.answer) or not np.array_equal(
        out1.attention_weights, out2.attention_weights
    )
    assert differ
