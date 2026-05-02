"""Abstract VLM backend interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class VLMOutput:
    """Raw output from a VLM inference call.

    Attributes:
        answer: The model's textual answer to the question.
        confidence: Overall confidence score in [0, 1].
        attention_weights: 2-D float32 array (H × W) with non-negative values;
            will be normalized downstream by AttentionExtractor.
        reasoning_steps: Ordered list of intermediate reasoning descriptions.
    """

    answer: str
    confidence: float
    attention_weights: np.ndarray
    reasoning_steps: list[str]


@dataclass(frozen=True)
class VLMStreamChunk:
    """Single chunk produced by :meth:`VLMBackend.stream_infer`.

    Two kinds are emitted:

    - ``kind="step"`` — one reasoning step is now available.  ``step_index``
      and ``step_description`` are populated.
    - ``kind="final"`` — the full :class:`VLMOutput` is available.  ``output``
      is populated.
    """

    kind: str
    step_index: Optional[int] = None
    step_description: Optional[str] = None
    output: Optional[VLMOutput] = None


class VLMBackend(ABC):
    """Pluggable vision-language model backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this backend."""
        ...

    @abstractmethod
    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
        """Run inference.

        Args:
            image_array: float32 array of shape (H, W, 3) with values in [0, 1].
            question: Natural-language question about the image.

        Returns:
            VLMOutput with answer, confidence, raw attention weights, and
            reasoning step descriptions.
        """
        ...

    def stream_infer(
        self, image_array: np.ndarray, question: str
    ) -> Iterator[VLMStreamChunk]:
        """Run inference and yield reasoning steps progressively.

        Default implementation runs :meth:`infer` once and replays its
        ``reasoning_steps`` as ``step`` chunks before emitting the final
        ``VLMOutput``.  Backends with native token-streaming support (e.g. an
        autoregressive VLM) should override this to yield steps as they are
        produced.
        """
        output = self.infer(image_array, question)
        for i, desc in enumerate(output.reasoning_steps):
            yield VLMStreamChunk(kind="step", step_index=i, step_description=desc)
        yield VLMStreamChunk(kind="final", output=output)
