"""Abstract VLM backend interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
