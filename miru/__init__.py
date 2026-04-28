"""Miru — multimodal reasoning tracer and VLM explainability engine."""
__version__ = "0.1.0"

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend
from miru.models.mock import MockVLMBackend
from miru.reasoning.tracer import ReasoningTracer

__all__ = [
    "VLMBackend",
    "MockVLMBackend",
    "AttentionExtractor",
    "ReasoningTracer",
]
