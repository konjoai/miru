"""Miru — multimodal reasoning tracer and VLM explainability engine."""
from __future__ import annotations

__version__ = "0.3.0"

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend
from miru.models.mock import MockVLMBackend
from miru.reasoning.tracer import ReasoningTracer
from miru.visualization.overlay import attention_to_heatmap, generate_overlay

__all__ = [
    "VLMBackend",
    "MockVLMBackend",
    "AttentionExtractor",
    "ReasoningTracer",
    "attention_to_heatmap",
    "generate_overlay",
]
