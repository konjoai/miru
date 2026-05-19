"""Miru — multimodal reasoning tracer and VLM explainability engine."""
from __future__ import annotations

__version__ = "1.4.0"

from miru.attention.extractor import AttentionExtractor
from miru.gradcam import GradCAMExplainer, GradCAMResult, compute_gradcam
from miru.models.base import VLMBackend
from miru.models.mock import MockVLMBackend
from miru.reasoning.tracer import ReasoningTracer
from miru.visualization.overlay import attention_to_heatmap, generate_overlay

__all__ = [
    "VLMBackend",
    "MockVLMBackend",
    "AttentionExtractor",
    "ReasoningTracer",
    "GradCAMExplainer",
    "GradCAMResult",
    "compute_gradcam",
    "attention_to_heatmap",
    "generate_overlay",
]
