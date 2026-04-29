"""ReasoningTracer: converts raw VLMOutput into a structured ReasoningTrace."""
from __future__ import annotations

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMOutput
from miru.schemas import AttentionMap, ReasoningStep, ReasoningTrace


class ReasoningTracer:
    """Build a structured ReasoningTrace from a VLMOutput.

    Args:
        extractor: AttentionExtractor instance; defaults to a fresh one with
            the default 16×16 resolution.
    """

    def __init__(self, extractor: AttentionExtractor | None = None) -> None:
        self._extractor: AttentionExtractor = extractor or AttentionExtractor()

    def trace(
        self,
        output: VLMOutput,
        backend_name: str,
        latency_ms: float,
        image_b64: str | None = None,
        generate_overlay: bool = False,
    ) -> ReasoningTrace:
        """Convert a raw VLMOutput into a fully structured ReasoningTrace.

        Confidence for each step decays slightly with step index to reflect
        that later reasoning steps have accumulated more uncertainty.

        Args:
            output: Raw inference result from a VLMBackend.
            backend_name: Identifier of the backend that produced *output*.
            latency_ms: Wall-clock inference time in milliseconds.
            image_b64: Optional base64-encoded source image.  Required when
                *generate_overlay* is True.
            generate_overlay: When True and *image_b64* is provided, generate
                a PNG overlay and attach it to the trace as ``overlay_b64``.

        Returns:
            Immutable ReasoningTrace ready for API serialisation.
        """
        # --- Attention map -----------------------------------------------
        attention_grid = self._extractor.extract(output.attention_weights)
        h, w = attention_grid.shape
        attn_map = AttentionMap(
            width=w,
            height=h,
            data=attention_grid.tolist(),
        )

        # --- Reasoning steps ----------------------------------------------
        # Each step's confidence decays by 5 % relative to the previous step
        # to model the compounding uncertainty of multi-step reasoning.
        steps: list[ReasoningStep] = [
            ReasoningStep(
                step=i + 1,
                description=desc,
                confidence=min(1.0, output.confidence * (1.0 - 0.05 * i)),
            )
            for i, desc in enumerate(output.reasoning_steps)
        ]

        # --- Optional overlay --------------------------------------------
        overlay_b64: str | None = None
        if generate_overlay and image_b64 is not None:
            try:
                from miru.visualization.overlay import generate_overlay as _gen_overlay

                overlay_b64 = _gen_overlay(image_b64, attention_grid)
            except Exception:  # noqa: BLE001 — never crash the trace on overlay failure
                overlay_b64 = None

        return ReasoningTrace(
            answer=output.answer,
            steps=steps,
            attention_map=attn_map,
            backend=backend_name,
            latency_ms=latency_ms,
            overlay_b64=overlay_b64,
        )
