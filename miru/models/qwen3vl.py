"""Qwen3-VL backend using HuggingFace transformers.

Qwen3-VL (Alibaba, Sept 2025) is an open vision-language model with a
ViT vision encoder and accessible cross-modal attention.  Unlike CLIP —
a dual-encoder that only scores image/text similarity — Qwen3-VL is a
generative VLM, so its attention reflects genuine cross-modal reasoning
between the question tokens and the image patches.  That makes it the
first miru backend whose saliency the synergy probe (``miru.synergy``)
and the deletion test (``miru.fidelity``) can meaningfully interrogate.

Attention extraction
--------------------
A forward pass with ``output_attentions=True`` yields per-layer maps of
shape ``(batch, heads, seq, seq)``.  We:

1. pick a **middle** decoder layer — cross-modal fusion concentrates in
   the middle of the stack (Qwen2.5-VL technical report, arXiv:2502.13923,
   found layers ~14-24 carry the peak vision-language signal);
2. read the **last prompt token's** attention to the **image tokens**
   (located via ``config.image_token_id``), averaged over heads;
3. reshape that 1-D vector to the nearest square patch grid.

The numeric reshaping logic lives in pure helpers
(:func:`_select_middle_layer`, :func:`_attention_row_to_grid`) so it is
unit-tested offline; the model load + generation path is exercised only
under ``MIRU_TEST_REAL_BACKENDS=1``.

Requires: ``pip install 'transformers>=4.57' torch`` (Qwen3-VL is
integrated natively in recent transformers).  ``output_attentions``
requires the eager attention implementation.
"""

from __future__ import annotations

import numpy as np

from miru.models.base import VLMBackend, VLMOutput

_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_MIDDLE_LAYER_FRACTION = 0.6  # ~60% deep — within the empirical fusion band.


def _select_middle_layer(
    num_layers: int, fraction: float = _MIDDLE_LAYER_FRACTION
) -> int:
    """Index of the decoder layer ``fraction`` of the way up the stack.

    Cross-modal fusion peaks mid-stack, so we sample there rather than at
    the final layer.  Clamped to a valid ``[0, num_layers - 1]`` index.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    idx = int(round((num_layers - 1) * fraction))
    return max(0, min(num_layers - 1, idx))


def _attention_row_to_grid(attn_row: np.ndarray) -> np.ndarray:
    """Reshape a 1-D image-token attention vector to a square float32 grid.

    The vector is truncated to the largest square ``g*g <= len`` and
    reshaped to ``(g, g)`` — mirroring the CLIP backend's patch-grid
    derivation.
    """
    flat = np.asarray(attn_row, dtype=np.float32).ravel()
    if flat.size == 0:
        raise ValueError("attn_row must be non-empty")
    grid = int(flat.size**0.5)
    if grid < 1:
        raise ValueError(f"too few image tokens to form a grid: {flat.size}")
    return flat[: grid * grid].reshape(grid, grid).astype(np.float32)


class Qwen3VLBackend(VLMBackend):
    """Generative VLM backend over Qwen3-VL with cross-modal attention.

    Lazy-loads weights on the first :meth:`infer` call — never at import
    time — so the module imports cleanly in mock-only environments.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load model + processor on first inference call."""
        if self._model is not None:
            return
        import torch  # noqa: F401  -- kept out of module scope (optional dep)
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name,
            attn_implementation="eager",  # required for output_attentions
        )
        self._model.eval()

    @property
    def name(self) -> str:
        return "qwen3vl"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
        """Run Qwen3-VL inference.

        Args:
            image_array: float32 ``(H, W, 3)`` in ``[0, 1]`` — converted to
                a uint8 PIL image internally.
            question: The user question conditioning the model.

        Returns:
            ``VLMOutput`` with the generated answer, a confidence derived
            from the first generated token's probability, a square
            cross-modal attention grid, and reasoning steps.
        """
        import torch
        from PIL import Image

        self._load()

        img_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[pil_image], return_tensors="pt")

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
            )
            forward = self._model(**inputs, output_attentions=True)

        answer = self._decode_answer(inputs, generated)
        confidence = _first_token_confidence(generated)
        attention_map = self._extract_attention(inputs, forward)

        return VLMOutput(
            answer=answer,
            confidence=confidence,
            attention_weights=attention_map,
            reasoning_steps=[
                f"Encoded image + question with Qwen3-VL ({self._model_name})",
                f"Generated answer: '{answer[:50]}'",
                "Read cross-modal attention from a middle decoder layer",
            ],
        )

    def _decode_answer(self, inputs: object, generated: object) -> str:
        """Strip the prompt tokens and decode only the generated continuation."""
        input_len = inputs["input_ids"].shape[1]  # type: ignore[index]
        new_tokens = generated.sequences[0][input_len:]  # type: ignore[attr-defined]
        return self._processor.batch_decode(  # type: ignore[union-attr]
            [new_tokens], skip_special_tokens=True
        )[0].strip()

    def _extract_attention(self, inputs: object, forward: object) -> np.ndarray:
        """Middle-layer, last-token attention over image tokens → square grid."""
        attentions = forward.attentions  # type: ignore[attr-defined]
        layer = _select_middle_layer(len(attentions))
        layer_attn = attentions[layer][0]  # (heads, seq, seq)
        image_token_id = self._model.config.image_token_id  # type: ignore[union-attr]
        ids = inputs["input_ids"][0]  # type: ignore[index]
        image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
        # Last prompt token's head-averaged attention to the image tokens.
        last_to_image = layer_attn[:, -1, :].mean(dim=0)[image_positions]
        return _attention_row_to_grid(last_to_image.float().cpu().numpy())


def _first_token_confidence(generated: object) -> float:
    """Confidence = softmax probability of the first generated token.

    A real signal of the model's certainty in its answer, clamped to
    ``[0, 1]``.  Falls back to ``0.5`` when scores are unavailable.
    """
    import torch

    scores = getattr(generated, "scores", None)
    if not scores:
        return 0.5
    probs = torch.softmax(scores[0][0], dim=-1)
    return float(np.clip(float(probs.max().item()), 0.0, 1.0))


__all__ = [
    "Qwen3VLBackend",
]
