"""CLIP ViT backend using HuggingFace transformers."""
from __future__ import annotations

import numpy as np

from miru.models.base import VLMBackend, VLMOutput

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class CLIPBackend(VLMBackend):
    """Vision encoder using CLIP ViT.

    Attention maps are derived from the last cross-attention layer's
    [CLS] token attention weights, averaged across heads, reshaped to
    a square grid (sqrt(num_patches) x sqrt(num_patches)).

    Requires: pip install transformers torch
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load model on first inference call."""
        if self._model is not None:
            return
        import torch  # noqa: F401  -- imported here to avoid module-level dependency
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model = CLIPModel.from_pretrained(
            self._model_name,
            output_attentions=True,
        )
        self._model.eval()

    @property
    def name(self) -> str:
        return "clip"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:
        """Run CLIP inference.

        image_array: float32 (H, W, 3) in [0, 1] — converted to uint8 PIL internally.
        Returns VLMOutput with:
          - answer: "yes" or "no" depending on text-image similarity
          - confidence: cosine similarity between image and text embeddings mapped to [0, 1]
          - attention_weights: last-layer vision attention for [CLS] token (H_p x W_p grid)
          - reasoning_steps: [image encoding, text encoding, similarity computation]
        """
        import torch
        from PIL import Image

        self._load()

        # Convert float32 [0, 1] -> uint8 PIL Image
        img_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8)

        inputs = self._processor(
            text=[question, "not " + question],
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

        # Text-image similarity as confidence
        image_embeds = outputs.image_embeds  # (1, D)
        text_embeds = outputs.text_embeds  # (2, D)
        image_embeds_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds_n = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_embeds_n @ text_embeds_n.T).squeeze(0)  # (2,)
        pos_sim = float(sims[0].item())
        confidence = float((pos_sim + 1) / 2)  # map [-1, 1] -> [0, 1]
        answer = "yes" if sims[0] > sims[1] else "no"

        # Extract attention map from last vision encoder layer.
        # vision_model.encoder.layers[-1] attention weights: (1, heads, seq, seq)
        attn = outputs.vision_model_output.attentions[-1]  # (1, H, seq, seq)
        # Mean over heads for [CLS] token (index 0) attending to all patch tokens (1:)
        cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (num_patches,)
        num_patches = cls_attn.shape[0]
        grid_size = int(num_patches**0.5)
        attention_map = cls_attn[: grid_size * grid_size].reshape(grid_size, grid_size).numpy()

        return VLMOutput(
            answer=answer,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            attention_weights=attention_map.astype(np.float32),
            reasoning_steps=[
                f"Encoded image with CLIP ViT ({self._model_name})",
                f"Encoded question: '{question[:50]}'",
                f"Computed text-image similarity: {pos_sim:.3f}",
            ],
        )
