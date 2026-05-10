"""Gradient-style saliency explainer.

What this implements
--------------------

True Grad-CAM (Selvaraju et al. 2016) backprops the gradient of the
target class score through the last conv/attention layer and reweights
the activations.  That requires a torch graph, a CNN/ViT, and access
to per-layer gradients — none of which is portable across a black-box
``VLMBackend`` interface.

So this module implements **occlusion-sensitivity saliency** (Zeiler &
Fergus, 2014, *Visualizing and Understanding Convolutional Networks*) —
the gradient-free cousin that the saliency-toolkit ecosystem (e.g.
captum, tf-explain) groups alongside Grad-CAM under "input-attribution
heatmaps".  We expose it under the ``gradcam`` method name because the
visual semantics are identical (per-region importance heatmap) and
swapping in true Grad-CAM later — once a torch backend lands — is a
drop-in replacement at this call site.

Method
------

For each cell of an ``occlusion_grid × occlusion_grid`` partition of
the image:

1. Build a perturbed image with that cell replaced by the per-image
   mean colour ("occluded").
2. Re-run ``backend.infer()`` and extract the post-extractor
   normalized attention map.
3. Compute the L1 change versus the unoccluded attention map.

Cells whose occlusion produces *large* attention shifts are the cells
the model relied on most → high saliency.

This is a real, citable, model-agnostic saliency method.  The output
is a normalized ``[0, 1]`` float32 grid at ``resolution × resolution``.

Konjo
-----
The docstring above and the ``/methods`` description on the API tell
the truth about what's implemented.  No silent mislabelling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miru.attention.extractor import AttentionExtractor
from miru.models.base import VLMBackend

DEFAULT_OCCLUSION_GRID = 8
DEFAULT_RESOLUTION = 16


@dataclass(frozen=True)
class GradCAMExplanation:
    """Result of one occlusion-sensitivity run."""

    saliency: np.ndarray            # (resolution, resolution) float32 in [0, 1]
    raw_response: np.ndarray        # (occlusion_grid, occlusion_grid) float32
    occlusion_grid: int
    n_calls: int


def explain(
    backend: VLMBackend,
    image: np.ndarray,
    question: str,
    *,
    occlusion_grid: int = DEFAULT_OCCLUSION_GRID,
    resolution: int = DEFAULT_RESOLUTION,
) -> GradCAMExplanation:
    """Compute a saliency map by occlusion sensitivity.

    Args:
        backend:        Any registered :class:`VLMBackend`.  ``infer()``
                        is called ``occlusion_grid**2 + 1`` times.
        image:          (H, W, 3) float32 image in ``[0, 1]``.
        question:       Prompt passed to every backend call.
        occlusion_grid: Side of the partition; total occlusions =
                        ``occlusion_grid ** 2``.
        resolution:     Output saliency grid resolution.

    Returns:
        :class:`GradCAMExplanation` with the normalized saliency map.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    if occlusion_grid < 2:
        raise ValueError(f"occlusion_grid must be >= 2, got {occlusion_grid}")

    h, w = image.shape[:2]
    extractor = AttentionExtractor(resolution=resolution)

    baseline = extractor.extract(backend.infer(image, question).attention_weights)
    baseline_flat = baseline.flatten().astype(np.float64)

    fill = image.mean(axis=(0, 1)).astype(np.float32)
    fill_image = np.broadcast_to(fill, image.shape).astype(np.float32)

    # Block boundaries on the source image grid.
    row_edges = np.linspace(0, h, occlusion_grid + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, occlusion_grid + 1, dtype=np.int32)

    response = np.zeros((occlusion_grid, occlusion_grid), dtype=np.float32)
    for r in range(occlusion_grid):
        for c in range(occlusion_grid):
            r0, r1 = int(row_edges[r]), int(row_edges[r + 1])
            c0, c1 = int(col_edges[c]), int(col_edges[c + 1])
            perturbed = image.copy()
            perturbed[r0:r1, c0:c1] = fill_image[r0:r1, c0:c1]

            attn = extractor.extract(backend.infer(perturbed, question).attention_weights)
            # L1 shift from baseline → "did occluding here matter?"
            response[r, c] = float(np.abs(attn.flatten().astype(np.float64) - baseline_flat).sum())

    # Resize occlusion-grid response → output resolution and normalize.
    if response.max() > response.min():
        normed = (response - response.min()) / (response.max() - response.min())
    else:
        normed = np.zeros_like(response)
    saliency = extractor.resize_to_grid(normed.astype(np.float32), resolution, resolution)

    return GradCAMExplanation(
        saliency=saliency,
        raw_response=response,
        occlusion_grid=occlusion_grid,
        n_calls=occlusion_grid * occlusion_grid + 1,
    )


__all__ = [
    "GradCAMExplanation",
    "DEFAULT_OCCLUSION_GRID",
    "DEFAULT_RESOLUTION",
    "explain",
]
