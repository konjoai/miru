"""Gradient-weighted Class Activation Mapping (Grad-CAM).

Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (https://arxiv.org/abs/1610.02391).

Math
----
Given a target class ``c`` and the activations ``A^k`` of the last
convolutional layer (``k`` indexes feature-map channel), the channel weight
is the global-average-pooled gradient of the class score:

    α_k^c = (1 / Z) · Σ_{i,j} ∂y^c / ∂A^k_{ij}

The Grad-CAM heatmap is the ReLU of the weighted activation sum:

    L^c = ReLU( Σ_k α_k^c · A^k )

This module separates the math (pure NumPy ``compute_gradcam``) from the
torch hook plumbing (``GradCAMExplainer``), so the algorithm is unit-testable
without torch installed.

Fallback
--------
Pure ViT models (e.g. CLIP-ViT) have no ``Conv2d`` layers, so canonical
Grad-CAM does not apply.  The explainer detects this at construction time and
switches to an attention-weight method: collapse the last self-attention
layer's [CLS]→patches weights (mean over heads) into a 2-D heatmap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GradCAMResult:
    """Output of a Grad-CAM (or attention-fallback) explanation.

    Attributes:
        heatmap: 2-D ``float32`` array with values in ``[0, 1]``.
        top_regions: ``[(row, col, score), ...]`` sorted score-desc.
        target_class: Class index used to compute gradients
            (``None`` when the fallback path is taken without a class score).
        used_fallback: ``True`` when the attention-weight method was used
            because the model lacks ``Conv2d`` layers.
    """

    heatmap: np.ndarray
    top_regions: list[tuple[int, int, float]]
    target_class: int | None
    used_fallback: bool


# ---------------------------------------------------------------------------
# Pure-numpy core — testable without torch
# ---------------------------------------------------------------------------


def compute_gradcam(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Compute the Grad-CAM heatmap from activations and gradients.

    Args:
        activations: Feature-map activations of shape ``(C, H, W)``.
        gradients:   Gradients of the target-class score wrt those activations,
            same shape as ``activations``.

    Returns:
        ``float32`` array of shape ``(H, W)`` with values in ``[0, 1]``.
        A degenerate (all-zero after ReLU) map returns all zeros instead
        of dividing by a near-zero range.
    """
    if activations.shape != gradients.shape:
        raise ValueError(
            f"activations {activations.shape} and gradients {gradients.shape} must match"
        )
    if activations.ndim != 3:
        raise ValueError(f"expected (C, H, W) tensors, got ndim={activations.ndim}")

    # α_k^c = mean over spatial dims of the gradient
    weights = gradients.mean(axis=(1, 2))  # (C,)
    cam = np.einsum("c,chw->hw", weights, activations)
    cam = np.maximum(cam, 0.0)  # ReLU — discard pixels with negative class evidence

    mn = float(cam.min())
    mx = float(cam.max())
    if mx - mn < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - mn) / (mx - mn)).astype(np.float32)


def attention_to_cam(attention: np.ndarray) -> np.ndarray:
    """Collapse a self-attention tensor into a 2-D heatmap (fallback path).

    Accepts either:
      - ``(H, W)`` — already a 2-D map; min-max normalised and returned.
      - ``(heads, seq, seq)`` — assumed to be a Transformer attention block;
        the [CLS] row is averaged across heads, dropped, and reshaped to a
        square patch grid.

    Args:
        attention: Float array, see shapes above.

    Returns:
        ``float32`` 2-D array in ``[0, 1]``.
    """
    arr = np.asarray(attention, dtype=np.float32)
    if arr.ndim == 2:
        m = arr
    elif arr.ndim == 3:
        cls_attn = arr[:, 0, 1:].mean(axis=0)  # (seq-1,)
        n = int(cls_attn.shape[0])
        side = int(round(n**0.5))
        if side * side > n or side == 0:
            raise ValueError(f"attention sequence length {n} is not a perfect square + 1")
        m = cls_attn[: side * side].reshape(side, side)
    else:
        raise ValueError(f"unsupported attention shape {arr.shape}")

    mn = float(m.min())
    mx = float(m.max())
    if mx - mn < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def top_k_regions(heatmap: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    """Return the *k* highest-scoring grid cells, sorted score-desc."""
    if k <= 0 or heatmap.size == 0:
        return []
    flat = heatmap.flatten()
    k_clamped = min(k, flat.size)
    idx = np.argpartition(flat, -k_clamped)[-k_clamped:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    rows, cols = np.unravel_index(idx, heatmap.shape)
    return [(int(r), int(c), float(heatmap[r, c])) for r, c in zip(rows, cols)]


# ---------------------------------------------------------------------------
# Torch-aware explainer
# ---------------------------------------------------------------------------


class GradCAMExplainer:
    """Hook-based Grad-CAM explainer for PyTorch models.

    Usage::

        explainer = GradCAMExplainer(model)               # auto-finds last conv
        result = explainer.explain(image_tensor)          # (1, 3, H, W) float
        result.heatmap                                    # (h, w) float32 in [0, 1]

    If ``model`` has no ``Conv2d`` layers, the explainer transparently falls
    back to the attention-weight method (``result.used_fallback == True``).

    The explainer can also be constructed without a model and used as a
    classmethod-style entry point for the pure-numpy paths
    (:meth:`from_arrays`, :meth:`from_attention`).
    """

    def __init__(self, model: Any | None = None, target_layer: Any | None = None) -> None:
        self._model = model
        self._target_layer = target_layer
        self._use_fallback = False
        if model is not None:
            self._target_layer = target_layer or self._find_last_conv(model)
            if self._target_layer is None:
                self._use_fallback = True

    # ------------------------------------------------------------------
    # Class properties
    # ------------------------------------------------------------------

    @property
    def uses_attention_fallback(self) -> bool:
        """``True`` if no ``Conv2d`` was found and the attention path will run."""
        return self._use_fallback

    @property
    def target_layer(self) -> Any | None:
        """The hooked module (``None`` when falling back to attention)."""
        return self._target_layer

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_conv(model: Any) -> Any | None:
        """Return the last ``nn.Conv2d`` in module-traversal order, or ``None``.

        Returns ``None`` when torch is not installed *or* the model has no
        convolutional layers.
        """
        try:
            import torch.nn as nn
        except ImportError:
            return None
        last_conv = None
        try:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
        except AttributeError:
            return None
        return last_conv

    # ------------------------------------------------------------------
    # Pure-numpy entry points
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        activations: np.ndarray,
        gradients: np.ndarray,
        target_class: int | None = None,
        top_k: int = 5,
    ) -> GradCAMResult:
        """Build a :class:`GradCAMResult` from already-extracted tensors."""
        heatmap = compute_gradcam(activations, gradients)
        return GradCAMResult(
            heatmap=heatmap,
            top_regions=top_k_regions(heatmap, k=top_k),
            target_class=target_class,
            used_fallback=False,
        )

    @classmethod
    def from_attention(
        cls,
        attention: np.ndarray,
        target_class: int | None = None,
        top_k: int = 5,
    ) -> GradCAMResult:
        """Build a fallback :class:`GradCAMResult` from a raw attention tensor."""
        heatmap = attention_to_cam(attention)
        return GradCAMResult(
            heatmap=heatmap,
            top_regions=top_k_regions(heatmap, k=top_k),
            target_class=target_class,
            used_fallback=True,
        )

    # ------------------------------------------------------------------
    # Torch-bound entry point
    # ------------------------------------------------------------------

    def explain(
        self,
        image_tensor: Any,
        target_class: int | None = None,
        top_k: int = 5,
    ) -> GradCAMResult:
        """Run Grad-CAM on a torch model.

        Args:
            image_tensor: ``torch.Tensor`` of shape ``(1, 3, H, W)``.
            target_class: Class index to backprop from.  When ``None``, the
                argmax over the model's output logits is used.
            top_k: Number of top regions to return.

        Returns:
            A :class:`GradCAMResult`.  ``used_fallback`` is ``True`` when the
            model has no ``Conv2d`` layers and the attention path was taken.
        """
        if self._model is None:
            raise RuntimeError(
                "GradCAMExplainer was constructed without a model; "
                "use GradCAMExplainer.from_arrays() or .from_attention() instead."
            )
        if self._use_fallback:
            return self._explain_via_attention(image_tensor, target_class, top_k)
        return self._explain_via_hooks(image_tensor, target_class, top_k)

    # ------------------------------------------------------------------
    # Internal: hook-based Grad-CAM
    # ------------------------------------------------------------------

    def _explain_via_hooks(
        self,
        image_tensor: Any,
        target_class: int | None,
        top_k: int,
    ) -> GradCAMResult:
        import torch  # local import — torch is optional at module level

        captured: dict[str, Any] = {}

        def fwd_hook(_module, _inputs, output):  # type: ignore[no-untyped-def]
            captured["activations"] = output.detach()

        def bwd_hook(_module, _grad_input, grad_output):  # type: ignore[no-untyped-def]
            captured["gradients"] = grad_output[0].detach()

        h_fwd = self._target_layer.register_forward_hook(fwd_hook)
        h_bwd = self._target_layer.register_full_backward_hook(bwd_hook)
        try:
            if hasattr(self._model, "zero_grad"):
                self._model.zero_grad()
            logits = self._model(image_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=-1).item())
            score = logits[0, target_class] if logits.ndim == 2 else logits.flatten()[target_class]
            score.backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

        if "activations" not in captured or "gradients" not in captured:
            raise RuntimeError("Grad-CAM hooks did not fire — check the target_layer is reached during forward")

        a = captured["activations"]
        g = captured["gradients"]
        # Strip leading batch dim and move to CPU numpy.
        a_np = a[0].cpu().numpy() if a.ndim == 4 else a.cpu().numpy()
        g_np = g[0].cpu().numpy() if g.ndim == 4 else g.cpu().numpy()

        heatmap = compute_gradcam(a_np, g_np)
        return GradCAMResult(
            heatmap=heatmap,
            top_regions=top_k_regions(heatmap, k=top_k),
            target_class=int(target_class),
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    # Internal: attention fallback
    # ------------------------------------------------------------------

    def _explain_via_attention(
        self,
        image_tensor: Any,
        target_class: int | None,
        top_k: int,
    ) -> GradCAMResult:
        import torch

        with torch.no_grad():
            try:
                output = self._model(image_tensor, output_attentions=True)
                attn_tensor = output.attentions[-1][0]  # (heads, seq, seq)
                attn_np = attn_tensor.cpu().numpy()
                heatmap = attention_to_cam(attn_np)
            except (TypeError, AttributeError):
                # Model doesn't support output_attentions — give up gracefully.
                output = self._model(image_tensor)
                logits = output if hasattr(output, "shape") else getattr(output, "logits", None)
                if logits is None:
                    heatmap = np.zeros((7, 7), dtype=np.float32)
                else:
                    side = int(round(logits.shape[-1] ** 0.5)) or 1
                    heatmap = np.zeros((side, side), dtype=np.float32)
        return GradCAMResult(
            heatmap=heatmap,
            top_regions=top_k_regions(heatmap, k=top_k),
            target_class=target_class,
            used_fallback=True,
        )
