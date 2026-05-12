"""SHAP-style perturbation explainer via tile masking.

Approximates Shapley values using a sampling-based approach:

    φᵢ ≈ E[f(x) | xᵢ present] − E[f(x) | xᵢ absent]

where feature i is a tile, presence = original pixel values,
absence = mean-colour fill (baseline).  Sampling M coalitions per tile;
final attribution is the mean output delta across coalitions.

Reference: Lundberg & Lee 2017 (SHAP), simplified for image grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from miru.models.base import VLMBackend


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SHAPConfig:
    """Configuration for :class:`SHAPExplainer`."""

    grid_size: int = 7          # tile grid (7×7 = 49 tiles)
    n_samples: int = 64         # coalitions sampled per tile
    baseline: str = "mean"      # "mean" | "black" | "white"
    seed: int = 42
    batch_size: int = 8         # tiles processed per forward call (unused; kept for API compat)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class SHAPExplainer:
    """Tile-masking SHAP approximation.

    Works with any VLMBackend — only calls backend.infer() with a
    synthetic PIL Image.  Does NOT require gradient access.

    Attribution: shape (grid_size, grid_size) float32 array, values in [-1, 1].
    Positive = tile contributed to output, negative = tile suppressed output.
    """

    def __init__(self, backend: VLMBackend, config: SHAPConfig | None = None) -> None:
        self._backend = backend
        self._cfg = config if config is not None else SHAPConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, image: Image.Image, prompt: str = "") -> np.ndarray:
        """Return attribution map (H_grid, W_grid) float32.

        Shapley approximation formula:
            φᵢ ≈ (1/M) Σⱼ [ f(S_j ∪ {i}) − f(S_j) ]

        where each S_j is a random coalition of tiles drawn uniformly,
        f(·) is the scalar score (mean of the attention map), and the
        sum is over M sampled coalitions.

        Steps:
            1. Compute per-channel baseline fill from image.
            2. For each tile i, sample n_samples random coalitions
               (binary masks over the other tiles), compute the mean
               delta of the score when tile i is present vs. absent.
            3. Normalise φ to [-1, 1] (min-max with signed centre).
        """
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed)

        img_arr = np.array(image, dtype=np.float32)   # (H, W, 3)
        baseline_fill = self._compute_baseline_fill(image).astype(np.float32)

        n = cfg.grid_size
        phi = np.zeros((n, n), dtype=np.float32)

        # Pre-sample n_samples coalition masks for all tiles (shape: (n_samples, n, n) bool).
        # Each coalition does NOT include the target tile (it is added/removed separately).
        coalition_masks = rng.integers(0, 2, size=(cfg.n_samples, n, n)).astype(bool)

        for row in range(n):
            for col in range(n):
                phi[row, col] = self._tile_delta(
                    img_arr, baseline_fill, row, col, coalition_masks, prompt
                )

        return _normalise_signed(phi)

    def explain_to_attention_map(self, image: Image.Image, prompt: str = "") -> np.ndarray:
        """Convert grid attribution to full image-resolution attention map via bilinear resize.

        Returns:
            float32 array of shape (H, W) matching the input image size.
        """
        grid = self.explain(image, prompt)
        h, w = image.size[1], image.size[0]   # PIL: (width, height)
        return _bilinear_resize(grid, h, w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tile_delta(
        self,
        img_arr: np.ndarray,
        baseline_fill: np.ndarray,
        tile_row: int,
        tile_col: int,
        coalition_masks: np.ndarray,
        prompt: str,
    ) -> float:
        """Compute mean score delta for one tile across all coalitions.

        For each coalition mask (ignoring the target tile's own bit), build
        two images — one with the tile present, one absent — and return
        mean( f(present) − f(absent) ).
        """
        n = self._cfg.grid_size
        deltas: list[float] = []

        for s in range(self._cfg.n_samples):
            # Coalition without the target tile.
            mask = coalition_masks[s].copy()
            mask[tile_row, tile_col] = False

            img_present = self._make_masked_image(
                img_arr, _set_tile(mask, tile_row, tile_col, True), baseline_fill
            )
            img_absent = self._make_masked_image(
                img_arr, _set_tile(mask, tile_row, tile_col, False), baseline_fill
            )
            delta = self._score(img_present, prompt) - self._score(img_absent, prompt)
            deltas.append(delta)

        return float(np.mean(deltas, dtype=np.float64))

    def _make_masked_image(
        self, img_arr: np.ndarray, mask: np.ndarray, baseline_fill: np.ndarray
    ) -> Image.Image:
        """Apply binary tile mask (1=present, 0=baseline) to image.

        Args:
            img_arr:       (H, W, 3) float32 source image.
            mask:          (grid_size, grid_size) bool — True = keep original.
            baseline_fill: (3,) float32 per-channel fill colour.

        Returns:
            PIL Image of the same size as img_arr.
        """
        h, w = img_arr.shape[:2]
        n = self._cfg.grid_size
        out = img_arr.copy()

        row_edges = np.linspace(0, h, n + 1, dtype=np.int32)
        col_edges = np.linspace(0, w, n + 1, dtype=np.int32)

        for r in range(n):
            for c in range(n):
                if not mask[r, c]:
                    r0, r1 = int(row_edges[r]), int(row_edges[r + 1])
                    c0, c1 = int(col_edges[c]), int(col_edges[c + 1])
                    out[r0:r1, c0:c1] = baseline_fill

        clipped = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(clipped, mode="RGB")

    def _score(self, image: Image.Image, prompt: str) -> float:
        """Run backend.infer() and return scalar score (mean of attention map).

        Converts the PIL image to the float32 (H, W, 3) ∈ [0, 1] array
        expected by VLMBackend.infer(), then returns the mean of the raw
        attention weights — a scalar that rises when the model attends more.
        """
        arr = np.array(image, dtype=np.float32) / 255.0
        out = self._backend.infer(arr, prompt)
        return float(np.mean(out.attention_weights, dtype=np.float64))

    def _compute_baseline_fill(self, image: Image.Image) -> np.ndarray:
        """Compute per-channel fill value for the chosen baseline.

        Returns:
            (3,) float32 array with values in [0, 255].
        """
        arr = np.array(image, dtype=np.float32)
        baseline = self._cfg.baseline
        if baseline == "mean":
            return arr.mean(axis=(0, 1)).astype(np.float32)
        if baseline == "black":
            return np.zeros(3, dtype=np.float32)
        if baseline == "white":
            return np.full(3, 255.0, dtype=np.float32)
        raise ValueError(f"Unknown baseline '{baseline}'. Use 'mean', 'black', or 'white'.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _set_tile(mask: np.ndarray, row: int, col: int, value: bool) -> np.ndarray:
    """Return a copy of mask with mask[row, col] set to value."""
    out = mask.copy()
    out[row, col] = value
    return out


def _normalise_signed(phi: np.ndarray) -> np.ndarray:
    """Min-max normalise phi to [-1, 1].

    Maps the minimum value to -1 and the maximum to +1.  If the range is
    degenerate (all equal), returns an all-zeros array.
    """
    lo = float(phi.min())
    hi = float(phi.max())
    span = hi - lo
    if span < 1e-9:
        return np.zeros_like(phi, dtype=np.float32)
    # Shift to [0, 1] then scale to [-1, 1].
    normed = (phi - lo) / span          # in [0, 1]
    return (2.0 * normed - 1.0).astype(np.float32)


def _bilinear_resize(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear upsample a (H_in, W_in) float32 grid to (target_h, target_w).

    Pure NumPy implementation — no PIL dependency for the resize itself.
    """
    src_h, src_w = grid.shape
    row_idx = (np.arange(target_h, dtype=np.float64) + 0.5) * src_h / target_h - 0.5
    col_idx = (np.arange(target_w, dtype=np.float64) + 0.5) * src_w / target_w - 0.5

    row0 = np.clip(np.floor(row_idx).astype(np.int32), 0, src_h - 1)
    row1 = np.clip(row0 + 1, 0, src_h - 1)
    col0 = np.clip(np.floor(col_idx).astype(np.int32), 0, src_w - 1)
    col1 = np.clip(col0 + 1, 0, src_w - 1)

    dr = (row_idx - row0).astype(np.float32)[:, None]   # (H, 1)
    dc = (col_idx - col0).astype(np.float32)[None, :]   # (1, W)

    top = grid[row0, :][:, col0] * (1 - dc) + grid[row0, :][:, col1] * dc
    bot = grid[row1, :][:, col0] * (1 - dc) + grid[row1, :][:, col1] * dc
    return (top * (1 - dr) + bot * dr).astype(np.float32)


__all__ = [
    "SHAPConfig",
    "SHAPExplainer",
]
