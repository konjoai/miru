"""Attention map extraction, normalization, and analysis utilities."""
import numpy as np


class AttentionExtractor:
    """Extract, normalize, and analyse raw attention weight arrays.

    Args:
        resolution: Output grid size N; ``extract`` always returns (N, N).
    """

    def __init__(self, resolution: int = 16) -> None:
        self.resolution = resolution

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def normalize(self, raw_weights: np.ndarray) -> np.ndarray:
        """Min-max normalize *raw_weights* to [0, 1].

        A uniform (or near-uniform) input returns an all-zeros array instead
        of dividing by a near-zero range.

        Args:
            raw_weights: Arbitrary non-negative float array.

        Returns:
            float32 array of the same shape with values in [0, 1].
        """
        min_val = float(raw_weights.min())
        max_val = float(raw_weights.max())
        if max_val - min_val < 1e-8:
            return np.zeros_like(raw_weights, dtype=np.float32)
        return ((raw_weights - min_val) / (max_val - min_val)).astype(np.float32)

    def resize_to_grid(
        self, weights: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Resize a 2-D attention map to (target_h, target_w) via block averaging.

        Pure NumPy — no SciPy or PIL dependency.

        Args:
            weights: 2-D float array of shape (src_h, src_w).
            target_h: Desired output height.
            target_w: Desired output width.

        Returns:
            float32 array of shape (target_h, target_w).
        """
        src_h, src_w = weights.shape
        result = np.zeros((target_h, target_w), dtype=np.float32)
        for i in range(target_h):
            r0 = int(i * src_h / target_h)
            r1 = int((i + 1) * src_h / target_h)
            r1 = max(r1, r0 + 1)  # guarantee at least one row
            for j in range(target_w):
                c0 = int(j * src_w / target_w)
                c1 = int((j + 1) * src_w / target_w)
                c1 = max(c1, c0 + 1)  # guarantee at least one column
                result[i, j] = float(weights[r0:r1, c0:c1].mean())
        return result

    def extract(self, raw_weights: np.ndarray) -> np.ndarray:
        """Full pipeline: normalize then resize to ``(resolution, resolution)``.

        Args:
            raw_weights: Raw attention weights from a VLM backend.

        Returns:
            float32 array of shape (self.resolution, self.resolution).
        """
        normalized = self.normalize(raw_weights)
        if normalized.shape != (self.resolution, self.resolution):
            normalized = self.resize_to_grid(normalized, self.resolution, self.resolution)
        return normalized

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def top_k_regions(
        self, attention_map: np.ndarray, k: int = 3
    ) -> list[tuple[int, int, float]]:
        """Return the *k* highest-activation grid cells sorted by score (desc).

        Args:
            attention_map: 2-D attention map (any resolution).
            k: Number of hotspots to return.

        Returns:
            List of (row, col, score) tuples, sorted highest-score first.
        """
        if k <= 0:
            return []
        flat = attention_map.flatten()
        n = flat.size
        k_clamped = min(k, n)
        # argpartition gives top-k in arbitrary order; argsort then sorts them.
        indices = np.argpartition(flat, -k_clamped)[-k_clamped:]
        indices = indices[np.argsort(flat[indices])[::-1]]
        rows, cols = np.unravel_index(indices, attention_map.shape)
        return [
            (int(r), int(c), float(attention_map[r, c]))
            for r, c in zip(rows, cols)
        ]
