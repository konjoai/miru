"""Deterministic mock VLM backend — used in tests and dev mode."""
import numpy as np

from miru.models.base import VLMBackend, VLMOutput

# Five canned answers; choice is driven by hash(question) % len(_ANSWERS).
_ANSWERS: tuple[str, ...] = (
    "The image shows a natural outdoor scene.",
    "This appears to be an urban environment.",
    "The subject occupies the central region of the image.",
    "Multiple distinct objects are present.",
    "The dominant color suggests an artificial light source.",
)

_REASONING_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "Identified salient foreground regions via spatial attention.",
        "Cross-referenced texture and color histograms with question context.",
        "Synthesized token-level evidence to produce the final answer.",
    ),
)


class MockVLMBackend(VLMBackend):
    """Deterministic mock backend for testing and development.

    The same (seed, question) pair always produces the same output, making it
    safe to use in snapshot regression tests.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "mock"

    def infer(self, image_array: np.ndarray, question: str) -> VLMOutput:  # noqa: ARG002
        """Return a deterministic VLMOutput driven by the question hash.

        The image_array is accepted for API compatibility but is not used by
        the mock — outputs depend only on the question string.
        """
        # Derive a stable integer key from the question so outputs are
        # reproducible regardless of Python's hash randomisation.
        q_key = _stable_hash(question)

        answer = _ANSWERS[q_key % len(_ANSWERS)]

        # Confidence: map question length (clamped to [1, 200]) to [0.70, 0.99].
        norm_len = min(len(question), 200) / 200.0
        confidence = float(0.70 + 0.29 * norm_len)

        attention_weights = self._make_gaussian_map(q_key)

        reasoning_steps = list(_REASONING_TEMPLATES[0])

        return VLMOutput(
            answer=answer,
            confidence=confidence,
            attention_weights=attention_weights,
            reasoning_steps=reasoning_steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_gaussian_map(self, q_key: int, size: int = 16, sigma: float = 3.0) -> np.ndarray:
        """Return a 16×16 float32 Gaussian attention blob.

        Center position is derived from q_key so different questions produce
        different attention patterns.  Uses the instance seed for the base RNG
        but the final map is *deterministic* given (seed, q_key).
        """
        row_center = float(q_key % size)
        col_center = float((q_key // size) % size)

        rows = np.arange(size, dtype=np.float32)
        cols = np.arange(size, dtype=np.float32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")  # (size, size)

        blob: np.ndarray = np.exp(
            -((rr - row_center) ** 2 + (cc - col_center) ** 2) / (2.0 * sigma**2)
        ).astype(np.float32)
        return blob


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """Stable, platform-independent hash for a string.

    Uses a simple polynomial rolling hash so results do not depend on Python's
    PYTHONHASHSEED environment variable.
    """
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFF_FFFF
    return h
