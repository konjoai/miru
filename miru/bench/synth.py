"""Deterministic synthetic image + ground-truth-mask generator.

A benchmark sample bundles four things:

- ``image``    — float32 array of shape ``(H, W, 3)`` in ``[0, 1]``
- ``mask``     — bool array of shape ``(H, W)`` marking the salient region(s)
- ``question`` — string prompt that hints at what the model should locate
- ``meta``     — dict with the seed, variant tag, and centroid(s) used

Every sample is fully reproducible from ``(seed, index)`` — no randomness
leaks across calls and no hidden dependency on machine state.

Three variants exercise different difficulty regimes:

- ``single``  — one bright Gaussian blob on light noise
- ``two``     — two blobs at distinct centroids; mask covers both
- ``low_snr`` — single blob with reduced amplitude over stronger noise

Math
----

Each blob is

    blob(x, y) = A · exp(-((x-cx)² + (y-cy)²) / (2 σ²))

The image is

    image[..., c] = clip(noise + Σ blob_i, 0, 1)

The ground-truth mask is the union of binary disks of radius ``r``
centred at each blob centroid.  ``r`` is set proportional to ``σ`` so
the disk covers the high-density core of the blob.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Variant = Literal["single", "two", "low_snr"]
_VARIANTS: tuple[Variant, ...] = ("single", "two", "low_snr")

DEFAULT_SIZE = 64

# Each variant gets its own bank of question templates so the benchmark
# also exercises the question-conditioning surface of a backend.
_QUESTIONS: dict[Variant, tuple[str, ...]] = {
    "single": (
        "Where is the bright spot?",
        "Locate the dominant region.",
        "Which area stands out the most?",
    ),
    "two": (
        "Where are the two bright regions?",
        "Locate both prominent spots.",
        "Identify the salient regions.",
    ),
    "low_snr": (
        "Find the subtle bright region.",
        "Where is the faint hotspot?",
        "Locate the partially-occluded area.",
    ),
}


@dataclass(frozen=True)
class SynthSample:
    """One benchmark example."""

    image: np.ndarray
    mask: np.ndarray
    question: str
    meta: dict = field(default_factory=dict)


def _gaussian_blob(
    h: int, w: int, cy: float, cx: float, sigma: float, amplitude: float
) -> np.ndarray:
    """Return an (h, w) float32 Gaussian blob at (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    blob = amplitude * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)
    )
    return blob.astype(np.float32)


def _disk_mask(h: int, w: int, cy: float, cx: float, radius: float) -> np.ndarray:
    """Return a boolean disk mask of given radius."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def generate_sample(
    seed: int,
    index: int = 0,
    *,
    size: int = DEFAULT_SIZE,
    variant: Variant | None = None,
) -> SynthSample:
    """Deterministically generate one benchmark sample.

    Args:
        seed: Top-level seed for the whole benchmark run.
        index: Sample index within the run (so the same seed can produce
            many distinct samples by varying ``index``).
        size: Side length of the square output image (default 64).
        variant: Force a specific variant.  Defaults to a deterministic
            cycle through :data:`_VARIANTS` driven by ``index``.

    Returns:
        A :class:`SynthSample` with ``image``, ``mask``, ``question``,
        and ``meta`` populated.
    """
    rng = np.random.default_rng(seed * 10_000 + index)
    chosen: Variant = variant if variant is not None else _VARIANTS[index % len(_VARIANTS)]

    h = w = size
    margin = max(6, size // 8)
    sigma_base = max(2.5, size / 16.0)

    if chosen == "single":
        amplitude = 1.0
        noise_amp = 0.10
        sigma = sigma_base
        centroids = [(_uniform(rng, margin, h - margin), _uniform(rng, margin, w - margin))]
    elif chosen == "two":
        amplitude = 0.85
        noise_amp = 0.10
        sigma = sigma_base * 0.85
        # Sample two centroids that are at least sigma*4 apart so they don't
        # blur into one blob — important for clean ground truth.
        centroids: list[tuple[float, float]] = []
        for _ in range(40):  # bounded retry
            cy = _uniform(rng, margin, h - margin)
            cx = _uniform(rng, margin, w - margin)
            if all(((cy - py) ** 2 + (cx - px) ** 2) ** 0.5 > sigma * 4 for py, px in centroids):
                centroids.append((cy, cx))
            if len(centroids) == 2:
                break
        if len(centroids) < 2:
            # Pathological seed — synthesize a guaranteed-separated pair.
            centroids = [(h * 0.3, w * 0.3), (h * 0.7, w * 0.7)]
    elif chosen == "low_snr":
        amplitude = 0.45
        noise_amp = 0.22
        sigma = sigma_base * 1.1
        centroids = [(_uniform(rng, margin, h - margin), _uniform(rng, margin, w - margin))]
    else:  # pragma: no cover — Literal restricts the value at type level
        raise ValueError(f"unknown variant {chosen!r}")

    # Smooth coloured noise — average three independent fields per channel
    # to avoid flat speckle, in the spirit of low-frequency natural-image noise.
    noise = rng.random((h, w, 3)).astype(np.float32) * noise_amp + 0.10
    smooth = np.zeros_like(noise)
    for ch in range(3):
        # 3-tap horizontal + vertical box smooth, two passes.  Cheap and dep-free.
        x = noise[..., ch]
        for _ in range(2):
            x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0)) / 3.0
            x = (x + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 3.0
        smooth[..., ch] = x

    # Per-channel tint for the blob makes it look less synthetic without
    # changing its spatial location.
    tint = rng.random(3).astype(np.float32) * 0.5 + 0.5

    image = smooth.copy()
    blob_total = np.zeros((h, w), dtype=np.float32)
    for cy, cx in centroids:
        blob = _gaussian_blob(h, w, cy, cx, sigma, amplitude)
        blob_total = np.maximum(blob_total, blob)
    for ch in range(3):
        image[..., ch] = np.clip(image[..., ch] + blob_total * tint[ch], 0.0, 1.0)

    radius = sigma * 1.6
    mask = np.zeros((h, w), dtype=bool)
    for cy, cx in centroids:
        mask |= _disk_mask(h, w, cy, cx, radius)

    question = _QUESTIONS[chosen][index % len(_QUESTIONS[chosen])]
    meta = {
        "seed": int(seed),
        "index": int(index),
        "variant": chosen,
        "size": int(size),
        "centroids": [(float(cy), float(cx)) for cy, cx in centroids],
        "sigma": float(sigma),
        "radius": float(radius),
    }
    return SynthSample(image=image, mask=mask, question=question, meta=meta)


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Float in [lo, hi). Wrapper for clarity in callsites."""
    return float(rng.uniform(lo, hi))


def generate_dataset(
    seed: int = 42,
    n: int = 50,
    *,
    size: int = DEFAULT_SIZE,
) -> list[SynthSample]:
    """Generate ``n`` deterministic samples — one of each variant in rotation."""
    return [generate_sample(seed, i, size=size) for i in range(n)]


__all__ = ["SynthSample", "Variant", "generate_sample", "generate_dataset", "DEFAULT_SIZE"]
