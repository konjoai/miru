"""Input sensitivity analysis for saliency maps.

How much does an explanation move when the input is barely perturbed?  A
faithful, trustworthy saliency map should be *robust*: small Gaussian noise on
the pixels should not relocate the highlighted regions.  Large drift is a red
flag — the explanation may be tracking noise rather than signal (cf. Ghorbani
et al. 2019, "Interpretation of Neural Networks Is Fragile").

For each noise level σ we re-run the explainer ``n_trials`` times on freshly
perturbed copies of the image and measure the mean absolute drift of the
normalised saliency from the clean baseline.  The aggregate **stability score**
is ``1 − mean_drift`` clamped to ``[0, 1]``: 1.0 = perfectly stable.

This module is method-agnostic: it takes a ``saliency_fn`` that maps an image
array to a 2-D saliency grid, so it works uniformly across every explainer the
API exposes (attention / lime / gradcam / shap).  Everything is seeded, so a
given ``(image, seed)`` yields identical results across runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: A saliency function maps a float32 ``(H, W, 3)`` image to a 2-D grid.
SaliencyFn = Callable[[np.ndarray], np.ndarray]

DEFAULT_SIGMAS: tuple[float, ...] = (0.01, 0.05, 0.1)
_EPS = 1e-12


@dataclass(frozen=True)
class PerturbationResult:
    """Drift statistics at a single noise level."""

    sigma: float
    mean_drift: float
    max_drift: float


@dataclass(frozen=True)
class SensitivityResult:
    """Aggregate robustness verdict for one explanation."""

    baseline_answer: str
    stability_score: float
    is_stable: bool
    worst_sigma: float
    worst_drift: float
    per_sigma: list[PerturbationResult]


def _normalise(heatmap: np.ndarray) -> np.ndarray:
    """Min-max normalise to ``[0, 1]`` so drift is scale-invariant."""
    m = np.asarray(heatmap, dtype=np.float64)
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < _EPS:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def attribution_drift(baseline: np.ndarray, perturbed: np.ndarray) -> float:
    """Mean absolute per-cell difference between two normalised saliency grids.

    Both grids are expected to share a shape (same explainer, same target
    resolution); a shape mismatch is a programming error and raises.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(perturbed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"saliency shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.abs(a - b).mean())


def compute_sensitivity(
    saliency_fn: SaliencyFn,
    image: np.ndarray,
    *,
    baseline_grid: np.ndarray | None = None,
    baseline_answer: str = "",
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    n_trials: int = 3,
    seed: int = 0,
    stability_threshold: float = 0.85,
) -> SensitivityResult:
    """Measure saliency robustness under Gaussian input perturbations.

    Args:
        saliency_fn: Maps an image array to a 2-D saliency grid.
        image: float32 ``(H, W, 3)`` array in ``[0, 1]``.
        baseline_grid: Pre-computed clean-image saliency. When ``None`` it is
            computed via ``saliency_fn(image)`` — pass it to avoid a redundant
            (potentially expensive) explainer run.
        baseline_answer: The model's answer on the clean image (echoed back).
        sigmas: Noise standard deviations to sweep.
        n_trials: Perturbed samples per σ (averaged).
        seed: RNG seed — identical inputs give identical results.
        stability_threshold: ``stability_score`` at/above which ``is_stable``.

    Returns:
        :class:`SensitivityResult` with per-σ drift and an aggregate verdict.
    """
    clean = saliency_fn(image) if baseline_grid is None else baseline_grid
    base_map = _normalise(clean)
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float32)
    per_sigma: list[PerturbationResult] = []
    for sigma in sigmas:
        drifts = _drifts_at_sigma(
            saliency_fn, img, base_map, float(sigma), n_trials, rng
        )
        per_sigma.append(
            PerturbationResult(
                float(sigma), float(np.mean(drifts)), float(np.max(drifts))
            )
        )

    mean_drift = float(np.mean([p.mean_drift for p in per_sigma])) if per_sigma else 0.0
    stability = float(np.clip(1.0 - mean_drift, 0.0, 1.0))
    worst = max(per_sigma, key=lambda p: p.mean_drift) if per_sigma else None
    return SensitivityResult(
        baseline_answer=baseline_answer,
        stability_score=stability,
        is_stable=stability >= stability_threshold,
        worst_sigma=worst.sigma if worst else 0.0,
        worst_drift=worst.mean_drift if worst else 0.0,
        per_sigma=per_sigma,
    )


def _drifts_at_sigma(
    saliency_fn: SaliencyFn,
    image: np.ndarray,
    base_map: np.ndarray,
    sigma: float,
    n_trials: int,
    rng: np.random.Generator,
) -> list[float]:
    """Collect drift values for ``n_trials`` perturbations at one σ."""
    drifts: list[float] = []
    for _ in range(max(1, n_trials)):
        noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
        noisy = np.clip(image + noise, 0.0, 1.0)
        moved = _normalise(saliency_fn(noisy))
        drifts.append(attribution_drift(base_map, moved))
    return drifts
