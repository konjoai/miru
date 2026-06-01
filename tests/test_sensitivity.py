"""Unit tests for input sensitivity / robustness analysis."""

import numpy as np
import pytest

from miru.sensitivity import (
    DEFAULT_SIGMAS,
    PerturbationResult,
    SensitivityResult,
    attribution_drift,
    compute_sensitivity,
)


def _image(seed: int, side: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((side, side, 3)).astype(np.float32)


def _blind_saliency(_image_array: np.ndarray) -> np.ndarray:
    """Saliency that ignores the image — trivially robust."""
    g = np.zeros((8, 8), dtype=np.float32)
    g[2:5, 2:5] = 1.0
    return g


def _image_dependent_saliency(image_array: np.ndarray) -> np.ndarray:
    """8×8 block-average of the grayscale image — moves when pixels move."""
    gray = image_array.mean(axis=2)
    h, w = gray.shape
    out = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        r0, r1 = i * h // 8, max(i * h // 8 + 1, (i + 1) * h // 8)
        for j in range(8):
            c0, c1 = j * w // 8, max(j * w // 8 + 1, (j + 1) * w // 8)
            out[i, j] = gray[r0:r1, c0:c1].mean()
    return out


# ---------------------------------------------------------------------------
# attribution_drift
# ---------------------------------------------------------------------------


def test_drift_identical_is_zero() -> None:
    g = _image(1)[:, :, 0]
    assert attribution_drift(g, g) == 0.0


def test_drift_zero_vs_one_is_one() -> None:
    assert attribution_drift(np.zeros((4, 4)), np.ones((4, 4))) == pytest.approx(1.0)


def test_drift_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        attribution_drift(np.zeros((4, 4)), np.zeros((4, 5)))


def test_drift_empty_is_zero() -> None:
    assert attribution_drift(np.zeros((0, 0)), np.zeros((0, 0))) == 0.0


# ---------------------------------------------------------------------------
# compute_sensitivity — image-blind saliency is trivially stable
# ---------------------------------------------------------------------------


def test_blind_saliency_is_perfectly_stable() -> None:
    res = compute_sensitivity(_blind_saliency, _image(1), seed=0)
    assert isinstance(res, SensitivityResult)
    assert res.is_stable is True
    assert res.stability_score == pytest.approx(1.0)
    assert all(p.mean_drift == 0.0 for p in res.per_sigma)


def test_default_sigmas_reported() -> None:
    res = compute_sensitivity(_blind_saliency, _image(1), seed=0)
    assert [p.sigma for p in res.per_sigma] == list(DEFAULT_SIGMAS)
    assert all(isinstance(p, PerturbationResult) for p in res.per_sigma)


def test_baseline_answer_echoed() -> None:
    res = compute_sensitivity(
        _blind_saliency, _image(1), baseline_answer="a cat", seed=0
    )
    assert res.baseline_answer == "a cat"


# ---------------------------------------------------------------------------
# compute_sensitivity — image-dependent saliency drifts
# ---------------------------------------------------------------------------


def test_image_dependent_saliency_drifts() -> None:
    res = compute_sensitivity(_image_dependent_saliency, _image(2), seed=0, n_trials=3)
    assert res.stability_score < 1.0
    assert any(p.mean_drift > 0.0 for p in res.per_sigma)
    assert 0.0 <= res.stability_score <= 1.0


def test_drift_grows_with_noise() -> None:
    res = compute_sensitivity(
        _image_dependent_saliency, _image(2), sigmas=(0.01, 0.2), seed=0, n_trials=5
    )
    low, high = res.per_sigma[0].mean_drift, res.per_sigma[1].mean_drift
    assert high >= low
    assert res.worst_sigma == pytest.approx(0.2)


def test_sensitivity_is_deterministic_under_seed() -> None:
    img = _image(2)
    a = compute_sensitivity(_image_dependent_saliency, img, seed=42, n_trials=3)
    b = compute_sensitivity(_image_dependent_saliency, img, seed=42, n_trials=3)
    assert a.stability_score == b.stability_score
    assert [p.mean_drift for p in a.per_sigma] == [p.mean_drift for p in b.per_sigma]


def test_threshold_controls_verdict() -> None:
    img = _image(2)
    strict = compute_sensitivity(
        _image_dependent_saliency, img, seed=0, stability_threshold=1.0
    )
    lenient = compute_sensitivity(
        _image_dependent_saliency, img, seed=0, stability_threshold=0.0
    )
    assert strict.is_stable is False
    assert lenient.is_stable is True


# ---------------------------------------------------------------------------
# baseline_grid short-circuit avoids a redundant clean-image explainer run
# ---------------------------------------------------------------------------


def test_baseline_grid_avoids_extra_clean_call() -> None:
    calls = {"n": 0}

    def counting_fn(image_array: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return _image_dependent_saliency(image_array)

    base = _image_dependent_saliency(_image(2))
    compute_sensitivity(
        counting_fn,
        _image(2),
        baseline_grid=base,
        sigmas=(0.05,),
        n_trials=2,
        seed=0,
    )
    # Only the 2 perturbation trials call the fn — the clean pass is supplied.
    assert calls["n"] == 2
