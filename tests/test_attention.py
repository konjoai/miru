"""Unit tests for AttentionExtractor."""
import numpy as np
import pytest

from miru.attention.extractor import AttentionExtractor


@pytest.fixture
def extractor() -> AttentionExtractor:
    return AttentionExtractor(resolution=16)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_uniform(extractor: AttentionExtractor) -> None:
    """A uniform input array should produce an all-zeros output."""
    arr = np.full((4, 4), 0.5, dtype=np.float32)
    result = extractor.normalize(arr)
    np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.float32))


def test_normalize_range(extractor: AttentionExtractor) -> None:
    """Output must always be in [0, 1]."""
    rng = np.random.default_rng(0)
    arr = rng.random((8, 8)).astype(np.float32) * 100.0
    result = extractor.normalize(arr)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0 + 1e-6


def test_normalize_preserves_order(extractor: AttentionExtractor) -> None:
    """Higher input values must produce higher output values."""
    arr = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    result = extractor.normalize(arr)
    # global max input → output == 1.0
    assert result[1, 1] == pytest.approx(1.0)
    # global min input → output == 0.0
    assert result[0, 0] == pytest.approx(0.0)
    # intermediate values preserve relative order
    assert result[0, 1] > result[1, 0]


# ---------------------------------------------------------------------------
# resize_to_grid()
# ---------------------------------------------------------------------------


def test_resize_to_grid(extractor: AttentionExtractor) -> None:
    """Output shape must match requested (target_h, target_w)."""
    arr = np.ones((32, 32), dtype=np.float32)
    result = extractor.resize_to_grid(arr, 8, 8)
    assert result.shape == (8, 8)


def test_resize_preserves_constant(extractor: AttentionExtractor) -> None:
    """Resizing a constant array must return the same constant."""
    arr = np.full((32, 32), 0.75, dtype=np.float32)
    result = extractor.resize_to_grid(arr, 8, 8)
    np.testing.assert_allclose(result, 0.75, atol=1e-5)


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


def test_extract_output_shape(extractor: AttentionExtractor) -> None:
    """extract() must always return (resolution, resolution)."""
    rng = np.random.default_rng(1)
    for shape in [(16, 16), (32, 32), (8, 8), (3, 5)]:
        arr = rng.random(shape).astype(np.float32)
        result = extractor.extract(arr)
        assert result.shape == (16, 16), f"Shape {shape} → {result.shape}, expected (16, 16)"


def test_extract_output_dtype(extractor: AttentionExtractor) -> None:
    arr = np.random.default_rng(2).random((16, 16)).astype(np.float64)
    result = extractor.extract(arr)
    assert result.dtype == np.float32


def test_extract_range(extractor: AttentionExtractor) -> None:
    arr = np.random.default_rng(3).random((16, 16)).astype(np.float32) * 50.0
    result = extractor.extract(arr)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# top_k_regions()
# ---------------------------------------------------------------------------


def test_top_k_regions_count(extractor: AttentionExtractor) -> None:
    """top_k_regions must return exactly k results."""
    arr = np.random.default_rng(4).random((16, 16)).astype(np.float32)
    results = extractor.top_k_regions(arr, k=3)
    assert len(results) == 3


def test_top_k_regions_sorted(extractor: AttentionExtractor) -> None:
    """Results must be sorted by score in descending order."""
    rng = np.random.default_rng(5)
    arr = rng.random((16, 16)).astype(np.float32)
    results = extractor.top_k_regions(arr, k=5)
    scores = [s for _, _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_highest_value(extractor: AttentionExtractor) -> None:
    """The first result must correspond to the global maximum of the map."""
    arr = np.zeros((16, 16), dtype=np.float32)
    arr[7, 11] = 1.0  # known global max
    results = extractor.top_k_regions(arr, k=3)
    best_row, best_col, best_score = results[0]
    assert (best_row, best_col) == (7, 11)
    assert best_score == pytest.approx(1.0)


def test_top_k_zero_k(extractor: AttentionExtractor) -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    assert extractor.top_k_regions(arr, k=0) == []
