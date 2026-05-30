"""Unit tests for miru/posthoc_consensus.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from miru.posthoc_consensus import (
    PosthocConsensusResult,
    build_consensus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_rec(
    analysis_id: str,
    grid: list[list[float]] | np.ndarray,
    *,
    method: str = "attention",
    backend: str = "mock",
    confidence: float = 0.7,
    fidelity_score: float | None = None,
) -> dict[str, Any]:
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    fidelity = ({"fidelity_score": fidelity_score}
                if fidelity_score is not None else None)
    return {
        "analysis_id": analysis_id,
        "ts": "2026-05-17T12:00:00+00:00",
        "trace": {
            "method": method,
            "explanation_method": method,
            "backend": backend,
            "confidence": confidence,
            "attention_grid": grid,
            "fidelity": fidelity,
        },
    }


# ===========================================================================
# Argument validation
# ===========================================================================


def test_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_consensus([])


def test_rejects_unknown_weighting() -> None:
    rec = make_rec("a", [[0.5]])
    with pytest.raises(ValueError, match="weighting"):
        build_consensus([rec], weighting="bogus")  # type: ignore[arg-type]


def test_rejects_invalid_top_k() -> None:
    rec = make_rec("a", [[0.5]])
    with pytest.raises(ValueError, match="top_k"):
        build_consensus([rec], top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        build_consensus([rec], top_k=65)


def test_rejects_record_without_grid() -> None:
    bad = {"analysis_id": "x", "trace": {}}
    with pytest.raises(ValueError, match="attention_grid"):
        build_consensus([bad])


def test_rejects_record_with_empty_grid() -> None:
    bad = {"analysis_id": "x", "trace": {"attention_grid": []}}
    with pytest.raises(ValueError, match="non-empty"):
        build_consensus([bad])


# ===========================================================================
# Math correctness
# ===========================================================================


def test_uniform_weighting_is_simple_mean() -> None:
    rng = np.random.default_rng(0)
    g_a = rng.random((4, 4))
    g_b = rng.random((4, 4))
    result = build_consensus(
        [make_rec("a", g_a), make_rec("b", g_b)],
        weighting="uniform",
    )
    expected = ((g_a + g_b) / 2).tolist()
    consensus = np.asarray(result.consensus_grid)
    np.testing.assert_allclose(consensus, expected, atol=1e-9)


def test_identical_records_have_unit_agreement() -> None:
    """When every record has the same grid, each agrees perfectly."""
    g = np.array([[0.1, 0.5], [0.8, 0.2]])
    result = build_consensus(
        [make_rec("a", g), make_rec("b", g), make_rec("c", g)],
        weighting="uniform",
    )
    for contrib in result.per_record:
        assert contrib.agreement_score == pytest.approx(1.0, abs=1e-9)


def test_outlier_has_lower_agreement() -> None:
    """Two aligned records + one different → outlier gets the lowest score."""
    g_signal = np.zeros((4, 4))
    g_signal[1, 1] = 1.0
    g_outlier = np.zeros((4, 4))
    g_outlier[3, 3] = 1.0
    result = build_consensus(
        [
            make_rec("a", g_signal),
            make_rec("b", g_signal),
            make_rec("c", g_outlier),
        ],
        weighting="uniform",
    )
    agreements = {p.analysis_id: p.agreement_score for p in result.per_record}
    assert agreements["a"] > agreements["c"]
    assert agreements["b"] > agreements["c"]


# ===========================================================================
# Weighting modes
# ===========================================================================


def test_fidelity_weighting_uses_fidelity_scores() -> None:
    """High-fidelity record dominates the consensus."""
    g_high = np.zeros((4, 4))
    g_high[0, 0] = 1.0
    g_low = np.zeros((4, 4))
    g_low[3, 3] = 1.0
    result = build_consensus(
        [
            make_rec("hi", g_high, fidelity_score=0.95),
            make_rec("lo", g_low, fidelity_score=0.05),
        ],
        weighting="fidelity",
    )
    consensus = np.asarray(result.consensus_grid)
    # The (0,0) cell should dominate.
    assert consensus[0, 0] > consensus[3, 3]
    # And the weights should match the input fidelity scores.
    weights = {p.analysis_id: p.weight for p in result.per_record}
    assert weights["hi"] == pytest.approx(0.95)
    assert weights["lo"] == pytest.approx(0.05)


def test_fidelity_weighting_falls_back_to_uniform_when_all_missing() -> None:
    """No record has fidelity → uniform weights, no crash."""
    g = np.array([[0.5]])
    result = build_consensus(
        [make_rec("a", g), make_rec("b", g)],
        weighting="fidelity",
    )
    weights = [p.weight for p in result.per_record]
    assert weights == [1.0, 1.0]
    # The mode echo still says "fidelity" — clients see what happened
    # via the uniform weights.
    assert result.weighting == "fidelity"


def test_fidelity_weighting_uses_floor_for_missing_records() -> None:
    """Records without fidelity get the minimum population fidelity."""
    g_a = np.array([[1.0]])
    g_b = np.array([[1.0]])
    g_c = np.array([[1.0]])
    result = build_consensus(
        [
            make_rec("with", g_a, fidelity_score=0.5),
            make_rec("with2", g_b, fidelity_score=0.9),
            make_rec("without", g_c),  # no fidelity
        ],
        weighting="fidelity",
    )
    weights = {p.analysis_id: p.weight for p in result.per_record}
    assert weights["with"] == pytest.approx(0.5)
    assert weights["with2"] == pytest.approx(0.9)
    # "without" gets the floor (0.5).
    assert weights["without"] == pytest.approx(0.5)


def test_confidence_weighting() -> None:
    g_high = np.zeros((4, 4)); g_high[0, 0] = 1.0
    g_low = np.zeros((4, 4)); g_low[3, 3] = 1.0
    result = build_consensus(
        [
            make_rec("hi", g_high, confidence=0.99),
            make_rec("lo", g_low, confidence=0.01),
        ],
        weighting="confidence",
    )
    consensus = np.asarray(result.consensus_grid)
    assert consensus[0, 0] > consensus[3, 3]


def test_confidence_zero_falls_back_to_uniform() -> None:
    g = np.array([[0.5]])
    result = build_consensus(
        [make_rec("a", g, confidence=0.0), make_rec("b", g, confidence=0.0)],
        weighting="confidence",
    )
    weights = [p.weight for p in result.per_record]
    assert weights == [1.0, 1.0]


# ===========================================================================
# Shape alignment + top regions
# ===========================================================================


def test_mismatched_shapes_get_aligned_to_max() -> None:
    g_small = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2
    g_large = np.zeros((8, 8))                     # 8x8
    g_large[0, 0] = 1.0
    result = build_consensus(
        [make_rec("s", g_small), make_rec("l", g_large)],
        weighting="uniform",
    )
    assert result.grid_h == 8
    assert result.grid_w == 8
    assert len(result.consensus_grid) == 8
    assert len(result.consensus_grid[0]) == 8


def test_top_regions_sorted_by_score_desc() -> None:
    g = np.zeros((4, 4))
    g[0, 0] = 1.0
    g[2, 2] = 0.5
    g[3, 3] = 0.7
    result = build_consensus([make_rec("a", g)], weighting="uniform", top_k=3)
    scores = [r.score for r in result.top_regions]
    assert scores == sorted(scores, reverse=True)
    assert result.top_regions[0].row == 0 and result.top_regions[0].col == 0


def test_metadata_carried_through() -> None:
    g = np.array([[0.1, 0.2]])
    result = build_consensus(
        [make_rec("a", g, method="lime", backend="clip")],
        weighting="uniform",
    )
    assert isinstance(result, PosthocConsensusResult)
    assert result.per_record[0].analysis_id == "a"
    assert result.per_record[0].method == "lime"
    assert result.per_record[0].backend == "clip"
    assert result.n_records == 1
