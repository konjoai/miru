"""Unit tests for miru/diff.py — post-hoc analysis-record diff."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from miru.diff import DiffResult, TopChangedRegion, diff_records


# ---------------------------------------------------------------------------
# Fixtures — synthetic record builder
# ---------------------------------------------------------------------------


def make_rec(
    analysis_id: str,
    grid: list[list[float]] | np.ndarray,
    *,
    method: str = "attention",
    backend: str = "mock",
) -> dict[str, Any]:
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    return {
        "analysis_id": analysis_id,
        "ts": "2026-05-17T12:00:00+00:00",
        "trace": {
            "method": method,
            "explanation_method": method,
            "backend": backend,
            "attention_grid": grid,
        },
    }


# ===========================================================================
# Tests
# ===========================================================================


def test_identical_grids_have_cosine_one_zero_l2() -> None:
    g = np.array([[0.1, 0.4, 0.9], [0.2, 0.5, 1.0], [0.0, 0.3, 0.7]])
    rec_a = make_rec("a", g)
    rec_b = make_rec("b", g)
    res = diff_records(rec_a, rec_b)
    assert isinstance(res, DiffResult)
    assert res.cosine_similarity == pytest.approx(1.0, abs=1e-9)
    assert res.l2_distance == pytest.approx(0.0, abs=1e-9)
    # All deltas are zero, so no entries survive the noise-floor cut.
    assert res.top_changed == []
    assert "identical" in res.summary.lower()


def test_orthogonal_grids_have_cosine_zero() -> None:
    g_a = np.zeros((4, 4))
    g_a[0, 0] = 1.0
    g_b = np.zeros((4, 4))
    g_b[3, 3] = 1.0
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b))
    assert res.cosine_similarity == pytest.approx(0.0, abs=1e-9)
    assert res.l2_distance > 0.0


def test_proportional_grids_have_cosine_one() -> None:
    """g_b = 2 * g_a → same direction, different magnitude → cosine == 1."""
    g_a = np.array([[0.1, 0.4], [0.2, 0.5]])
    g_b = 2.0 * g_a
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b))
    assert res.cosine_similarity == pytest.approx(1.0, abs=1e-9)


def test_delta_grid_shape_matches_input() -> None:
    g_a = np.random.default_rng(0).random((8, 8))
    g_b = np.random.default_rng(1).random((8, 8))
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b))
    assert len(res.delta_grid) == 8
    assert len(res.delta_grid[0]) == 8


def test_different_shapes_get_aligned_to_max() -> None:
    """Smaller grid is upsampled to the larger's shape."""
    g_a = np.random.default_rng(0).random((4, 4))
    g_b = np.random.default_rng(1).random((16, 16))
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b))
    assert len(res.delta_grid) == 16
    assert len(res.delta_grid[0]) == 16


def test_top_changed_sorted_by_abs_delta_desc() -> None:
    rng = np.random.default_rng(42)
    g_a = rng.random((6, 6))
    g_b = rng.random((6, 6))
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b), top_n=8)
    abs_deltas = [abs(t.delta) for t in res.top_changed]
    assert abs_deltas == sorted(abs_deltas, reverse=True)
    assert len(res.top_changed) <= 8


def test_top_changed_carries_both_values() -> None:
    g_a = np.zeros((4, 4))
    g_b = np.zeros((4, 4))
    g_b[2, 1] = 1.0
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b), top_n=3)
    assert res.top_changed[0].row == 2
    assert res.top_changed[0].col == 1
    assert res.top_changed[0].value_a == pytest.approx(0.0)
    assert res.top_changed[0].value_b == pytest.approx(1.0)
    assert res.top_changed[0].delta == pytest.approx(1.0)


def test_summary_mentions_direction_when_shifted() -> None:
    """A focuses bottom-left, B focuses top-right → summary should say so."""
    g_a = np.zeros((9, 9))
    g_a[7:9, 0:2] = 1.0  # bottom-left hot
    g_b = np.zeros((9, 9))
    g_b[0:2, 7:9] = 1.0  # top-right hot
    res = diff_records(make_rec("a", g_a), make_rec("b", g_b))
    s = res.summary.lower()
    assert "bottom" in s and "top" in s


def test_method_and_backend_carried_through() -> None:
    g = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = diff_records(
        make_rec("a", g, method="lime", backend="clip"),
        make_rec("b", g, method="gradcam", backend="mock"),
    )
    assert res.method_a == "lime"
    assert res.method_b == "gradcam"
    assert res.backend_a == "clip"
    assert res.backend_b == "mock"
    assert res.analysis_id_a == "a"
    assert res.analysis_id_b == "b"


def test_missing_attention_grid_raises() -> None:
    rec = {"analysis_id": "x", "trace": {}}
    g = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="attention_grid"):
        diff_records(rec, make_rec("y", g))


def test_malformed_attention_grid_raises() -> None:
    bad = {"analysis_id": "x", "trace": {"attention_grid": []}}
    g = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="2-D"):
        diff_records(bad, make_rec("y", g))


def test_rejects_invalid_top_n() -> None:
    g = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="top_n"):
        diff_records(make_rec("a", g), make_rec("b", g), top_n=0)
    with pytest.raises(ValueError, match="top_n"):
        diff_records(make_rec("a", g), make_rec("b", g), top_n=257)


def test_flat_grids_have_zero_cosine() -> None:
    """Two all-zero grids should not blow up the cosine computation."""
    g = np.zeros((4, 4))
    res = diff_records(make_rec("a", g), make_rec("b", g))
    assert res.cosine_similarity == 0.0
    assert res.l2_distance == 0.0
