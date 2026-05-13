"""Tests for the multi-method saliency consensus module."""
from __future__ import annotations

import numpy as np
import pytest

from miru.consensus import compute_consensus


def test_consensus_rejects_single_map() -> None:
    g = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_consensus([("only", g)])


def test_consensus_rejects_top_pct_out_of_range() -> None:
    g = np.ones((4, 4), dtype=np.float32)
    h = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_consensus([("a", g), ("b", h)], top_pct=0.0)
    with pytest.raises(ValueError):
        compute_consensus([("a", g), ("b", h)], top_pct=1.0)


def test_identical_maps_perfect_consensus() -> None:
    """Two identical maps must produce Jaccard 1.0 and no disagreement."""
    g = np.zeros((8, 8), dtype=np.float32)
    g[2:5, 2:5] = 1.0
    r = compute_consensus([("a", g), ("b", g.copy())], top_pct=0.25)
    assert r.consensus_score == pytest.approx(1.0, abs=1e-9)
    assert r.pairwise_jaccard["a|b"] == pytest.approx(1.0, abs=1e-9)
    assert r.disagreement_regions == []
    # Agreement grid: every cell value is 1.0 or 0.0 (both methods agree).
    unique_values = set(np.round(r.agreement_grid, 6).flatten().tolist())
    assert unique_values <= {0.0, 1.0}


def test_disjoint_maps_zero_consensus() -> None:
    """Two maps with disjoint top-pct regions ⇒ Jaccard 0.0 and all disagree."""
    g = np.zeros((8, 8), dtype=np.float32)
    h = np.zeros((8, 8), dtype=np.float32)
    g[0:2, 0:2] = 1.0  # top-left corner = top 4 of 64 cells (6.25%)
    h[6:8, 6:8] = 1.0  # bottom-right corner, same size
    r = compute_consensus([("a", g), ("b", h)], top_pct=0.0625)
    assert r.consensus_score == 0.0
    assert r.pairwise_jaccard["a|b"] == 0.0
    # Every cell in the union of the two top-pct sets is in exactly one mask.
    assert len(r.disagreement_regions) == 8  # 4 from each side


def test_three_method_agreement_grid_is_average_membership() -> None:
    """With 3 methods and one shared cell, that cell scores 1.0 in agreement."""
    R = 8
    a = np.zeros((R, R), dtype=np.float32); a[0, 0] = 1.0
    b = np.zeros((R, R), dtype=np.float32); b[0, 0] = 1.0
    c = np.zeros((R, R), dtype=np.float32); c[0, 0] = 1.0
    # top_pct = 1/64 → exactly 1 cell per mask, all the same cell.
    r = compute_consensus(
        [("a", a), ("b", b), ("c", c)], top_pct=1 / 64
    )
    assert r.agreement_grid[0, 0] == pytest.approx(1.0)
    assert r.consensus_score == pytest.approx(1.0)
    # 3 unordered pairs across 3 methods.
    assert set(r.pairwise_jaccard.keys()) == {"a|b", "a|c", "b|c"}


def test_disagreement_regions_sorted_by_summed_saliency() -> None:
    """Disagreement list ranks most-disputed cells first.

    Both maps have one shared cell (1,1) — included in each top-pct, so
    NOT a disagreement. The exclusive cells (2,2) and (3,3) are; (2,2)
    has higher summed saliency so it appears first.
    """
    R = 8
    a = np.zeros((R, R), dtype=np.float32)
    b = np.zeros((R, R), dtype=np.float32)
    a[1, 1] = 0.3
    a[2, 2] = 0.9    # exclusive to a
    b[1, 1] = 0.3    # shared cell — agrees
    b[3, 3] = 0.4    # exclusive to b
    # top_pct = 2 / 64 = 0.03125 → exactly 2 cells per mask.
    r = compute_consensus([("a", a), ("b", b)], top_pct=2 / 64)
    assert (2, 2) in r.disagreement_regions
    assert (3, 3) in r.disagreement_regions
    assert (1, 1) not in r.disagreement_regions  # shared
    assert r.disagreement_regions[0] == (2, 2)


def test_consensus_resamples_to_common_resolution() -> None:
    """Maps of different resolutions still compare cleanly."""
    a = np.zeros((8, 8), dtype=np.float32); a[0, 0] = 1.0
    b = np.zeros((4, 4), dtype=np.float32); b[0, 0] = 1.0
    r = compute_consensus([("a", a), ("b", b)], top_pct=0.10, resolution=8)
    assert r.agreement_grid.shape == (8, 8)
    assert r.consensus_score > 0.0


def test_method_names_preserved() -> None:
    g = np.ones((4, 4), dtype=np.float32)
    r = compute_consensus([("attention", g), ("lime", g)])
    assert r.method_names == ["attention", "lime"]
