"""Unit tests for miru/search.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from miru.search import SearchResult, search_by_pattern


def make_rec(
    analysis_id: str,
    grid: list[list[float]] | np.ndarray,
    *,
    method: str = "attention",
    backend: str = "mock",
    ts: str = "2026-05-17T12:00:00+00:00",
    question: str = "q",
) -> dict[str, Any]:
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    return {
        "analysis_id": analysis_id,
        "ts": ts,
        "question": question,
        "trace": {
            "method": method,
            "explanation_method": method,
            "backend": backend,
            "attention_grid": grid,
        },
    }


# ===========================================================================
# Argument validation
# ===========================================================================


def test_rejects_neither_query_supplied() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        search_by_pattern(source=[])


def test_rejects_both_queries_supplied() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        search_by_pattern(
            query_grid=[[0.5]], query_analysis_id="x", source=[],
        )


def test_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        search_by_pattern(query_grid=[[0.5]], top_k=0, source=[])
    with pytest.raises(ValueError, match="top_k"):
        search_by_pattern(query_grid=[[0.5]], top_k=51, source=[])


def test_rejects_invalid_max_scan() -> None:
    with pytest.raises(ValueError, match="max_scan"):
        search_by_pattern(query_grid=[[0.5]], max_scan=0, source=[])
    with pytest.raises(ValueError, match="max_scan"):
        search_by_pattern(query_grid=[[0.5]], max_scan=2001, source=[])


def test_rejects_invalid_min_similarity() -> None:
    with pytest.raises(ValueError, match="min_similarity"):
        search_by_pattern(query_grid=[[0.5]], min_similarity=1.5, source=[])
    with pytest.raises(ValueError, match="min_similarity"):
        search_by_pattern(query_grid=[[0.5]], min_similarity=-1.5, source=[])


def test_rejects_unknown_query_id() -> None:
    with pytest.raises(ValueError, match="not found"):
        search_by_pattern(query_analysis_id="missing", source=[])


# ===========================================================================
# Basic search behaviour
# ===========================================================================


def test_empty_source_returns_no_matches() -> None:
    result = search_by_pattern(query_grid=[[0.5, 0.5], [0.5, 0.5]], source=[])
    assert isinstance(result, SearchResult)
    assert result.matches == []
    assert result.n_candidates == 0
    assert result.n_scanned == 0


def test_exact_match_scores_one() -> None:
    g = np.array([[0.1, 0.5], [0.8, 0.2]])
    source = [make_rec("x", g)]
    result = search_by_pattern(query_grid=g.tolist(), source=source)
    assert len(result.matches) == 1
    assert result.matches[0].analysis_id == "x"
    assert result.matches[0].similarity == pytest.approx(1.0, abs=1e-9)


def test_matches_sorted_by_similarity_desc() -> None:
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    source = [
        make_rec("near", np.array([[1.0, 0.0], [0.0, 0.1]])),
        make_rec("far", np.array([[0.0, 0.0], [0.0, 1.0]])),
        make_rec("exact", target),
    ]
    result = search_by_pattern(query_grid=target.tolist(), source=source)
    sims = [m.similarity for m in result.matches]
    assert sims == sorted(sims, reverse=True)
    assert result.matches[0].analysis_id == "exact"


def test_query_by_analysis_id_excludes_self() -> None:
    g = np.array([[0.1, 0.5], [0.8, 0.2]])
    source = [
        make_rec("query", g),
        make_rec("other", g),  # identical grid — should still appear
    ]
    result = search_by_pattern(query_analysis_id="query", source=source)
    ids = {m.analysis_id for m in result.matches}
    assert "query" not in ids
    assert "other" in ids


# ===========================================================================
# Filters
# ===========================================================================


def test_method_filter_excludes_non_matching() -> None:
    g = np.array([[1.0, 0.0]])
    source = [
        make_rec("a", g, method="attention"),
        make_rec("b", g, method="gradcam"),
    ]
    result = search_by_pattern(
        query_grid=g.tolist(), method="gradcam", source=source,
    )
    assert [m.analysis_id for m in result.matches] == ["b"]


def test_model_filter_excludes_non_matching() -> None:
    g = np.array([[1.0, 0.0]])
    source = [
        make_rec("a", g, backend="mock"),
        make_rec("b", g, backend="clip"),
    ]
    result = search_by_pattern(
        query_grid=g.tolist(), model="clip", source=source,
    )
    assert [m.analysis_id for m in result.matches] == ["b"]


def test_min_similarity_drops_below_threshold() -> None:
    target = np.array([[1.0, 0.0]])
    source = [
        make_rec("near", np.array([[1.0, 0.01]])),
        make_rec("far",  np.array([[0.0, 1.0]])),
    ]
    result = search_by_pattern(
        query_grid=target.tolist(),
        min_similarity=0.5,
        source=source,
    )
    ids = {m.analysis_id for m in result.matches}
    assert "near" in ids
    assert "far" not in ids


def test_top_k_caps_returned_matches() -> None:
    target = np.array([[1.0]])
    source = [make_rec(f"a{i}", target) for i in range(20)]
    result = search_by_pattern(
        query_grid=target.tolist(), top_k=5, source=source,
    )
    assert len(result.matches) == 5
    assert result.n_candidates == 20


# ===========================================================================
# Shape alignment + edge cases
# ===========================================================================


def test_query_and_candidate_can_differ_in_shape() -> None:
    """8×8 query against 4×4 candidate — bilinear-aligned then scored."""
    query = np.zeros((8, 8))
    query[0:2, 0:2] = 1.0
    candidate = np.zeros((4, 4))
    candidate[0, 0] = 1.0  # roughly the same upper-left activation
    source = [make_rec("c", candidate)]
    result = search_by_pattern(query_grid=query.tolist(), source=source)
    assert len(result.matches) == 1
    assert result.matches[0].similarity > 0.5


def test_skips_candidates_without_attention_grid() -> None:
    """A malformed record (no grid) is silently skipped, not raised."""
    target = np.array([[1.0, 0.0]])
    source = [
        make_rec("good", target),
        {"analysis_id": "bad", "trace": {}},  # no attention_grid
    ]
    result = search_by_pattern(query_grid=target.tolist(), source=source)
    ids = [m.analysis_id for m in result.matches]
    assert ids == ["good"]


def test_max_scan_caps_candidate_count() -> None:
    target = np.array([[1.0]])
    source = [make_rec(f"a{i}", target) for i in range(50)]
    result = search_by_pattern(
        query_grid=target.tolist(),
        max_scan=10,
        source=source,
    )
    assert result.n_candidates == 10
    assert result.n_scanned == 50


def test_match_carries_metadata() -> None:
    g = np.array([[1.0]])
    source = [make_rec("x", g, method="lime", backend="clip",
                       ts="2026-05-18T00:00:00Z",
                       question="what's salient?")]
    result = search_by_pattern(query_grid=g.tolist(), source=source)
    m = result.matches[0]
    assert m.method == "lime"
    assert m.backend == "clip"
    assert m.question == "what's salient?"
    assert m.ts == "2026-05-18T00:00:00Z"
