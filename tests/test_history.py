"""Unit tests for miru/history.py — query_records + compute_calibration."""
from __future__ import annotations

from typing import Any

import pytest

from miru.history import (
    CalibrationResult,
    HistoryPage,
    HistoryRecord,
    compute_calibration,
    query_records,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic record builder
# ---------------------------------------------------------------------------


def make_record(
    *,
    analysis_id: str = "id-0",
    ts: str = "2026-05-17T12:00:00+00:00",
    method: str = "attention",
    backend: str = "mock",
    confidence: float = 0.8,
    fidelity_score: float | None = None,
    cache_hit: bool = False,
    question: str = "q",
    image_sha256: str | None = "abc",
    attention_grid: list[list[float]] | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "answer": "x",
        "confidence": confidence,
        "backend": backend,
        "method": method,
        "explanation_method": method,
        "latency_ms": 1.0,
        "attention_grid": attention_grid or [[0.0]],
        "top_regions": [],
        "fidelity": (
            {"fidelity_score": fidelity_score} if fidelity_score is not None else None
        ),
        "cache_hit": cache_hit,
    }
    return {
        "analysis_id": analysis_id,
        "ts": ts,
        "question": question,
        "image_sha256": image_sha256,
        "trace": trace,
    }


# ===========================================================================
# query_records — filtering, ordering, pagination
# ===========================================================================


def test_query_empty_source_returns_empty_page() -> None:
    page = query_records(source=[], limit=10)
    assert isinstance(page, HistoryPage)
    assert page.items == []
    assert page.total == 0
    assert page.limit == 10
    assert page.offset == 0


def test_query_basic_round_trip() -> None:
    recs = [make_record(analysis_id="x")]
    page = query_records(source=recs)
    assert page.total == 1
    assert len(page.items) == 1
    item = page.items[0]
    assert isinstance(item, HistoryRecord)
    assert item.analysis_id == "x"
    assert item.method == "attention"
    assert item.backend == "mock"


def test_query_filter_by_method() -> None:
    recs = [
        make_record(analysis_id="a", method="attention"),
        make_record(analysis_id="b", method="gradcam"),
        make_record(analysis_id="c", method="lime"),
    ]
    page = query_records(source=recs, method="gradcam")
    assert page.total == 1
    assert page.items[0].analysis_id == "b"


def test_query_filter_by_model() -> None:
    recs = [
        make_record(analysis_id="a", backend="mock"),
        make_record(analysis_id="b", backend="clip"),
    ]
    page = query_records(source=recs, model="clip")
    assert [item.analysis_id for item in page.items] == ["b"]


def test_query_filter_by_min_confidence() -> None:
    recs = [
        make_record(analysis_id="lo", confidence=0.3),
        make_record(analysis_id="mid", confidence=0.6),
        make_record(analysis_id="hi", confidence=0.9),
    ]
    page = query_records(source=recs, min_confidence=0.5)
    assert sorted(item.analysis_id for item in page.items) == ["hi", "mid"]


def test_query_filter_by_since() -> None:
    recs = [
        make_record(analysis_id="old", ts="2026-01-01T00:00:00+00:00"),
        make_record(analysis_id="new", ts="2026-12-01T00:00:00+00:00"),
    ]
    page = query_records(source=recs, since="2026-06-01T00:00:00+00:00")
    assert [item.analysis_id for item in page.items] == ["new"]


def test_query_orders_newest_first() -> None:
    recs = [
        make_record(analysis_id="middle", ts="2026-06-01T00:00:00+00:00"),
        make_record(analysis_id="oldest", ts="2026-01-01T00:00:00+00:00"),
        make_record(analysis_id="newest", ts="2026-12-01T00:00:00+00:00"),
    ]
    page = query_records(source=recs)
    assert [item.analysis_id for item in page.items] == ["newest", "middle", "oldest"]


def test_query_pagination_limit_offset() -> None:
    recs = [
        make_record(
            analysis_id=f"id-{i}",
            ts=f"2026-05-{17 - i:02d}T00:00:00+00:00",
        )
        for i in range(5)
    ]
    page = query_records(source=recs, limit=2, offset=1)
    # Sorted newest-first → ids: id-0, id-1, id-2, id-3, id-4
    assert [item.analysis_id for item in page.items] == ["id-1", "id-2"]
    assert page.total == 5
    assert page.limit == 2
    assert page.offset == 1


def test_query_combined_filters() -> None:
    recs = [
        make_record(analysis_id="a", method="attention", backend="mock", confidence=0.4),
        make_record(analysis_id="b", method="attention", backend="mock", confidence=0.9),
        make_record(analysis_id="c", method="gradcam", backend="mock", confidence=0.9),
    ]
    page = query_records(
        source=recs, method="attention", model="mock", min_confidence=0.5,
    )
    assert [item.analysis_id for item in page.items] == ["b"]


def test_query_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        query_records(source=[], limit=0)
    with pytest.raises(ValueError, match="limit"):
        query_records(source=[], limit=201)


def test_query_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="offset"):
        query_records(source=[], offset=-1)


def test_query_invalid_since_treated_as_no_filter() -> None:
    """A bad timestamp shouldn't crash — just disable the since filter."""
    recs = [make_record(analysis_id="x")]
    page = query_records(source=recs, since="not-a-timestamp")
    assert page.total == 1


def test_query_extracts_fidelity_score() -> None:
    recs = [make_record(analysis_id="x", fidelity_score=0.72)]
    page = query_records(source=recs)
    assert page.items[0].fidelity_score == pytest.approx(0.72)


def test_query_handles_missing_fidelity_block() -> None:
    recs = [make_record(analysis_id="x")]  # no fidelity
    page = query_records(source=recs)
    assert page.items[0].fidelity_score is None


# ===========================================================================
# compute_calibration — ECE math
# ===========================================================================


def test_calibration_empty_records_returns_zero_ece() -> None:
    res = compute_calibration([], n_bins=10)
    assert isinstance(res, CalibrationResult)
    assert res.n == 0
    assert res.ece == 0.0
    assert len(res.bins) == 10
    assert all(b.count == 0 for b in res.bins)


def test_calibration_records_without_fidelity_skipped() -> None:
    recs = [make_record(analysis_id=f"x{i}", fidelity_score=None) for i in range(5)]
    res = compute_calibration(recs)
    assert res.n == 0


def test_calibration_perfect_means_zero_ece() -> None:
    """conf == fidelity for every record → ECE = 0."""
    recs = [
        make_record(analysis_id=f"x{i}", confidence=c, fidelity_score=c)
        for i, c in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    res = compute_calibration(recs, n_bins=10)
    assert res.n == 5
    assert res.ece == pytest.approx(0.0, abs=1e-9)


def test_calibration_maximum_miscalibration() -> None:
    """All records have conf=1 but fidelity=0 → ECE = 1.0."""
    recs = [
        make_record(analysis_id=f"x{i}", confidence=1.0, fidelity_score=0.0)
        for i in range(10)
    ]
    res = compute_calibration(recs, n_bins=10)
    assert res.ece == pytest.approx(1.0, abs=1e-9)


def test_calibration_bin_edges_cover_unit_interval() -> None:
    res = compute_calibration([], n_bins=5)
    assert res.bins[0].lo == 0.0
    assert res.bins[-1].hi == pytest.approx(1.0)
    # Bins are contiguous, no gaps.
    for i in range(len(res.bins) - 1):
        assert res.bins[i].hi == pytest.approx(res.bins[i + 1].lo)


def test_calibration_records_in_correct_bin() -> None:
    recs = [
        make_record(analysis_id="a", confidence=0.05, fidelity_score=0.05),
        make_record(analysis_id="b", confidence=0.55, fidelity_score=0.55),
        make_record(analysis_id="c", confidence=1.0,  fidelity_score=1.0),
    ]
    res = compute_calibration(recs, n_bins=10)
    counts = [b.count for b in res.bins]
    assert counts[0] == 1   # 0.05 → bin 0
    assert counts[5] == 1   # 0.55 → bin 5
    assert counts[9] == 1   # 1.00 → last bin (closed upper edge)
    assert sum(counts) == 3


def test_calibration_rejects_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration([], n_bins=1)
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration([], n_bins=51)


def test_calibration_clamps_out_of_range_values() -> None:
    """Confidence > 1 or fidelity < 0 from a corrupt record should clamp, not crash."""
    recs = [make_record(analysis_id="x", confidence=1.5, fidelity_score=-0.2)]
    res = compute_calibration(recs)
    assert res.n == 1
    # Clamped to [0, 1], so the gap = |1 - 0| = 1.
    assert res.ece == pytest.approx(1.0)
