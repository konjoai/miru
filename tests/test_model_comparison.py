"""Unit tests for miru/model_comparison.py."""
from __future__ import annotations

from typing import Any

import pytest

from miru.model_comparison import (
    ModelComparisonResult,
    ModelStats,
    compare_models,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_record(
    *,
    analysis_id: str,
    backend: str = "mock",
    method: str = "attention",
    confidence: float = 0.8,
    latency_ms: float = 1.0,
    fidelity_score: float | None = None,
    ts: str = "2026-05-17T12:00:00+00:00",
) -> dict[str, Any]:
    fidelity = ({"fidelity_score": fidelity_score}
                if fidelity_score is not None else None)
    return {
        "analysis_id": analysis_id,
        "ts": ts,
        "question": "q",
        "image_sha256": "abc",
        "trace": {
            "answer": "x",
            "confidence": confidence,
            "backend": backend,
            "method": method,
            "explanation_method": method,
            "latency_ms": latency_ms,
            "attention_grid": [[0.0]],
            "top_regions": [],
            "fidelity": fidelity,
            "cache_hit": False,
        },
    }


@pytest.fixture
def record_dir(tmp_path, monkeypatch):
    """Per-test isolated recorder directory."""
    target = tmp_path / "traces"
    target.mkdir()
    monkeypatch.setenv("MIRU_RECORD", "0")  # don't write — we'll do it ourselves
    monkeypatch.setenv("MIRU_RECORD_PATH", str(target))
    return target


def write_records(record_dir, records: list[dict[str, Any]]) -> None:
    """Write records into the recorder dir as JSONL."""
    import json
    path = record_dir / "traces-test.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ===========================================================================
# Argument validation
# ===========================================================================


def test_rejects_empty_models() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compare_models([])


def test_rejects_duplicate_models() -> None:
    with pytest.raises(ValueError, match="distinct"):
        compare_models(["mock", "mock"])


def test_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        compare_models(["mock"], limit=0)
    with pytest.raises(ValueError, match="limit"):
        compare_models(["mock"], limit=201)


# ===========================================================================
# Empty store → all-None stats
# ===========================================================================


def test_empty_store_returns_none_winners(record_dir) -> None:
    result = compare_models(["mock", "clip"], directory=str(record_dir))
    assert isinstance(result, ModelComparisonResult)
    assert result.models == ["mock", "clip"]
    assert result.winner_by_confidence is None
    assert result.winner_by_fidelity is None
    assert result.winner_by_ece is None
    for st in result.stats.values():
        assert st.n_records == 0
        assert st.mean_confidence is None
        assert st.mean_fidelity is None
        assert st.ece is None


# ===========================================================================
# Single-model aggregate
# ===========================================================================


def test_single_model_basic_aggregate(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id="a", confidence=0.6, latency_ms=2.0),
        make_record(analysis_id="b", confidence=0.8, latency_ms=4.0),
    ])
    result = compare_models(["mock"], directory=str(record_dir))
    st = result.stats["mock"]
    assert st.n_records == 2
    assert st.mean_confidence == pytest.approx(0.7)
    assert st.mean_latency_ms == pytest.approx(3.0)
    assert st.mean_fidelity is None
    assert st.n_with_fidelity == 0
    assert st.ece is None
    assert result.winner_by_confidence == "mock"
    assert result.winner_by_fidelity is None
    assert result.winner_by_ece is None


def test_fidelity_aggregation_and_ece(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id="a", confidence=0.9, fidelity_score=0.9),
        make_record(analysis_id="b", confidence=0.5, fidelity_score=0.5),
    ])
    result = compare_models(["mock"], directory=str(record_dir))
    st = result.stats["mock"]
    assert st.mean_fidelity == pytest.approx(0.7)
    assert st.n_with_fidelity == 2
    # Perfect calibration in each bin → ECE == 0.
    assert st.ece is not None
    assert st.ece == pytest.approx(0.0, abs=1e-9)


def test_method_distribution(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id=f"a{i}", method=m)
        for i, m in enumerate(["attention", "attention", "gradcam", "lime"])
    ])
    result = compare_models(["mock"], directory=str(record_dir))
    st = result.stats["mock"]
    assert st.method_distribution == {"attention": 2, "gradcam": 1, "lime": 1}


# ===========================================================================
# Multi-model winner logic
# ===========================================================================


def test_winner_by_confidence_picks_higher_mean(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id="a", backend="mock", confidence=0.4),
        make_record(analysis_id="b", backend="clip", confidence=0.9),
    ])
    result = compare_models(["mock", "clip"], directory=str(record_dir))
    assert result.winner_by_confidence == "clip"


def test_winner_by_ece_picks_lower(record_dir) -> None:
    write_records(record_dir, [
        # Perfect calibration → ECE = 0
        make_record(analysis_id="a", backend="mock",
                    confidence=0.5, fidelity_score=0.5),
        # Max miscalibration → ECE = 1
        make_record(analysis_id="b", backend="clip",
                    confidence=1.0, fidelity_score=0.0),
    ])
    result = compare_models(["mock", "clip"], directory=str(record_dir))
    assert result.winner_by_ece == "mock"


def test_method_filter_isolates_subset(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id="a", backend="mock", method="attention",
                    confidence=0.9),
        make_record(analysis_id="b", backend="mock", method="gradcam",
                    confidence=0.3),
    ])
    result = compare_models(["mock"], method="attention",
                            directory=str(record_dir))
    assert result.stats["mock"].n_records == 1
    assert result.stats["mock"].mean_confidence == pytest.approx(0.9)
    assert result.filter_method == "attention"


# ===========================================================================
# Edge cases
# ===========================================================================


def test_limit_caps_records_per_model(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id=f"a{i}", confidence=0.5,
                    ts=f"2026-05-{17 - i:02d}T00:00:00+00:00")
        for i in range(5)
    ])
    result = compare_models(["mock"], limit=3, directory=str(record_dir))
    assert result.stats["mock"].n_records == 3
    assert result.limit == 3


def test_models_with_no_data_have_none_winners(record_dir) -> None:
    write_records(record_dir, [
        make_record(analysis_id="a", backend="mock", confidence=0.7),
        # No "clip" records
    ])
    result = compare_models(["mock", "clip"], directory=str(record_dir))
    assert result.winner_by_confidence == "mock"
    assert result.stats["clip"].n_records == 0
    assert result.stats["clip"].mean_confidence is None
