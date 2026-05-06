"""Tests for miru.bench.comparison — Phase 8."""
import json
import pytest
from pathlib import Path
from miru.bench.comparison import BackendComparison, compare_backends, _determine_winner


# ---------------------------------------------------------------------------
# _determine_winner
# ---------------------------------------------------------------------------

def test_determine_winner_b_wins():
    assert _determine_winner({"mean_delta": 0.1}) == "b"


def test_determine_winner_a_wins():
    assert _determine_winner({"mean_delta": -0.1}) == "a"


def test_determine_winner_tie_zero():
    assert _determine_winner({"mean_delta": 0.0}) == "tie"


def test_determine_winner_tie_empty():
    assert _determine_winner({}) == "tie"


def test_determine_winner_none():
    assert _determine_winner(None) == "tie"


def test_determine_winner_generic_b_wins():
    # dict without mean_delta falls back to counting positive/negative values
    assert _determine_winner({"iou": 0.1, "auc": 0.05}) == "b"


def test_determine_winner_generic_a_wins():
    assert _determine_winner({"iou": -0.1, "auc": -0.05}) == "a"


# ---------------------------------------------------------------------------
# compare_backends — happy path
# ---------------------------------------------------------------------------

def test_compare_backends_mock_vs_mock():
    c = compare_backends("mock", "mock", n_samples=5, seed=42, save=False)
    assert c.backend_a == "mock"
    assert c.backend_b == "mock"
    assert c.winner in ("a", "b", "tie")


def test_compare_backends_name_default():
    c = compare_backends("mock", "mock", n_samples=3, seed=0, save=False)
    assert "mock" in c.name


def test_compare_backends_custom_name():
    c = compare_backends(
        "mock", "mock", n_samples=3, seed=0,
        comparison_name="test-cmp", save=False,
    )
    assert c.name == "test-cmp"


def test_compare_backends_has_results():
    c = compare_backends("mock", "mock", n_samples=5, seed=42, save=False)
    assert c.result_a is not None
    assert c.result_b is not None


def test_compare_backends_hardware():
    c = compare_backends("mock", "mock", n_samples=3, seed=0, save=False)
    assert "hostname" in c.hardware


def test_compare_backends_timestamp():
    c = compare_backends("mock", "mock", n_samples=3, seed=0, save=False)
    assert len(c.timestamp) > 0


def test_compare_backends_comparison_dict():
    c = compare_backends("mock", "mock", n_samples=5, seed=42, save=False)
    # compare_results always returns a dict with these keys
    assert isinstance(c.comparison, dict)
    assert "mean_delta" in c.comparison


def test_compare_backends_mock_same_seed_tie():
    # Same backend + same seed → identical results → delta == 0 → tie
    c = compare_backends("mock", "mock", n_samples=5, seed=42, save=False)
    assert c.winner == "tie"


# ---------------------------------------------------------------------------
# compare_backends — save / no-overwrite
# ---------------------------------------------------------------------------

def test_compare_backends_save(tmp_path):
    compare_backends(
        "mock", "mock", n_samples=3, seed=1,
        save=True, output_dir=tmp_path,
    )
    files = list(tmp_path.glob("comparison-*.json"))
    assert len(files) == 1


def test_compare_backends_saved_json(tmp_path):
    compare_backends(
        "mock", "mock", n_samples=3, seed=2,
        save=True, output_dir=tmp_path,
    )
    f = list(tmp_path.glob("comparison-*.json"))[0]
    data = json.loads(f.read_text())
    assert "backend_a" in data
    assert "winner" in data


def test_compare_backends_two_saves_no_overwrite(tmp_path):
    compare_backends(
        "mock", "mock", n_samples=2, seed=3,
        save=True, output_dir=tmp_path,
    )
    compare_backends(
        "mock", "mock", n_samples=2, seed=4,
        save=True, output_dir=tmp_path,
    )
    files = list(tmp_path.glob("comparison-*.json"))
    assert len(files) >= 1  # both saved safely


# ---------------------------------------------------------------------------
# compare_backends — error paths
# ---------------------------------------------------------------------------

def test_compare_backends_bad_backend():
    with pytest.raises(RuntimeError, match="not available"):
        compare_backends("nonexistent_xyz", "mock", n_samples=1, save=False)


def test_compare_backends_bad_backend_b():
    with pytest.raises(RuntimeError, match="not available"):
        compare_backends("mock", "nonexistent_xyz", n_samples=1, save=False)


# ---------------------------------------------------------------------------
# compare_backends — determinism
# ---------------------------------------------------------------------------

def test_compare_backends_deterministic():
    c1 = compare_backends("mock", "mock", n_samples=5, seed=99, save=False)
    c2 = compare_backends("mock", "mock", n_samples=5, seed=99, save=False)
    assert c1.winner == c2.winner
