"""Tests for miru.bench.profile and miru profile CLI — Phase 9."""
from __future__ import annotations

import json
import io
import pytest
from pathlib import Path

from miru.bench.profile import ProfileResult, _percentile_stats, profile_backend


# ---------------------------------------------------------------------------
# _percentile_stats
# ---------------------------------------------------------------------------


def test_percentile_stats_keys():
    stats = _percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    for key in ("mean", "std", "min", "max", "p50", "p95", "p99", "p999"):
        assert key in stats


def test_percentile_stats_mean():
    stats = _percentile_stats([10.0, 20.0, 30.0])
    assert abs(stats["mean"] - 20.0) < 1e-9


def test_percentile_stats_min_max():
    stats = _percentile_stats([5.0, 2.0, 8.0])
    assert stats["min"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(8.0)


def test_percentile_stats_p50_is_median():
    stats = _percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["p50"] == pytest.approx(3.0)


def test_percentile_stats_single_element():
    stats = _percentile_stats([7.0])
    assert stats["mean"] == pytest.approx(7.0)
    assert stats["p99"] == pytest.approx(7.0)
    assert stats["std"] == pytest.approx(0.0)


def test_percentile_stats_empty():
    stats = _percentile_stats([])
    assert stats["mean"] == 0.0


# ---------------------------------------------------------------------------
# profile_backend — happy path
# ---------------------------------------------------------------------------


def test_profile_backend_returns_profile_result():
    result = profile_backend("mock", n_warmup=1, n_timed=5)
    assert isinstance(result, ProfileResult)


def test_profile_backend_correct_n_timed():
    result = profile_backend("mock", n_warmup=1, n_timed=7)
    assert result.n_timed == 7
    assert len(result.raw_ms) == 7


def test_profile_backend_correct_n_warmup():
    result = profile_backend("mock", n_warmup=2, n_timed=5)
    assert result.n_warmup == 2


def test_profile_backend_raw_ms_positive():
    result = profile_backend("mock", n_warmup=0, n_timed=5)
    assert all(ms >= 0.0 for ms in result.raw_ms)


def test_profile_backend_latency_ms_keys():
    result = profile_backend("mock", n_warmup=1, n_timed=5)
    for key in ("mean", "std", "min", "max", "p50", "p95", "p99", "p999"):
        assert key in result.latency_ms


def test_profile_backend_calls_per_second_positive():
    result = profile_backend("mock", n_warmup=1, n_timed=5)
    assert result.calls_per_second > 0.0


def test_profile_backend_backend_name():
    result = profile_backend("mock", n_warmup=1, n_timed=3)
    assert result.backend == "mock"


def test_profile_backend_timestamp_nonempty():
    result = profile_backend("mock", n_warmup=1, n_timed=3)
    assert len(result.timestamp) > 0


def test_profile_backend_hardware_keys():
    result = profile_backend("mock", n_warmup=1, n_timed=3)
    for key in ("platform", "python", "machine"):
        assert key in result.hardware


def test_profile_backend_image_size_recorded():
    result = profile_backend("mock", n_warmup=0, n_timed=3, image_size=32)
    assert result.image_size == 32


def test_profile_backend_deterministic_raw_ms_length():
    # Two calls with same params should both return n_timed raw_ms entries.
    r1 = profile_backend("mock", n_warmup=1, n_timed=4, seed=7)
    r2 = profile_backend("mock", n_warmup=1, n_timed=4, seed=7)
    assert len(r1.raw_ms) == len(r2.raw_ms) == 4


# ---------------------------------------------------------------------------
# profile_backend — error paths
# ---------------------------------------------------------------------------


def test_profile_backend_bad_backend():
    with pytest.raises(RuntimeError, match="not in registry"):
        profile_backend("nonexistent_xyz", n_warmup=0, n_timed=1)


def test_profile_backend_zero_timed_raises():
    with pytest.raises(ValueError, match="n_timed"):
        profile_backend("mock", n_warmup=1, n_timed=0)


# ---------------------------------------------------------------------------
# ProfileResult.save / to_dict
# ---------------------------------------------------------------------------


def test_profile_result_to_dict_keys():
    result = profile_backend("mock", n_warmup=1, n_timed=3)
    d = result.to_dict()
    for key in ("schema_version", "backend", "timestamp", "n_warmup",
                "n_timed", "image_size", "latency_ms", "calls_per_second",
                "hardware", "raw_ms"):
        assert key in d, f"missing key: {key}"


def test_profile_result_save(tmp_path):
    result = profile_backend("mock", n_warmup=1, n_timed=3)
    path = result.save(output_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["backend"] == "mock"


def test_profile_result_save_no_overwrite(tmp_path):
    r1 = profile_backend("mock", n_warmup=1, n_timed=3)
    r2 = profile_backend("mock", n_warmup=1, n_timed=3)
    # Force same timestamp to test the collision branch.
    r2.timestamp = r1.timestamp
    p1 = r1.save(output_dir=tmp_path)
    p2 = r2.save(output_dir=tmp_path)
    assert p1 != p2  # different filenames
    assert p2.exists()


def test_profile_backend_save_flag(tmp_path):
    profile_backend("mock", n_warmup=1, n_timed=3, save=True, output_dir=tmp_path)
    files = list(tmp_path.glob("profile-mock-*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# CLI — miru profile
# ---------------------------------------------------------------------------


def test_cli_profile_basic():
    from miru.cli import main
    buf = io.StringIO()
    rc = main(["profile", "mock", "--n-timed", "3", "--n-warmup", "1"], stream=buf)
    assert rc == 0
    output = buf.getvalue()
    assert "mock" in output
    assert "calls/s" in output


def test_cli_profile_bad_backend():
    from miru.cli import main
    buf = io.StringIO()
    rc = main(["profile", "nonexistent_xyz", "--n-timed", "1"], stream=buf)
    assert rc == 1
    assert "error" in buf.getvalue().lower()


def test_cli_profile_save(tmp_path):
    from miru.cli import main
    out_file = str(tmp_path / "profile_out.json")
    buf = io.StringIO()
    rc = main(
        ["profile", "mock", "--n-timed", "3", "--n-warmup", "1", "--out", out_file],
        stream=buf,
    )
    assert rc == 0
    p = Path(out_file)
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["backend"] == "mock"


def test_cli_profile_output_contains_percentiles():
    from miru.cli import main
    buf = io.StringIO()
    main(["profile", "mock", "--n-timed", "5"], stream=buf)
    output = buf.getvalue()
    assert "p95" in output
    assert "p99" in output
