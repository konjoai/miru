"""Unit tests for miru/explain_cache.py."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from miru.explain_cache import (
    CACHE_ENABLED_ENV,
    CACHE_PATH_ENV,
    ExplainCache,
    cache_key,
    get_cache,
    is_cache_enabled,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> ExplainCache:
    return ExplainCache(str(tmp_path / "test.db"))


@pytest.fixture
def env_cache(tmp_path: Path, monkeypatch) -> ExplainCache:
    """Fresh singleton wired to a per-test SQLite file."""
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "singleton.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "1")
    reset_cache()
    inst = get_cache()
    assert inst is not None
    yield inst
    reset_cache()


# ---------------------------------------------------------------------------
# cache_key — determinism + sensitivity
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    a = cache_key("img", "attention", "mock", {"alpha": 0.5, "top_k": 5})
    b = cache_key("img", "attention", "mock", {"top_k": 5, "alpha": 0.5})  # different dict order
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_cache_key_changes_with_method() -> None:
    a = cache_key("img", "attention", "mock", {})
    b = cache_key("img", "gradcam", "mock", {})
    assert a != b


def test_cache_key_changes_with_image() -> None:
    a = cache_key("img1", "attention", "mock", {})
    b = cache_key("img2", "attention", "mock", {})
    assert a != b


def test_cache_key_changes_with_param() -> None:
    a = cache_key("img", "attention", "mock", {"alpha": 0.5})
    b = cache_key("img", "attention", "mock", {"alpha": 0.6})
    assert a != b


def test_cache_key_changes_with_fidelity_flag() -> None:
    a = cache_key("img", "attention", "mock", {"fidelity": False})
    b = cache_key("img", "attention", "mock", {"fidelity": True})
    assert a != b


# ---------------------------------------------------------------------------
# Get / put / round-trip
# ---------------------------------------------------------------------------


def test_get_miss_returns_none(cache: ExplainCache) -> None:
    assert cache.get("nonexistent") is None


def test_put_then_get_round_trip(cache: ExplainCache) -> None:
    payload = {"answer": "yes", "confidence": 0.91, "grid": [[1.0, 2.0]]}
    cache.put("key1", payload, method="attention", model_name="mock")
    out = cache.get("key1")
    assert out == payload


def test_put_replaces_on_same_key(cache: ExplainCache) -> None:
    cache.put("key", {"v": 1}, method="attention", model_name="mock")
    cache.put("key", {"v": 2}, method="attention", model_name="mock")
    assert cache.get("key") == {"v": 2}


def test_get_increments_hit_count(cache: ExplainCache) -> None:
    cache.put("k", {"v": 1}, method="attention", model_name="mock")
    cache.get("k"); cache.get("k"); cache.get("k")
    # Inspect via SQL — hit_count column.
    with sqlite3.connect(cache.path) as conn:
        (n,) = conn.execute(
            "SELECT hit_count FROM explanation_cache WHERE key = 'k'"
        ).fetchone()
    assert n == 3


def test_corrupt_entry_self_heals(cache: ExplainCache) -> None:
    """A malformed JSON payload is treated as a miss and deleted."""
    cache.put("k", {"v": 1}, method="attention", model_name="mock")
    # Manually corrupt the row.
    with sqlite3.connect(cache.path) as conn:
        conn.execute("UPDATE explanation_cache SET payload = 'not-json' WHERE key = 'k'")
        conn.commit()
    assert cache.get("k") is None
    # And the corrupt row was purged.
    with sqlite3.connect(cache.path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM explanation_cache WHERE key='k'").fetchone()
    assert n == 0


def test_put_skips_uncacheable_payload(cache: ExplainCache, caplog) -> None:
    """A non-JSON-serialisable payload triggers a warning, not a crash."""
    class NotSerialisable:
        pass

    cache.put("k", {"obj": NotSerialisable()}, method="attention", model_name="mock")
    assert cache.get("k") is None


# ---------------------------------------------------------------------------
# Stats + clear
# ---------------------------------------------------------------------------


def test_stats_initial_state(cache: ExplainCache) -> None:
    s = cache.stats()
    assert s["enabled"] is True
    assert s["total_entries"] == 0
    assert s["total_hits"] == 0
    assert s["total_misses"] == 0
    assert s["hit_rate"] is None


def test_stats_after_traffic(cache: ExplainCache) -> None:
    cache.put("a", {"v": 1}, method="attention", model_name="mock")
    cache.put("b", {"v": 2}, method="gradcam", model_name="mock")
    cache.get("a"); cache.get("a")          # 2 hits
    cache.get("missing")                     # 1 miss
    s = cache.stats()
    assert s["total_entries"] == 2
    assert s["total_hits"] == 2
    assert s["total_misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-9)
    assert s["per_method"] == {"attention": 1, "gradcam": 1}
    assert s["size_bytes"] > 0


def test_clear_drops_all_rows(cache: ExplainCache) -> None:
    cache.put("a", {"v": 1}, method="attention", model_name="mock")
    cache.put("b", {"v": 2}, method="lime", model_name="mock")
    cache.get("a"); cache.get("missing")
    cleared = cache.clear()
    assert cleared == 2
    assert cache.stats()["total_entries"] == 0
    assert cache.stats()["total_hits"] == 0
    assert cache.stats()["total_misses"] == 0


# ---------------------------------------------------------------------------
# Singleton + env gating
# ---------------------------------------------------------------------------


def test_is_cache_enabled_truthy(monkeypatch) -> None:
    for v in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv(CACHE_ENABLED_ENV, v)
        assert is_cache_enabled() is True


def test_is_cache_enabled_falsy(monkeypatch) -> None:
    for v in ("0", "false", "no", "off"):
        monkeypatch.setenv(CACHE_ENABLED_ENV, v)
        assert is_cache_enabled() is False


def test_get_cache_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_ENABLED_ENV, "0")
    reset_cache()
    try:
        assert get_cache() is None
    finally:
        reset_cache()


def test_get_cache_singleton(env_cache: ExplainCache) -> None:
    again = get_cache()
    assert again is env_cache


def test_reset_cache_creates_fresh_instance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "x.db"))
    monkeypatch.setenv(CACHE_ENABLED_ENV, "1")
    reset_cache()
    a = get_cache()
    reset_cache()
    b = get_cache()
    try:
        assert a is not b
    finally:
        reset_cache()
