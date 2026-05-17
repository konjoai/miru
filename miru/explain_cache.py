"""Content-addressed cache for explanation results.

The cache stores serialised :class:`ExplainResponse`-shaped dicts keyed on a
SHA-256 digest of every input that materially affects the saliency map:

- ``image_b64`` (the raw base64 string — equal images collide via hash)
- ``model_name`` (which backend ran)
- ``method`` (attention / lime / gradcam / shap)
- ``question`` (some backends condition attention on it)
- All knobs that change the explanation: ``alpha``, ``colormap``,
  ``top_k``, ``n_samples``, ``n_segments``, ``occlusion_grid``,
  ``shap_grid``, ``shap_samples``, and the ``fidelity`` flag (which
  changes the response shape).

Backend
-------

SQLite at ``MIRU_CACHE_PATH`` (default ``./miru_cache.db``). One short-lived
connection per operation so the cache is safe under FastAPI's threadpool.
Write locks serialise automatically — fine for a single-process demo.

Disabling
---------

Set ``MIRU_CACHE_ENABLED=0`` to bypass entirely.  :func:`get_cache`
returns ``None`` and callers should treat every request as a miss.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


CACHE_PATH_ENV = "MIRU_CACHE_PATH"
CACHE_ENABLED_ENV = "MIRU_CACHE_ENABLED"
DEFAULT_CACHE_PATH = "./miru_cache.db"
SCHEMA_VERSION = 1


def is_cache_enabled() -> bool:
    """Truthy values: ``1`` / ``true`` / ``yes`` / ``on``. Default on."""
    raw = os.environ.get(CACHE_ENABLED_ENV, "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def cache_key(
    image_b64: str,
    method: str,
    model_name: str,
    params: dict[str, Any],
) -> str:
    """SHA-256 hex digest of all output-affecting inputs.

    Params should be a dict containing every keyword that materially
    influences the saliency map (question, alpha, top_k, n_samples,
    n_segments, occlusion_grid, shap_grid, shap_samples, fidelity).
    Unknown extra keys are accepted and included in the hash — so a
    future param addition naturally invalidates old entries.
    """
    digest = hashlib.sha256()
    digest.update(image_b64.encode("utf-8"))
    digest.update(b"|")
    digest.update(method.encode("utf-8"))
    digest.update(b"|")
    digest.update(model_name.encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ExplainCache — SQLite-backed
# ---------------------------------------------------------------------------


class ExplainCache:
    """Thread-safe SQLite-backed cache.

    Args:
        path: Filesystem path to the SQLite database.  The parent
            directory is created if missing.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    # ---- connection helper -----------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # One connection per operation. ``check_same_thread=False`` lets
        # the same connection migrate across the threadpool, but we
        # always close immediately so this is paranoia-safety.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS explanation_cache (
                    key        TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    method     TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    hit_count  INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS cache_meta (
                    name  TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO cache_meta(name, value) VALUES
                    ('schema_version', ?),
                    ('total_hits', 0),
                    ('total_misses', 0);
                """,
                # Use a regular execute for the parameterised INSERT;
                # executescript doesn't bind params.
            )
            conn.execute(
                "INSERT OR IGNORE INTO cache_meta(name, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    # ---- public surface --------------------------------------------

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return the cached payload as a dict, or ``None`` on miss.

        Increments the row's ``hit_count`` and the global ``total_hits``
        on hit; increments ``total_misses`` on miss.  Errors during
        deserialization are treated as misses and the corrupt row is
        deleted so the cache self-heals.
        """
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM explanation_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                self._increment_meta(conn, "total_misses")
                return None
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                # Corrupt entry — purge and treat as miss so we re-populate.
                conn.execute("DELETE FROM explanation_cache WHERE key = ?", (key,))
                self._increment_meta(conn, "total_misses")
                return None
            conn.execute(
                "UPDATE explanation_cache SET hit_count = hit_count + 1 WHERE key = ?",
                (key,),
            )
            self._increment_meta(conn, "total_hits")
            return payload

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        method: str = "",
        model_name: str = "",
    ) -> None:
        """Insert or replace the entry under *key*.

        ``method`` and ``model_name`` are stored alongside the payload
        as columns so operators can run ad-hoc SQL queries (e.g. "purge
        all gradcam entries") without parsing the JSON.
        """
        try:
            payload = json.dumps(value, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.warning("explain_cache: skipping uncacheable payload: %s", exc)
            return
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO explanation_cache(key, payload, method, model_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    payload    = excluded.payload,
                    method     = excluded.method,
                    model_name = excluded.model_name,
                    created_at = excluded.created_at
                """,
                (
                    key,
                    payload,
                    method,
                    model_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def stats(self) -> dict[str, Any]:
        """Return aggregate cache statistics.

        Fields:
            - ``enabled``        — always ``True`` for a live instance
            - ``path``           — absolute path to the SQLite file
            - ``total_entries``  — current row count
            - ``total_hits``     — cumulative since DB creation
            - ``total_misses``   — cumulative since DB creation
            - ``hit_rate``       — ``hits / (hits + misses)``; ``None`` when both are 0
            - ``size_bytes``     — DB file size on disk
            - ``per_method``     — ``{method: count}`` breakdown
        """
        with self._conn() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM explanation_cache").fetchone()
            hits = self._get_meta(conn, "total_hits")
            misses = self._get_meta(conn, "total_misses")
            per_method = {
                row["method"]: row["c"]
                for row in conn.execute(
                    "SELECT method, COUNT(*) AS c FROM explanation_cache GROUP BY method"
                )
            }
        denom = hits + misses
        hit_rate = float(hits) / denom if denom > 0 else None
        return {
            "enabled": True,
            "path": str(self._path.resolve()),
            "total_entries": int(n),
            "total_hits": int(hits),
            "total_misses": int(misses),
            "hit_rate": hit_rate,
            "size_bytes": self._path.stat().st_size if self._path.exists() else 0,
            "per_method": per_method,
        }

    def clear(self) -> int:
        """Drop every entry; reset hit/miss counters.  Returns rows deleted."""
        with self._lock, self._conn() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM explanation_cache").fetchone()
            conn.execute("DELETE FROM explanation_cache")
            conn.execute("UPDATE cache_meta SET value = 0 WHERE name IN ('total_hits', 'total_misses')")
        return int(n)

    # ---- internal helpers ------------------------------------------

    def _increment_meta(self, conn: sqlite3.Connection, name: str) -> None:
        conn.execute(
            "UPDATE cache_meta SET value = value + 1 WHERE name = ?",
            (name,),
        )

    def _get_meta(self, conn: sqlite3.Connection, name: str) -> int:
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE name = ?", (name,)
        ).fetchone()
        return int(row["value"]) if row else 0


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_CACHE: Optional[ExplainCache] = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> Optional[ExplainCache]:
    """Return the singleton cache, or ``None`` when disabled.

    Lazy-instantiated on first call.  Subsequent calls return the same
    instance.  Reads ``MIRU_CACHE_PATH`` and ``MIRU_CACHE_ENABLED`` at
    construction time.
    """
    global _CACHE
    if not is_cache_enabled():
        return None
    with _CACHE_LOCK:
        if _CACHE is None:
            path = os.environ.get(CACHE_PATH_ENV, DEFAULT_CACHE_PATH)
            _CACHE = ExplainCache(path)
        return _CACHE


def reset_cache() -> None:
    """Drop the singleton (test hook).  Does NOT delete the DB file."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


__all__ = [
    "CACHE_PATH_ENV",
    "CACHE_ENABLED_ENV",
    "DEFAULT_CACHE_PATH",
    "SCHEMA_VERSION",
    "ExplainCache",
    "is_cache_enabled",
    "cache_key",
    "get_cache",
    "reset_cache",
]
