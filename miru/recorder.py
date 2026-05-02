"""Async trace recorder — JSONL writer for fine-tuning data pipelines.

Behaviour
---------

When the ``MIRU_RECORD`` environment variable is truthy (``1``/``true``/``yes``),
every reasoning trace produced by ``/analyze`` and ``/analyze/stream`` is
enqueued onto a background writer thread that batches records into JSONL
files inside ``MIRU_RECORD_PATH`` (default ``./miru_traces``).

Privacy
-------

Raw image bytes are **never** persisted.  Each record stores only the
SHA-256 hex digest of the source ``image_b64`` payload under
``image_sha256``, alongside the question, an ISO-8601 UTC timestamp, and
the trace itself with the ``overlay_b64`` field stripped (the overlay is
a derivative of the source image and falls under the same policy).

Storage backend
---------------

If ``MIRU_RECORD_PATH`` contains a URI scheme (``s3://``, ``gs://``, …)
the writer uses :mod:`fsspec` to open the file; otherwise the local
filesystem is used directly with no external dependency.  fsspec is
imported lazily so it remains an optional install.

Files are partitioned per-day (``traces-YYYYMMDD.jsonl``) so individual
files stay tractable for later processing pipelines.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

RECORD_ENV = "MIRU_RECORD"
RECORD_PATH_ENV = "MIRU_RECORD_PATH"
DEFAULT_PATH = "./miru_traces"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_recording_enabled() -> bool:
    """Return True when the MIRU_RECORD env var is set to a truthy value."""
    return os.environ.get(RECORD_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def hash_image(image_b64: str) -> str:
    """SHA-256 hex digest of the base64 image string.

    The hash covers the *encoded* string verbatim — no decoding step — so
    that identical payloads collide regardless of whether they decode to
    valid imagery.  This is sufficient for de-duplication of training data.
    """
    return hashlib.sha256(image_b64.encode("utf-8")).hexdigest()


def build_record(
    trace_dict: dict[str, Any],
    *,
    image_b64: Optional[str],
    question: str,
) -> dict[str, Any]:
    """Build a privacy-stripped record dict ready for JSONL serialization.

    The returned dict contains:

    - ``ts``           — UTC ISO-8601 timestamp (microsecond precision)
    - ``question``     — original question string (verbatim)
    - ``image_sha256`` — hex SHA-256 of the source image_b64, or ``None``
    - ``trace``        — the trace dict with ``overlay_b64`` removed
    """
    stripped_trace = {k: v for k, v in trace_dict.items() if k != "overlay_b64"}
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "image_sha256": hash_image(image_b64) if image_b64 else None,
        "trace": stripped_trace,
    }


# ---------------------------------------------------------------------------
# Filesystem abstraction
# ---------------------------------------------------------------------------


def _is_uri(path: str) -> bool:
    return "://" in path


def _join(base: str, name: str) -> str:
    """Join ``name`` onto ``base`` whether ``base`` is a local path or URI."""
    if _is_uri(base):
        return base.rstrip("/") + "/" + name
    return str(Path(base) / name)


@contextlib.contextmanager
def _open_write(path: str) -> Iterator[Any]:
    """Yield a writable text-mode file handle for *path*.  Local or fsspec.

    Append mode is intentionally avoided: cloud object stores (S3, GCS) do
    not support it, and even fsspec's in-memory backend rejects ``ab`` for
    paths that do not yet exist.  Instead, callers write one file per batch
    using a unique timestamped name so storage semantics are uniform across
    backends.
    """
    if _is_uri(path):
        import fsspec  # type: ignore

        with fsspec.open(path, mode="w", encoding="utf-8") as f:
            yield f
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yield f


def _list_files(directory: str) -> list[str]:
    """List ``traces-*.jsonl`` files in *directory*.  Sorted, fully-qualified."""
    if _is_uri(directory):
        import fsspec  # type: ignore

        fs, _, paths = fsspec.get_fs_token_paths(directory)
        try:
            entries = fs.ls(directory)
        except FileNotFoundError:
            return []
        # fsspec returns either str or dict entries depending on backend.
        names = [e["name"] if isinstance(e, dict) else e for e in entries]
        return sorted(n for n in names if n.endswith(".jsonl") and "traces-" in n)
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("traces-*.jsonl"))


def _read_lines(path: str) -> Iterator[str]:
    if _is_uri(path):
        import fsspec  # type: ignore

        with fsspec.open(path, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Threaded JSONL writer.

    A daemon thread drains an internal :class:`queue.Queue`, batching up
    to ``batch_size`` records per write, and flushes on a ticker every
    ``flush_interval`` seconds.  ``stop()`` signals the thread to drain
    and exit.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        batch_size: int = 64,
        flush_interval: float = 5.0,
    ) -> None:
        self._dir = path or os.environ.get(RECORD_PATH_ENV) or DEFAULT_PATH
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.05, flush_interval)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def directory(self) -> str:
        return self._dir

    def _next_batch_path(self) -> str:
        """Return a unique batch-file path under the recording directory.

        File names sort lexicographically by write time::

            traces-YYYYMMDDTHHMMSS-<microseconds>.jsonl
        """
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S")
        # Microseconds break ties when multiple batches land in the same second.
        return _join(self._dir, f"traces-{stamp}-{now.microsecond:06d}.jsonl")

    def start(self) -> None:
        """Start the background writer thread.  Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="miru-recorder"
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the writer thread to drain and exit, then flush any tail."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Final synchronous drain in case the thread missed records.
        self.flush()

    def enqueue(self, record: dict[str, Any]) -> None:
        """Hand a record to the writer thread (non-blocking)."""
        self._queue.put(record)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._flush_interval)
            try:
                self._flush_batches()
            except Exception:  # noqa: BLE001
                # Recorder must never crash the server; swallow and continue.
                continue
        # Final drain on shutdown
        try:
            self._flush_batches()
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> int:
        """Drain the queue synchronously.  Returns the number of records written."""
        return self._flush_batches()

    def _flush_batches(self) -> int:
        """Drain the queue in batches of ``batch_size``, returning total rows written."""
        total = 0
        while True:
            batch: list[dict[str, Any]] = []
            for _ in range(self._batch_size):
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                break
            self._write_batch(batch)
            total += len(batch)
        return total

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        target = self._next_batch_path()
        with _open_write(target) as f:
            for record in batch:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Singleton accessor (lazy)
# ---------------------------------------------------------------------------


_RECORDER: Optional[TraceRecorder] = None
_RECORDER_LOCK = threading.Lock()


def get_recorder() -> TraceRecorder:
    """Return the process-wide recorder, lazily started on first access."""
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = TraceRecorder()
            _RECORDER.start()
        return _RECORDER


def reset_recorder() -> None:
    """Stop and clear the singleton.  Used by tests and ``stop`` lifecycle."""
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None:
            _RECORDER.stop()
            _RECORDER = None


def maybe_record(
    trace_dict: dict[str, Any],
    *,
    image_b64: Optional[str],
    question: str,
) -> None:
    """Enqueue a record if recording is enabled.  No-op otherwise.

    Errors during enqueue are swallowed — the recorder must never break
    the request path.
    """
    if not is_recording_enabled():
        return
    try:
        record = build_record(trace_dict, image_b64=image_b64, question=question)
        get_recorder().enqueue(record)
    except Exception:  # noqa: BLE001
        # Best-effort recording — never propagate to the caller.
        return


__all__ = [
    "RECORD_ENV",
    "RECORD_PATH_ENV",
    "DEFAULT_PATH",
    "TraceRecorder",
    "is_recording_enabled",
    "hash_image",
    "build_record",
    "get_recorder",
    "reset_recorder",
    "maybe_record",
]
