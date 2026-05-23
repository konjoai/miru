"""Filtered query + calibration over the recorded explanation store.

The recorder writes one privacy-stripped JSONL record per ``/explain``
call.  This module is the read side: an indexed (well, scanned)
query layer plus a calibration aggregator that turns recorded
confidence / fidelity pairs into an Expected Calibration Error.

Public surface
--------------

- :func:`query_records` — filtered + paginated newest-first listing.
- :func:`compute_calibration` — Expected Calibration Error + per-bin
  reliability curve from records that carry a fidelity score.

Both functions take an iterable of record dicts so callers can plug in
their own source (test fixtures, in-memory caches, etc.).  The
file-scanning entry point :func:`load_records` is a thin shim around
:mod:`miru.recorder` so the rest of the module stays pure.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from miru.recorder import DEFAULT_PATH, RECORD_PATH_ENV, _list_files, _read_lines


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(directory: Optional[str] = None) -> Iterator[dict[str, Any]]:
    """Yield every recorded JSONL record from ``directory`` (or env default).

    Drains the singleton recorder's in-memory queue before scanning so
    records produced microseconds ago by ``maybe_record`` are visible
    here. Without this, callers that record-then-immediately-list would
    miss recent rows until the recorder's periodic flush — the same
    issue ``find_record_by_id`` fixes for lookups.

    Caller-deferred filtering keeps this generator cheap on large stores.
    Corrupt lines are skipped silently — the recorder's own write path
    never produces them, so they only appear when an operator hand-edits
    the JSONL.
    """
    # Best-effort drain: the singleton may not exist yet (returns None
    # from the lazy accessor); when it does, flush failures must not
    # block reads.
    from miru.recorder import _RECORDER, _RECORDER_LOCK

    with _RECORDER_LOCK:
        rec = _RECORDER
    if rec is not None:
        try:
            rec.flush()
        except (OSError, ValueError):
            pass

    base = directory or os.environ.get(RECORD_PATH_ENV) or DEFAULT_PATH
    for path in _list_files(base):
        for line in _read_lines(path):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Record summary — the public shape for /explain/history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRecord:
    """Lightweight summary of a recorded explanation.

    Drops the bulky ``attention_grid`` and ``top_regions`` so history
    responses stay light — clients that need the full trace can fetch
    ``/analysis/{id}/export?format=json``.
    """

    analysis_id: str
    ts: str
    question: str
    image_sha256: Optional[str]
    backend: str
    method: str
    confidence: float
    latency_ms: float
    fidelity_score: Optional[float]
    cache_hit: bool

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "HistoryRecord":
        trace = record.get("trace") or {}
        fidelity = trace.get("fidelity") or {}
        return cls(
            analysis_id=str(record.get("analysis_id") or ""),
            ts=str(record.get("ts") or ""),
            question=str(record.get("question") or ""),
            image_sha256=record.get("image_sha256"),
            backend=str(trace.get("backend") or ""),
            method=str(trace.get("method") or trace.get("explanation_method") or ""),
            confidence=float(trace.get("confidence") or 0.0),
            latency_ms=float(trace.get("latency_ms") or 0.0),
            fidelity_score=(
                float(fidelity["fidelity_score"])
                if isinstance(fidelity, dict) and "fidelity_score" in fidelity
                else None
            ),
            cache_hit=bool(trace.get("cache_hit", False)),
        )


# ---------------------------------------------------------------------------
# Filtering + pagination
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> Optional[datetime]:
    """Best-effort ISO-8601 parse — used for time filtering only."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _matches(
    record: dict[str, Any],
    *,
    method: Optional[str],
    model: Optional[str],
    min_confidence: Optional[float],
    since: Optional[datetime],
) -> bool:
    trace = record.get("trace") or {}
    if method is not None:
        rec_method = trace.get("method") or trace.get("explanation_method")
        if rec_method != method:
            return False
    if model is not None and trace.get("backend") != model:
        return False
    if min_confidence is not None:
        conf = trace.get("confidence")
        if conf is None or float(conf) < min_confidence:
            return False
    if since is not None:
        rec_ts = _parse_iso(str(record.get("ts") or ""))
        if rec_ts is None or rec_ts < since:
            return False
    return True


@dataclass(frozen=True)
class HistoryPage:
    """One page of :func:`query_records` results."""

    items: list[HistoryRecord]
    total: int          # total records matching the filter (across all pages)
    limit: int
    offset: int


def query_records(
    *,
    directory: Optional[str] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    min_confidence: Optional[float] = None,
    since: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    source: Optional[Iterable[dict[str, Any]]] = None,
) -> HistoryPage:
    """Filter + paginate the recorded explanation store, newest first.

    Args:
        directory: Recorder path override (defaults to ``MIRU_RECORD_PATH``).
            Ignored when *source* is supplied.
        method: Exact-match explanation method filter.
        model: Exact-match backend / model filter.
        min_confidence: Lower bound on ``trace.confidence``.
        since: ISO-8601 timestamp; records strictly older are excluded.
            Invalid timestamps are silently dropped (no filter applied).
        limit: Maximum records to return.  Must be ``1..200``.
        offset: Number of matching records to skip before the page starts.
        source: Optional iterable of pre-loaded records (tests).

    Returns:
        A :class:`HistoryPage` whose ``items`` are the requested slice
        in newest-first order, and whose ``total`` is the full count of
        records matching the filter (useful for pagination UI).
    """
    if not 1 <= limit <= 200:
        raise ValueError(f"limit must be in 1..200, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    since_dt = _parse_iso(since) if since else None
    records_iter: Iterable[dict[str, Any]] = source if source is not None else load_records(directory)

    matched: list[tuple[datetime, dict[str, Any]]] = []
    for record in records_iter:
        if not _matches(
            record,
            method=method,
            model=model,
            min_confidence=min_confidence,
            since=since_dt,
        ):
            continue
        ts = _parse_iso(str(record.get("ts") or "")) or datetime.min.replace(tzinfo=timezone.utc)
        matched.append((ts, record))

    matched.sort(key=lambda pair: pair[0], reverse=True)
    page = matched[offset : offset + limit]
    return HistoryPage(
        items=[HistoryRecord.from_record(rec) for _, rec in page],
        total=len(matched),
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# Calibration — Expected Calibration Error + reliability curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    """One bucket in the reliability curve."""

    lo: float                # bin lower edge (inclusive)
    hi: float                # bin upper edge (exclusive, except the last bin)
    count: int
    mean_confidence: float   # 0 when bin is empty
    mean_fidelity: float     # 0 when bin is empty
    gap: float               # |mean_confidence - mean_fidelity|


@dataclass(frozen=True)
class CalibrationResult:
    """Output of :func:`compute_calibration`."""

    n: int
    n_bins: int
    ece: float
    mean_confidence: float
    mean_fidelity: float
    bins: list[CalibrationBin] = field(default_factory=list)


def compute_calibration(
    records: Iterable[dict[str, Any] | HistoryRecord],
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error + per-bin reliability curve.

    A record contributes only when it carries both ``confidence`` and a
    fidelity block — records without fidelity (the default ``/explain``
    call path) are skipped silently.

    ECE = Σ_bins (n_bin / N) × |mean_confidence_bin − mean_fidelity_bin|

    Bins are equal-width over ``[0, 1]``; the last bin includes the
    closed upper edge (``conf == 1.0`` goes into bin ``n_bins - 1``).

    Args:
        records: Iterable of either raw recorder dicts or
            :class:`HistoryRecord` summaries.
        n_bins: Number of equal-width bins on ``[0, 1]``. Must be
            ``2..50``.

    Returns:
        :class:`CalibrationResult`.  ``ece == 0.0`` and empty bins when
        no usable records were found — caller is responsible for
        deciding what that signal means.
    """
    if not 2 <= n_bins <= 50:
        raise ValueError(f"n_bins must be in 2..50, got {n_bins}")

    pairs: list[tuple[float, float]] = []
    for raw in records:
        summary = raw if isinstance(raw, HistoryRecord) else HistoryRecord.from_record(raw)
        if summary.fidelity_score is None:
            continue
        conf = max(0.0, min(1.0, float(summary.confidence)))
        fid = max(0.0, min(1.0, float(summary.fidelity_score)))
        pairs.append((conf, fid))

    n = len(pairs)
    if n == 0:
        return CalibrationResult(
            n=0,
            n_bins=n_bins,
            ece=0.0,
            mean_confidence=0.0,
            mean_fidelity=0.0,
            bins=[
                CalibrationBin(
                    lo=i / n_bins, hi=(i + 1) / n_bins,
                    count=0, mean_confidence=0.0, mean_fidelity=0.0, gap=0.0,
                )
                for i in range(n_bins)
            ],
        )

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, fid in pairs:
        idx = min(n_bins - 1, int(conf * n_bins))
        buckets[idx].append((conf, fid))

    bins: list[CalibrationBin] = []
    ece = 0.0
    for i, bucket in enumerate(buckets):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not bucket:
            bins.append(CalibrationBin(lo=lo, hi=hi, count=0,
                                       mean_confidence=0.0,
                                       mean_fidelity=0.0, gap=0.0))
            continue
        mean_c = sum(c for c, _ in bucket) / len(bucket)
        mean_f = sum(f for _, f in bucket) / len(bucket)
        gap = abs(mean_c - mean_f)
        ece += (len(bucket) / n) * gap
        bins.append(CalibrationBin(
            lo=lo, hi=hi, count=len(bucket),
            mean_confidence=mean_c, mean_fidelity=mean_f, gap=gap,
        ))

    return CalibrationResult(
        n=n,
        n_bins=n_bins,
        ece=ece,
        mean_confidence=sum(c for c, _ in pairs) / n,
        mean_fidelity=sum(f for _, f in pairs) / n,
        bins=bins,
    )


__all__ = [
    "HistoryRecord",
    "HistoryPage",
    "CalibrationBin",
    "CalibrationResult",
    "load_records",
    "query_records",
    "compute_calibration",
]
