"""Per-model aggregation over the recorded explanation store.

Drives ``GET /explain/models/compare``. For each model name supplied
by the caller, this module pulls the most-recent ``limit`` records
from history, computes a small fixed set of summary statistics, and
returns a side-by-side comparison plus a "winner" verdict across
three metrics: mean confidence (higher wins), mean fidelity (higher
wins), ECE (lower wins).

The module is read-only: no recorder writes, no backend calls. All
work is in-memory aggregation over what :mod:`miru.history` returns.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from miru.history import HistoryRecord, compute_calibration, query_records


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStats:
    """Aggregate stats for one model.

    All ``mean_*`` fields are ``None`` when the underlying sample size
    is zero — clients render an "insufficient data" state rather than
    seeing a misleading zero.
    """

    model: str
    n_records: int
    mean_confidence: Optional[float]
    mean_latency_ms: Optional[float]
    mean_fidelity: Optional[float]      # None when no records carry fidelity
    n_with_fidelity: int
    ece: Optional[float]                # None when n_with_fidelity == 0
    method_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelComparisonResult:
    """Output of :func:`compare_models`."""

    models: list[str]                   # requested order, preserved
    stats: dict[str, ModelStats]
    winner_by_confidence: Optional[str]
    winner_by_fidelity: Optional[str]
    winner_by_ece: Optional[str]
    filter_method: Optional[str]
    limit: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_one(model: str, records: list[HistoryRecord]) -> ModelStats:
    confs = [r.confidence for r in records]
    latencies = [r.latency_ms for r in records]
    fidelities = [r.fidelity_score for r in records if r.fidelity_score is not None]
    methods = Counter(r.method for r in records if r.method)

    # ECE requires records with fidelity. Empty population → None
    # (clients render an "insufficient data" state).
    if fidelities:
        # compute_calibration ignores records without fidelity, so it's
        # safe to pass the whole record set.
        calibration = compute_calibration(records, n_bins=10)
        ece: Optional[float] = calibration.ece if calibration.n > 0 else None
    else:
        ece = None

    return ModelStats(
        model=model,
        n_records=len(records),
        mean_confidence=_safe_mean(confs),
        mean_latency_ms=_safe_mean(latencies),
        mean_fidelity=_safe_mean(fidelities),
        n_with_fidelity=len(fidelities),
        ece=ece,
        method_distribution=dict(methods),
    )


def compare_models(
    models: list[str],
    *,
    limit: int = 50,
    method: Optional[str] = None,
    directory: Optional[str] = None,
) -> ModelComparisonResult:
    """Aggregate per-model stats over recorded history.

    Args:
        models: Distinct model names to compare. Length 1..8 enforced
            at the API boundary; this function accepts any list ≥ 1.
        limit: Per-model record cap; the most-recent ``limit`` matching
            records contribute to the stats. Bounded ``1..200``.
        method: Optional explanation-method filter applied to every
            model (e.g. only compare ``gradcam`` runs across models).
        directory: Override the recorder directory (test hook).

    Returns:
        :class:`ModelComparisonResult` with per-model :class:`ModelStats`
        and three winner verdicts.  A winner is ``None`` when no model
        has data for that metric.

    Raises:
        ValueError: When ``models`` is empty or has duplicates, or
            when ``limit`` is out of bounds.
    """
    if not models:
        raise ValueError("models must be a non-empty list")
    if len(set(models)) != len(models):
        raise ValueError(f"models must be distinct; got {models}")
    if not 1 <= limit <= 200:
        raise ValueError(f"limit must be in 1..200, got {limit}")

    stats: dict[str, ModelStats] = {}
    for model in models:
        page = query_records(
            directory=directory,
            method=method,
            model=model,
            limit=limit,
            offset=0,
        )
        stats[model] = _aggregate_one(model, page.items)

    return ModelComparisonResult(
        models=list(models),
        stats=stats,
        winner_by_confidence=_argmax(stats, "mean_confidence"),
        winner_by_fidelity=_argmax(stats, "mean_fidelity"),
        winner_by_ece=_argmin(stats, "ece"),
        filter_method=method,
        limit=limit,
    )


def _argmax(stats: dict[str, ModelStats], field_name: str) -> Optional[str]:
    """Return the model with the highest value on ``field_name``.

    Ignores models where the field is ``None``. Ties broken by the
    order of the input ``models`` list (i.e. first model wins ties).
    """
    best_model: Optional[str] = None
    best_value: Optional[float] = None
    for model, st in stats.items():
        value = getattr(st, field_name)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_model = model
    return best_model


def _argmin(stats: dict[str, ModelStats], field_name: str) -> Optional[str]:
    """Return the model with the lowest value on ``field_name``.

    Ignores models where the field is ``None``. Ties broken by the
    order of the input ``models`` list (i.e. first model wins ties).
    """
    best_model: Optional[str] = None
    best_value: Optional[float] = None
    for model, st in stats.items():
        value = getattr(st, field_name)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_model = model
    return best_model


__all__ = [
    "ModelStats",
    "ModelComparisonResult",
    "compare_models",
]
