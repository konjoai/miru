"""Miru metrics package — Prometheus exposition for API traffic.

Exports :class:`MiruMetrics` and the module-level singleton accessor
:func:`get_metrics`.
"""
from __future__ import annotations

from miru.metrics.collector import MiruMetrics

__all__ = ["MiruMetrics", "get_metrics"]

_singleton: MiruMetrics | None = None


def get_metrics() -> MiruMetrics:
    """Return the module-level MiruMetrics singleton.

    Initialised lazily on first call. Subsequent calls return the same instance.

    Returns:
        The shared :class:`MiruMetrics` instance.
    """
    global _singleton  # noqa: PLW0603
    if _singleton is None:
        _singleton = MiruMetrics()
    return _singleton
