"""Prometheus metrics collector for Miru API traffic.

Requires the optional ``prometheus-client`` library; gracefully no-ops if absent.
All metrics are labeled per-backend. When prometheus_client is absent, all
methods are no-ops and :meth:`expose` returns an empty string.
"""
from __future__ import annotations

import threading

__all__ = ["MiruMetrics"]

_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _HAVE_PROMETHEUS = False


class MiruMetrics:
    """Prometheus metrics collector for miru API traffic.

    All metrics are per-backend labeled. When prometheus_client is absent,
    all methods are no-ops and expose() returns an empty string.
    """

    def __init__(self, registry: object = None) -> None:
        """Create metrics. registry=None uses a fresh CollectorRegistry.

        If prometheus_client is not installed, silently degrades to no-op.

        Args:
            registry: An optional ``CollectorRegistry`` to register metrics on.
                      Defaults to a new isolated registry (not the global default).
        """
        self._lock = threading.Lock()
        self._seen_backends: set[str] = set()

        if not _HAVE_PROMETHEUS:
            self._enabled = False
            return

        self._enabled = True
        self._registry = registry if registry is not None else CollectorRegistry()

        self._requests_total = Counter(
            "miru_requests_total",
            "Total /analyze requests by backend and status",
            labelnames=["backend", "status"],
            registry=self._registry,
        )

        self._latency = Histogram(
            "miru_latency_seconds",
            "Latency of /analyze requests in seconds",
            labelnames=["backend"],
            buckets=_LATENCY_BUCKETS,
            registry=self._registry,
        )

        self._active_backends = Gauge(
            "miru_active_backends",
            "Number of distinct backends that have received at least one request",
            registry=self._registry,
        )

    @property
    def enabled(self) -> bool:
        """True if prometheus_client is installed and metrics are active."""
        return self._enabled

    def record_request(self, backend: str, latency_ms: float, success: bool) -> None:
        """Record one /analyze request. Thread-safe.

        Args:
            backend: Backend name label (e.g. ``"mock"``).
            latency_ms: Wall-clock latency of the inference call in milliseconds.
            success: ``True`` for a successful inference, ``False`` for an error.
        """
        if not self._enabled:
            return

        status = "ok" if success else "error"
        latency_s = latency_ms / 1_000.0

        self._requests_total.labels(backend=backend, status=status).inc()
        self._latency.labels(backend=backend).observe(latency_s)

        with self._lock:
            if backend not in self._seen_backends:
                self._seen_backends.add(backend)
                self._active_backends.set(len(self._seen_backends))

    def expose(self) -> str:
        """Return Prometheus text format string. Empty string if prometheus absent.

        Returns:
            A UTF-8 string in Prometheus text exposition format 0.0.4, or an
            empty string when ``prometheus_client`` is not installed.
        """
        if not self._enabled:
            return ""
        raw = generate_latest(self._registry)
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw
