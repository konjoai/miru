"""Prometheus metrics for Miru request latency and performance.

Exports per-backend histograms for /analyze, /analyze/stream, and attention extraction.
Requires the optional prometheus-client library; gracefully no-ops if absent.
"""
from __future__ import annotations

__all__ = ["MiruMetrics"]

try:
    from prometheus_client import CollectorRegistry, Histogram, generate_latest

    _HAVE_PROMETHEUS = True
except ImportError:
    _HAVE_PROMETHEUS = False


class MiruMetrics:
    """Prometheus metrics collector for Miru latencies.

    Tracks per-backend request latencies for analyze, stream_analyze, and attention extraction.
    Histograms use Prometheus default buckets (exponential distribution, 0.005s to 10s).

    Raises:
        ImportError: If prometheus-client is not installed.
    """

    def __init__(self) -> None:
        if not _HAVE_PROMETHEUS:
            raise ImportError(
                "prometheus-client is required for Miru metrics; "
                "install with: pip install prometheus-client"
            )

        self.registry = CollectorRegistry()

        # Per-backend histograms for analyze and stream_analyze latencies (seconds)
        self.analyze_latency = Histogram(
            "miru_analyze_latency_seconds",
            "Latency of /analyze requests",
            labelnames=["backend"],
            registry=self.registry,
        )

        self.stream_latency = Histogram(
            "miru_stream_analyze_latency_seconds",
            "Latency of /analyze/stream requests",
            labelnames=["backend"],
            registry=self.registry,
        )

        # Attention extraction latency (no backend label, global)
        self.extraction_latency = Histogram(
            "miru_attention_extraction_latency_seconds",
            "Latency of attention map extraction",
            registry=self.registry,
        )

    def observe_analyze(self, backend: str, latency_s: float) -> None:
        """Record an /analyze request latency for a given backend.

        Args:
            backend: Backend name (e.g., 'mock', 'clip').
            latency_s: Request latency in seconds (float).
        """
        if latency_s >= 0 and not (latency_s != latency_s):  # Check for NaN/Inf
            self.analyze_latency.labels(backend=backend).observe(latency_s)

    def observe_stream(self, backend: str, latency_s: float) -> None:
        """Record an /analyze/stream request latency for a given backend.

        Args:
            backend: Backend name (e.g., 'mock', 'clip').
            latency_s: Request latency in seconds (float).
        """
        if latency_s >= 0 and not (latency_s != latency_s):
            self.stream_latency.labels(backend=backend).observe(latency_s)

    def observe_extraction(self, latency_s: float) -> None:
        """Record an attention extraction latency.

        Args:
            latency_s: Extraction latency in seconds (float).
        """
        if latency_s >= 0 and not (latency_s != latency_s):
            self.extraction_latency.observe(latency_s)

    def render(self) -> bytes:
        """Render metrics in Prometheus text exposition format.

        Returns:
            Bytes suitable for HTTP response body with content-type
            "text/plain; version=0.0.4; charset=utf-8".
        """
        return generate_latest(self.registry)
