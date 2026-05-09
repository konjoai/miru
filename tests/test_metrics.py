"""Tests for Prometheus metrics integration."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from miru.main import app
from miru.metrics import MiruMetrics


class TestMiruMetrics:
    """Unit tests for MiruMetrics class."""

    def test_metrics_init_requires_prometheus(self):
        """MiruMetrics.__init__ raises ImportError if prometheus-client is absent."""
        with patch.dict(sys.modules, {"prometheus_client": None}):
            with pytest.raises(ImportError, match="prometheus-client"):
                with patch("miru.metrics._HAVE_PROMETHEUS", False):
                    MiruMetrics()

    def test_metrics_registry_created(self):
        """MiruMetrics creates a CollectorRegistry on init."""
        metrics = MiruMetrics()
        assert metrics.registry is not None
        assert hasattr(metrics.registry, "register")

    def test_analyze_histogram_exists(self):
        """MiruMetrics has an analyze_latency histogram."""
        metrics = MiruMetrics()
        assert hasattr(metrics, "analyze_latency")
        assert metrics.analyze_latency is not None

    def test_stream_histogram_exists(self):
        """MiruMetrics has a stream_latency histogram."""
        metrics = MiruMetrics()
        assert hasattr(metrics, "stream_latency")
        assert metrics.stream_latency is not None

    def test_extraction_histogram_exists(self):
        """MiruMetrics has an extraction_latency histogram."""
        metrics = MiruMetrics()
        assert hasattr(metrics, "extraction_latency")
        assert metrics.extraction_latency is not None

    def test_observe_analyze_positive_latency(self):
        """observe_analyze records a positive latency."""
        metrics = MiruMetrics()
        metrics.observe_analyze("mock", 0.123)
        # Verify the histogram incremented (cannot directly read prometheus histogram count)
        output = metrics.render()
        assert b"miru_analyze_latency_seconds" in output
        assert b'backend="mock"' in output

    def test_observe_stream_positive_latency(self):
        """observe_stream records a positive latency."""
        metrics = MiruMetrics()
        metrics.observe_stream("mock", 0.456)
        output = metrics.render()
        assert b"miru_stream_analyze_latency_seconds" in output
        assert b'backend="mock"' in output

    def test_observe_extraction_latency(self):
        """observe_extraction records a latency."""
        metrics = MiruMetrics()
        metrics.observe_extraction(0.789)
        output = metrics.render()
        assert b"miru_attention_extraction_latency_seconds" in output

    def test_observe_ignores_nan(self):
        """observe_* methods silently ignore NaN and Inf latencies."""
        metrics = MiruMetrics()
        # Should not raise
        metrics.observe_analyze("mock", float("nan"))
        metrics.observe_stream("mock", float("inf"))
        metrics.observe_extraction(float("-inf"))

    def test_render_returns_bytes(self):
        """render() returns bytes in Prometheus exposition format."""
        metrics = MiruMetrics()
        metrics.observe_analyze("mock", 0.100)
        output = metrics.render()
        assert isinstance(output, bytes)
        assert b"miru_analyze_latency_seconds" in output

    def test_multiple_backends_tracked(self):
        """Multiple backends are tracked as separate label values."""
        metrics = MiruMetrics()
        metrics.observe_analyze("mock", 0.100)
        metrics.observe_analyze("clip", 0.150)
        output = metrics.render()
        assert b'backend="mock"' in output
        assert b'backend="clip"' in output


class TestMetricsEndpoint:
    """Integration tests for GET /metrics endpoint."""

    def test_metrics_endpoint_returns_200(self):
        """GET /metrics returns 200 when prometheus-client is available."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self):
        """GET /metrics returns proper Prometheus exposition format."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.headers["content-type"].startswith("text/plain")
        assert b"# HELP" in response.content or b"miru_" in response.content

    def test_metrics_endpoint_not_found_without_prometheus(self, monkeypatch):
        """GET /metrics returns 404 when prometheus-client not available."""
        monkeypatch.setattr("miru.api.routes._metrics", None)
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 404

    def test_analyze_increments_metrics(self):
        """POST /analyze increments the analyze histogram."""
        client = TestClient(app)
        payload = {
            "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "question": "What is this?",
            "backend": "mock",
        }
        client.post("/analyze", json=payload)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"miru_analyze_latency_seconds" in response.content
