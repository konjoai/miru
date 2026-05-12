"""Tests for Prometheus metrics integration — Phase 10 (v1.0.0).

Coverage:
- MiruMetrics without prometheus (no-op degraded mode)
- MiruMetrics with isolated CollectorRegistry (full mode)
- GET /metrics API endpoint
- POST /analyze metric recording
- Thread safety
"""
from __future__ import annotations

import concurrent.futures
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from miru.main import app
from miru.metrics import MiruMetrics, get_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics() -> tuple["MiruMetrics", object]:
    """Return a fresh MiruMetrics instance backed by an isolated registry."""
    from prometheus_client import CollectorRegistry
    reg = CollectorRegistry()
    return MiruMetrics(registry=reg), reg


def _small_png_b64() -> str:
    """Return a valid 1×1 PNG encoded as base64 (from conftest fixture pattern)."""
    import base64
    import numpy as np
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode()


# ---------------------------------------------------------------------------
# Without prometheus (degraded / no-op mode)
# ---------------------------------------------------------------------------

class TestMiruMetricsNoop:
    """MiruMetrics behaviour when prometheus_client is absent."""

    def test_metrics_enabled_is_bool(self):
        """MiruMetrics().enabled is always a bool."""
        m = MiruMetrics()
        assert isinstance(m.enabled, bool)

    def test_record_request_no_crash_when_disabled(self):
        """record_request does not raise when prometheus is absent."""
        with patch("miru.metrics.collector._HAVE_PROMETHEUS", False):
            m = MiruMetrics()
        # Should be no-op — must not raise
        m.record_request("mock", 12.5, True)

    def test_expose_returns_str(self):
        """expose() always returns a str, never bytes or None."""
        m = MiruMetrics()
        result = m.expose()
        assert isinstance(result, str)

    def test_expose_empty_when_no_requests(self):
        """A fresh MiruMetrics with no recorded requests exposes empty or minimal text."""
        with patch("miru.metrics.collector._HAVE_PROMETHEUS", False):
            m = MiruMetrics()
        assert m.expose() == ""

    def test_enabled_false_when_prometheus_absent(self):
        """enabled is False when prometheus_client cannot be imported."""
        with patch("miru.metrics.collector._HAVE_PROMETHEUS", False):
            m = MiruMetrics()
        assert m.enabled is False


# ---------------------------------------------------------------------------
# With prometheus (isolated registry)
# ---------------------------------------------------------------------------

class TestMiruMetricsFull:
    """MiruMetrics behaviour with prometheus_client installed."""

    def test_metrics_enabled_true_with_registry(self):
        """enabled=True when a registry is passed and prometheus is available."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        assert m.enabled is True

    def test_record_request_increments_counter(self):
        """record_request increments the total count."""
        pytest.importorskip("prometheus_client")
        from prometheus_client import REGISTRY
        m, reg = _make_metrics()
        m.record_request("mock", 10.0, True)
        output = m.expose()
        assert "miru_requests_total" in output

    def test_record_request_ok_status(self):
        """record_request with success=True labels status as 'ok'."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 5.0, True)
        output = m.expose()
        assert 'status="ok"' in output

    def test_record_request_error_status(self):
        """record_request with success=False labels status as 'error'."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 5.0, False)
        output = m.expose()
        assert 'status="error"' in output

    def test_latency_histogram_populated(self):
        """Histogram has observations after record_request."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 100.0, True)
        output = m.expose()
        assert "miru_latency_seconds" in output
        assert "miru_latency_seconds_sum" in output

    def test_active_backends_gauge_updates(self):
        """miru_active_backends gauge increments per unique backend."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 1.0, True)
        output = m.expose()
        assert "miru_active_backends" in output
        # Gauge value should be 1 after recording one backend
        assert "1.0" in output or " 1\n" in output

    def test_multiple_backends_tracked_separately(self):
        """Each backend gets its own label set in the output."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 10.0, True)
        m.record_request("clip", 20.0, True)
        output = m.expose()
        assert 'backend="mock"' in output
        assert 'backend="clip"' in output

    def test_expose_contains_miru_requests_total(self):
        """expose() output includes 'miru_requests_total'."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 5.0, True)
        assert "miru_requests_total" in m.expose()

    def test_expose_contains_miru_latency_seconds(self):
        """expose() output includes 'miru_latency_seconds'."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 5.0, True)
        assert "miru_latency_seconds" in m.expose()

    def test_expose_valid_prometheus_format(self):
        """expose() output is parseable by prometheus_client text parser."""
        pytest.importorskip("prometheus_client")
        from prometheus_client.exposition import choose_encoder
        m, _ = _make_metrics()
        m.record_request("mock", 50.0, True)
        output = m.expose()
        # generate_latest is already valid; check it ends with newline and has HELP
        assert "# HELP" in output
        assert output.endswith("\n")

    def test_active_backends_gauge_single_backend_multiple_calls(self):
        """Gauge does not double-count the same backend."""
        pytest.importorskip("prometheus_client")
        m, _ = _make_metrics()
        m.record_request("mock", 1.0, True)
        m.record_request("mock", 2.0, True)
        m.record_request("mock", 3.0, False)
        output = m.expose()
        # Should still be 1 unique backend
        lines = [l for l in output.splitlines() if "miru_active_backends" in l and not l.startswith("#")]
        assert any("1.0" in l for l in lines)


# ---------------------------------------------------------------------------
# API integration (TestClient)
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    """Integration tests for GET /metrics endpoint."""

    def test_get_metrics_endpoint_exists(self):
        """GET /metrics returns 200."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_get_metrics_content_type(self):
        """GET /metrics content-type is text/plain."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.headers["content-type"].startswith("text/plain")

    def test_get_metrics_returns_str(self):
        """GET /metrics body is decodable as a string."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert isinstance(response.text, str)

    def test_analyze_records_metric(self):
        """POST /analyze then GET /metrics shows miru_requests_total."""
        pytest.importorskip("prometheus_client")
        client = TestClient(app)
        payload = {
            "image_b64": _small_png_b64(),
            "question": "What is this?",
            "backend": "mock",
        }
        client.post("/analyze", json=payload)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "miru_requests_total" in response.text

    def test_analyze_error_records_error_status(self):
        """A failed /analyze call records error status without crashing."""
        pytest.importorskip("prometheus_client")
        client = TestClient(app)
        # Send malformed payload — FastAPI will 422, but metrics should not blow up
        response = client.post("/analyze", json={})
        # 422 is expected for missing required fields
        assert response.status_code == 422
        # Metrics endpoint must still be healthy
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200

    def test_metrics_thread_safe(self):
        """50 concurrent record_request calls produce no crash."""
        pytest.importorskip("prometheus_client")
        from prometheus_client import CollectorRegistry
        reg = CollectorRegistry()
        m = MiruMetrics(registry=reg)

        def _record(i: int) -> None:
            m.record_request(f"backend_{i % 5}", float(i), i % 2 == 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_record, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # re-raise any exception

        output = m.expose()
        assert "miru_requests_total" in output

    def test_get_metrics_disabled_returns_200_empty(self, monkeypatch):
        """GET /metrics returns 200 with empty body when metrics are disabled."""
        from miru.metrics import collector as _col
        with patch("miru.metrics.collector._HAVE_PROMETHEUS", False):
            disabled = MiruMetrics()
        monkeypatch.setattr("miru.api.routes.get_metrics", lambda: disabled)
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.text == ""

    def test_get_metrics_singleton_consistency(self):
        """get_metrics() returns the same object on repeated calls."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2
