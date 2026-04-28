"""Tests for the backend registry (miru/models/registry.py).

All tests in this file run without MIRU_TEST_REAL_BACKENDS=1 — no real model
weights are loaded.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from miru.models import registry as reg
from miru.models.base import VLMBackend
from miru.models.mock import MockVLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry() -> None:
    """Wipe the global registry dict so tests are independent."""
    reg._REGISTRY.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    """Registering a lambda factory and retrieving it should work."""
    _fresh_registry()
    reg.register("dummy", MockVLMBackend)
    instance = reg.get("dummy")
    assert isinstance(instance, MockVLMBackend)
    _fresh_registry()


def test_get_unknown_raises_keyerror() -> None:
    """Requesting an unregistered backend must raise KeyError."""
    _fresh_registry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("does_not_exist")
    _fresh_registry()


def test_available_returns_sorted_list() -> None:
    """available() must return a sorted list of registered names."""
    _fresh_registry()
    reg.register("zebra", MockVLMBackend)
    reg.register("alpha", MockVLMBackend)
    reg.register("mock", MockVLMBackend)
    result = reg.available()
    assert result == sorted(result), "available() must be sorted"
    assert set(result) == {"alpha", "mock", "zebra"}
    _fresh_registry()


def test_register_defaults_includes_mock() -> None:
    """register_defaults() must register at least the 'mock' backend."""
    _fresh_registry()
    reg.register_defaults()
    assert "mock" in reg.available()
    _fresh_registry()


def test_register_defaults_idempotent() -> None:
    """Calling register_defaults() twice must not raise an error."""
    _fresh_registry()
    reg.register_defaults()
    reg.register_defaults()  # second call must not crash
    assert "mock" in reg.available()
    _fresh_registry()


def test_get_mock_returns_vlmbackend_instance() -> None:
    """get('mock') must return a VLMBackend subclass instance."""
    _fresh_registry()
    reg.register_defaults()
    instance = reg.get("mock")
    assert isinstance(instance, VLMBackend)
    _fresh_registry()


def test_get_mock_name_is_mock() -> None:
    """The 'mock' backend's .name property must equal 'mock'."""
    _fresh_registry()
    reg.register_defaults()
    instance = reg.get("mock")
    assert instance.name == "mock"
    _fresh_registry()


def test_health_endpoint_includes_registered_backends() -> None:
    """GET /health must list all registered backends in its response."""
    # Ensure registry is populated (prior isolation tests may have cleared it).
    _fresh_registry()
    reg.register_defaults()

    from miru.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    backends = resp.json()["backends"]
    assert isinstance(backends, list)
    assert "mock" in backends
    _fresh_registry()
