"""Backend registry: maps names to VLMBackend factory functions."""
from __future__ import annotations

from typing import Callable

from miru.models.base import VLMBackend

_REGISTRY: dict[str, Callable[[], VLMBackend]] = {}


def register(name: str, factory: Callable[[], VLMBackend]) -> None:
    """Register a backend factory under `name`."""
    _REGISTRY[name] = factory


def get(name: str) -> VLMBackend:
    """Instantiate and return a backend by name. Raises KeyError if not registered."""
    if name not in _REGISTRY:
        raise KeyError(f"Backend '{name}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_REGISTRY)


def register_defaults() -> None:
    """Register built-in backends. Called once at app startup."""
    from miru.models.mock import MockVLMBackend

    register("mock", MockVLMBackend)
    try:
        from miru.models.clip import CLIPBackend

        register("clip", CLIPBackend)
    except ImportError:
        pass  # transformers not installed
