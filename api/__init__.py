"""Miru deployable REST API surface (M10).

Thin FastAPI layer that exposes miru's saliency-map generation,
synthetic-mask benchmark harness, and backend comparison over HTTP for
production deployment (Render, Fly, Cloud Run, etc.).

Distinct from ``miru/api/`` — that one is the in-package router used by
the development server.  This package is the deployable artefact: it
imports the in-package primitives, adds REST/JSON ergonomics, and is
what ``api/Dockerfile`` ships.
"""

from api.main import app

__all__ = ["app"]
