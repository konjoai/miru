"""Miru FastAPI application entry point."""
from fastapi import FastAPI

from miru.api.routes import router
from miru.config import settings

app = FastAPI(
    title="Miru",
    description="Multimodal reasoning tracer and VLM explainability engine",
    version=settings.version,
)

app.include_router(router)
