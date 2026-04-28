"""Application configuration via plain Pydantic BaseModel (no extra deps)."""
from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    """Immutable application settings with sane defaults."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "miru"
    version: str = "0.1.0"
    default_backend: str = "mock"
    max_image_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    attention_resolution: int = 16  # NxN grid size


settings = Settings()
