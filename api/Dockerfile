# Miru explainability API — slim production image.
#
# Build:  docker build -f api/Dockerfile -t miru-api .
# Run:    docker run -p 8000:8000 miru-api
#
# The build context is the repo root so the image can copy both the
# top-level miru/ package and the api/ deployment surface.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Install runtime deps first so the layer caches across code-only edits.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy source.  miru/ is the library; api/ is the deployable surface.
COPY miru /app/miru
COPY api /app/api

# Non-root user for runtime — defence in depth on the deploy host.
RUN useradd --create-home --uid 10001 miru && chown -R miru:miru /app
USER miru

EXPOSE 8000

# Render injects $PORT; default to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
