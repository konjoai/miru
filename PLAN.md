# PLAN.md — Miru Roadmap

**Project:** Miru — Multimodal Reasoning Tracer  
**Current version:** v0.1.0  
**Status:** Core engine complete, 30/30 tests passing

---

## Phase 1 — Core Engine (v0.1.0) ✅ COMPLETE

**Goal:** Establish the full service skeleton with a deterministic mock backend, attention extraction, reasoning tracer, and FastAPI REST API.

**Delivered:**
- `miru/` Python package with layered architecture
- Abstract `VLMBackend` interface (`miru/models/base.py`)
- Deterministic `MockVLMBackend` with stable-hash Gaussian attention maps (`miru/models/mock.py`)
- `AttentionExtractor`: min-max normalization, block-average resize, top-k hotspot detection (`miru/attention/extractor.py`)
- `ReasoningTracer`: structured reasoning trace with decay confidence (`miru/reasoning/tracer.py`)
- FastAPI app with `GET /health` and `POST /analyze` endpoints (`miru/api/routes.py`)
- Pydantic v2 schemas: `ImageInput`, `AttentionMap`, `ReasoningStep`, `ReasoningTrace`, `HealthResponse`, `ErrorResponse`
- 30 passing tests across `test_models`, `test_attention`, `test_reasoning`, `test_api`
- GitHub Actions CI (Python 3.11, ubuntu-latest)
- `pyproject.toml` with `[dev]` extras

**Ship gate:** 30/30 tests pass, no placeholders, no TODOs.

---

## Phase 2 — Real VLM Backends (v0.2.0)

**Goal:** Wire in CLIP and LLaVA-1.5 as optional HuggingFace-backed backends.

**Planned work:**
- `miru/models/clip.py` — CLIP ViT image encoder; return cross-attention as attention map
- `miru/models/llava.py` — LLaVA-1.5 7B via `transformers`; extract cross-attention from decoder layers
- Backend registry auto-discovery via entry points
- `backend: "clip"` and `backend: "llava-1.5"` accepted by `/analyze`
- Optional dependency group `[backends]` in `pyproject.toml`
- Integration tests gated on `MIRU_TEST_REAL_BACKENDS=1`

**Ship gate:** mock tests still pass; each real backend produces cosine similarity ≥ 0.90 versus BF16 reference on a 5-image fixture set.

---

## Phase 3 — Visualization (v0.3.0)

**Goal:** Return attention overlays as PNG/base64 alongside the JSON trace.

**Planned work:**
- `miru/visualization/overlay.py` — bilinear upsample attention map → RGBA heatmap overlay on input image (Pillow)
- `ReasoningTrace.overlay_b64: str | None` — optional PNG overlay
- `/analyze?overlay=true` query parameter
- GradCAM support for transformer attention layers
- Tests: pixel-level regression against known fixture overlays

---

## Phase 4 — Streaming (v0.4.0)

**Goal:** SSE streaming for token-by-token reasoning trace delivery.

**Planned work:**
- `GET /analyze/stream` SSE endpoint — streams `ReasoningStep` events as the model generates tokens
- `EventSourceResponse` via `sse-starlette`
- Backpressure and timeout handling
- Tests: consume full SSE stream and validate final state matches `/analyze`

---

## Phase 5 — Dataset Recorder (v0.5.0)

**Goal:** Record reasoning traces to disk/S3 for fine-tuning data pipelines.

**Planned work:**
- `miru/recorder.py` — async queue writer; flushes JSONL batches to `fsspec` backend
- `/analyze` middleware hook to enqueue traces when `MIRU_RECORD=1`
- `miru record list` / `miru record export` CLI
- Privacy: strip raw image bytes from stored traces; store hash only
- Tests: mock fsspec; assert JSONL schema matches `ReasoningTrace`
