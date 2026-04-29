# PLAN.md — Miru Roadmap

**Project:** Miru — Multimodal Reasoning Tracer  
**Current version:** v0.3.0  
**Status:** Visualization overlay complete, 65/65 tests passing (4 real-backend tests skipped without MIRU_TEST_REAL_BACKENDS=1)

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

## Phase 2 — Real VLM Backends (v0.2.0) ✅ COMPLETE

**Goal:** Wire in CLIP as an optional HuggingFace-backed backend and introduce a proper backend registry.

**Delivered:**
- `miru/models/registry.py` — `register()`, `get()`, `available()`, `register_defaults()` factory registry
- `miru/models/clip.py` — `CLIPBackend` using `transformers.CLIPModel` + `CLIPProcessor`; lazy-loads weights on first `infer()` call; derives 2-D attention map from last ViT encoder layer's [CLS] token attention averaged across heads
- `miru/api/routes.py` — rewired to use registry; `register_defaults()` called at module import; `/health` returns `registry.available()`; `/analyze` falls back to default backend on `KeyError`
- `pyproject.toml` — added `[backends]` optional dependency group (`transformers>=4.35.0`, `torch>=2.0.0`, `Pillow>=9.0.0`)
- `tests/test_registry.py` — 8 tests covering register/get/available/defaults/idempotency/health endpoint
- `tests/test_clip_backend.py` — 8 tests: 4 structural (no model load) + 4 gated behind `MIRU_TEST_REAL_BACKENDS=1`

**Ship gate:** 53/53 tests pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`; all 41 Phase 1 tests still pass.

---

## Phase 3 — Visualization (v0.3.0) ✅ COMPLETE

**Goal:** Return attention overlays as PNG/base64 alongside the JSON trace.

**Delivered:**
- `miru/visualization/overlay.py` — bilinear upsample attention map → RGBA heatmap overlay on input image; jet/hot/viridis colormaps implemented as piecewise-linear functions (no matplotlib); Pillow used when available, pure-zlib PNG encoder as fallback
- `miru/visualization/__init__.py` — exports `attention_to_heatmap`, `overlay_attention_on_image`, `encode_png_b64`, `decode_image_b64`, `generate_overlay`
- `ReasoningTrace.overlay_b64: str | None = None` — optional base64 PNG overlay field
- `ReasoningTracer.trace()` — extended with `image_b64` and `generate_overlay` optional params; silently falls back to `overlay_b64=None` on any overlay error
- `POST /analyze?overlay=true` — query parameter wires image payload → overlay pipeline → `overlay_b64` field in response
- `miru/__init__.py` — exports `attention_to_heatmap`, `generate_overlay`; bumped to v0.3.0
- `tests/test_overlay.py` — 8 tests: zero-attention blue, full-attention red, dtype/range, overlay shape, encode valid base64, round-trip decode, generate_overlay pipeline, schema field default
- `tests/test_api_overlay.py` — 4 tests: no-overlay null, overlay non-empty string, invalid-image no crash, health regression guard

**Ship gate:** 65/65 tests pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`; all 53 Phase 1+2 tests still pass.

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
