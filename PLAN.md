# PLAN.md — Miru Roadmap

**Project:** Miru — Multimodal Reasoning Tracer  
**Current version:** v1.1.0  
**Status:** Grad-CAM + visual explainability demo, 252 tests passing (4 real-backend tests skipped without MIRU_TEST_REAL_BACKENDS=1)

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

## Phase 4 — Streaming (v0.4.0) ✅ COMPLETE

**Goal:** SSE streaming for token-by-token reasoning trace delivery.

**Delivered:**
- `miru/api/streaming.py` — `stream_analyze()` async generator, byte-level SSE framing (`step` / `trace` / `done` / `error` events), `: keepalive` comments, per-request `timeout_seconds` budget, backpressure via `asyncio.Queue(maxsize=64)` between thread-pool producer and event-loop consumer
- `miru/models/base.py` — `VLMStreamChunk` dataclass + `VLMBackend.stream_infer()` default generator that replays `infer()` reasoning steps progressively; native streaming backends override
- `miru/reasoning/tracer.py` — extracted `step_confidence()` so streamed and synchronous traces produce identical per-step confidence values
- `miru/api/routes.py` — `POST /analyze/stream` with `overlay: bool` and `timeout_seconds: float` (1–300) query params; same `ImageInput` payload, same unknown-backend fallback
- `tests/test_streaming.py` — 10 tests covering frame ordering, payload shape, parity with `/analyze`, overlay propagation, timeout budget, and backend-failure error events

**Deviations:**
- Endpoint is `POST`, not `GET` (per original sketch) — image_b64 doesn't fit a GET query string. POST + `text/event-stream` is the canonical pattern.
- No `sse-starlette` dependency — SSE framing is hand-rolled (5 lines). 건조.

**Ship gate:** 75/75 tests pass; all 65 prior tests still pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 5 — Dataset Recorder (v0.5.0) ✅ COMPLETE

**Goal:** Record reasoning traces to disk/S3 for fine-tuning data pipelines.

**Delivered:**
- `miru/recorder.py` — `TraceRecorder` threaded JSONL writer; per-batch timestamped files for fsspec compatibility (S3, GCS, memory, local); `MIRU_RECORD` env gating; `MIRU_RECORD_PATH` directory override
- Privacy: SHA-256 hex of `image_b64` is the only image-derived data persisted; raw bytes and overlay PNG are stripped before serialisation
- API hooks: `maybe_record()` called from `POST /analyze` and inside `stream_analyze` for `POST /analyze/stream`; errors swallowed so recording can never break the request path
- CLI (`miru/cli/`): `miru record list` (tab-separated record-count/size/path per file) and `miru record export --out <f> [--format jsonl|csv]`; entry point registered under `[project.scripts] miru`
- Optional `[storage]` extras install of `fsspec`; `fsspec` also bundled in `[dev]`
- 25 new tests across `tests/test_recorder.py` (17) and `tests/test_record_cli.py` (8); `memory://` round-trip exercises the fsspec path

**Ship gate:** 100/100 tests pass; all 75 prior tests still pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 6 — Saliency Benchmark Harness (v0.6.0) ✅ COMPLETE

**Goal:** Quantify how well an attention map tracks the salient region of an image, with a deterministic harness that runs in CI.

**Delivered:**
- `miru/bench/synth.py` — deterministic synthetic image + ground-truth-mask generator (single / two-blob / low-SNR variants), reproducible from `(seed, index)`
- `miru/bench/metrics.py` — `iou_at_topk_pct`, `auc_roc` (Mann-Whitney U with tie correction), `hit_at_k`; pure NumPy bilinear resampler
- `miru/bench/runner.py` — `run_benchmark()` drives any registered backend over the synth dataset, aggregates `{mean, std, p50, p95}` per metric, captures hardware metadata, persists JSON; `compare_results()` reports paired delta + paired-t statistic
- `miru/cli/bench.py` — `miru bench run / show / compare` subcommands wired into the existing CLI entry point
- `benchmarks/results/baseline-mock.json` — first locked-in baseline (n=30, seed=42): IoU 0.062, AUC 0.627, hit@1 0.100, latency 0.080 ms — confirms the mock backend's attention is question-hash-driven, not image-driven
- 29 new tests across `tests/test_bench.py`

**Ship gate:** 129/129 tests pass; all 100 prior tests still pass; first real benchmark recorded against the mock backend.

---

## Phase 7 — Attention-Map Export + HTML Report (v0.7.0) ✅ COMPLETE

**Goal:** Turn any saved benchmark JSON into a visual artefact — per-sample
attention-map overlay PNGs and a self-contained HTML report — so developers
can visually inspect what the synth harness looks like and spot attention
anomalies without writing notebook code.

**Delivered:**
- `miru/bench/export.py` — `generate_report()` driver; `render_sample()` (re-generates
  synth image from `(seed, index)`, runs mock inference, composites the jet/hot/viridis
  heatmap, draws a yellow GT-mask border via Porter-Duff over compositing);
  `_composite_overlay`, `_mask_border_rgba`, `_alpha_composite`, `_image_to_rgba` helpers;
  zero new runtime dependencies (pure NumPy + existing visualization layer)
- `miru/cli/export.py` — `run_export_report()` with `alpha`, `colormap`,
  `--no-mask-border`, `--no-png-tiles` flags
- `miru/cli/__init__.py` — `miru export <result.json> <out_dir>` subcommand wired in
- `pyproject.toml` — added `Pillow>=9.0.0` to `[dev]` extras (was missing, caused
  pre-existing `test_analyze_with_overlay_returns_nonempty_string` failure)
- 32 new tests in `tests/test_export.py` across rendering helpers, `generate_report`,
  and CLI integration

**Ship gate:** 161/161 tests pass; all 129 prior tests still pass; 4 real-backend tests
skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 8 — CLIP vs Mock Backend Comparison (v0.8.0) ✅ COMPLETE

**Goal:** Produce a concrete `clip-vs-mock` comparison artefact — concrete proof
that attention quality varies meaningfully across backends, and a regression gate
for future backend changes.

**Delivered:**
- `miru/bench/comparison.py` — `BackendComparison` dataclass + `compare_backends()`
  function; runs `run_benchmark()` on two named backends with matching seed, calls
  `compare_results()` for the paired delta, determines winner from `mean_delta`,
  captures hardware metadata; `BackendComparison.save()` writes a timestamped JSON
  artefact to `benchmarks/results/` and never overwrites an existing file
- `miru/cli/__init__.py` — `miru compare <backend_a> <backend_b>` top-level subcommand
  with `--n`, `--seed`, `--name`, `--save`, `--out-dir` flags; prints a summary table
- `miru/__init__.py` — bumped to v0.8.0
- `pyproject.toml` — bumped to v0.8.0
- 15 new tests in `tests/test_comparison.py`

**Ship gate:** 176/176 tests pass; all 161 prior tests still pass; 4 real-backend tests
skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 9 — Latency Profiler (v0.9.0) ✅ COMPLETE

**Goal:** Dedicated per-backend latency profiler with warmup, high-resolution
percentiles, and a `miru profile` CLI command.

**Delivered:**
- `miru/bench/profile.py` — `profile_backend()` runs *n_warmup* discarded calls
  then *n_timed* measured calls against a fixed synth probe image; computes mean,
  std, min, max, p50, p95, p99, p99.9 and single-call throughput (calls/s);
  `ProfileResult` dataclass with `save()` / `to_dict()` and no-overwrite collision
  avoidance; no new runtime dependencies (pure NumPy + stdlib)
- `miru/cli/profile.py` — `run_profile()` CLI handler with `_format_profile()` table
  renderer
- `miru/cli/__init__.py` — `miru profile <backend>` subcommand wired in with
  `--n-warmup`, `--n-timed`, `--size`, `--seed`, `--out` flags
- `miru/bench/__init__.py` — exports `ProfileResult`, `profile_backend`
- `miru/__init__.py` — bumped to v0.9.0
- `pyproject.toml` — bumped to v0.9.0
- 20 new tests in `tests/test_profile.py`

**Ship gate:** 196/196 tests pass; all 176 prior tests still pass; 4 real-backend tests
skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 10 — Prometheus Metrics Endpoint (v1.0.0) ✅ COMPLETE

**Goal:** Expose per-backend request counts, latency histograms, and active-backend gauge in Prometheus text format via `GET /metrics`.

**Delivered:**

| Deliverable | File | Notes |
|---|---|---|
| `MiruMetrics` collector | `miru/metrics/collector.py` | Counter, Histogram, Gauge; no-op if prometheus absent |
| Metrics package | `miru/metrics/__init__.py` | `MiruMetrics`, `get_metrics()` singleton |
| `GET /metrics` endpoint | `miru/api/routes.py` | Returns 200 + empty body when prometheus absent |
| `POST /analyze` instrumentation | `miru/api/routes.py` | Wraps inference in try/finally; metrics errors log warning only |
| Optional dep group | `pyproject.toml` | `[metrics]` — `prometheus-client>=0.17.0` |
| 24 new tests | `tests/test_metrics.py` | No-op, full, API, thread-safety coverage |

**Metrics exposed:**
- `miru_requests_total{backend, status}` — Counter; status = "ok" | "error"
- `miru_latency_seconds{backend}` — Histogram; buckets: 0.01…10.0 s
- `miru_active_backends` — Gauge; distinct backends with ≥ 1 request

**Ship gate:** 233/233 tests pass (4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`). Milestone release v1.0.0.

---

## Phase 11 — Grad-CAM + Visual Demo (v1.1.0) ✅ COMPLETE

**Goal:** Implement Grad-CAM (Selvaraju et al., 2017) and ship an interactive
explainability demo that lets a developer compare attention vs. Grad-CAM on
their own images.

**Delivered:**
- `miru/gradcam.py` — `compute_gradcam(activations, gradients)` pure-NumPy
  core; `attention_to_cam(attention)` ViT fallback (multi-head collapse →
  square patch grid); `GradCAMExplainer` with auto-detected last `Conv2d`,
  forward + full-backward hooks for hook-based gradient capture, and
  graceful fallback when the model exposes no conv layers; `GradCAMResult`
  frozen dataclass.
- `POST /explain` endpoint — `method: attention | gradcam` both implemented;
  `method: lime | shap` return `501` (roadmap); unknown methods return `422`.
  Top regions carry normalised `[0, 1]` image-relative bboxes.
- `EXPLAIN_METHODS` dict — canonical implemented/roadmap registry exported
  from `miru.api.routes`.
- `demo/visual.html` — single-page interactive demo. Dark theme, three
  procedural sample images (`demo/sample_images/`), method/backend/question
  selectors, side-by-side original (with SVG bbox overlay) vs. heatmap
  overlay, top-5 regions table with score bars. Pure same-origin `fetch` to
  `/explain?overlay=true`; client-side jet-colormap canvas fallback when
  the server can't render an overlay.
- `tests/test_gradcam.py` — 22 new tests across the numeric core, the
  attention fallback, the explainer entry points, and the API endpoint
  (including the M11 ship gate: `gradcam` returns 200).
- `miru/__init__.py`, `miru/config.py`, `pyproject.toml` — version 1.0.0 →
  1.1.0; `tests/test_api.py` health-version assertion updated.

**Ship gate:** 246 / 246 tests pass; 4 real-backend tests skip without
`MIRU_TEST_REAL_BACKENDS=1`; all 224 prior tests still pass.

---

## Phase 12 — Deployable REST API (in progress)

**Goal:** Ship a deployable, dashboard-ready HTTP surface for miru's saliency
generation, benchmark scoring, and method comparison — distinct from the
in-package dev router and ready for Render / Fly / Cloud Run.

**Delivered:**
- `api/main.py` — five-endpoint FastAPI app: `/health`, `/methods`, `/explain`,
  `/benchmark`, `/compare`
- Honest `method` semantics: only `attention` is implemented; `gradcam | lime |
  shap` are reported as `roadmap` and return 400 on request
- `api/requirements.txt`, `api/Dockerfile` (non-root, `$PORT` honoured),
  `render.yaml` Web-Service manifest
- `api/test_api.py` — 13 tests with real synthetic PNG fixtures + error-contract
  coverage for malformed image, unknown model, roadmap method, unknown method,
  unknown benchmark backend, and `n`-cap

**Ship gate:** 237/237 tests pass (13 new + 224 existing); 4 real-backend tests
still skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 13 — LIME + GradCAM + side-by-side eye UI (in progress)

**Goal:** Promote two `roadmap` explainers to `implemented` behind the deployable
`/explain` surface and ship an interactive eye-UI demo that uses the new
`/explain/compare` endpoint for side-by-side method comparison.

**Delivered:**
- `miru/lime_explainer.py` — pure-NumPy LIME (Ribeiro 2016): grid-based
  superpixel segmentation, mean-colour occlusion, weighted-LSQ surrogate via
  `np.linalg.lstsq`. Deterministic under seed; no scikit-learn dependency.
- `miru/gradcam_explainer.py` — occlusion-sensitivity (Zeiler & Fergus 2014).
  The gradient-free cousin of true Grad-CAM, exposed under the `gradcam` name
  with an explicit docstring that real backprop-based Grad-CAM requires a
  torch/CNN backend (kept as future work).
- `POST /explain` dispatches on `method ∈ {attention, lime, gradcam}` —
  uniform response shape so callers stay method-agnostic.
- `POST /explain/compare` — runs two methods on one image, returns both
  overlays + grids + top regions in a single call.
- `GET /methods` reports lime + gradcam as `implemented`; only `shap` remains
  on the roadmap.
- New bounded request fields: `n_samples ≤ 256`, `n_segments ≤ 144`,
  `occlusion_grid ≤ 16` — bounds backend.infer() calls per request.
- `demo/visual.html` — single stylized eye on a deep `#06060f` field with a
  Konjo-purple iris, two counter-rotating ring layers, breathing pulse;
  image loads INTO the iris (pupil swells 130→200px); heatmap bleeds outward
  with `mix-blend-mode: screen`; method-icon tiles for gradcam / lime /
  attention; numbered amber focus rings for top-3 regions; "split view"
  divides the eye into two for `/explain/compare`.
- `tests/test_explainers.py` (8) + `api/test_api.py` (7 new) — parametrised
  `/explain` over each method, `/explain/compare` happy-path + error
  contracts, `/methods` status guard.

**Ship gate:** 252/252 tests pass (15 new, 237 existing); 4 real-backend tests
still skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 14 — TBD

Open candidates:
- True Grad-CAM run against a torch-loaded CLIP-RN50 checkpoint (RN50 has
  conv layers, unlike ViT-B/32) — drop-in `clip-rn` backend + benchmark
  comparison to occlusion-sensitivity Grad-CAM.
- SHAP perturbation explainer behind the same `/explain` surface — flip the
  last roadmap entry to `implemented`.
- Native VLM streaming backend (LLaVA / Idefics / Qwen-VL with token-level
  attention) so `/analyze/stream` produces genuinely incremental reasoning
  instead of replaying a single-shot inference.
- Real-image benchmark slice: plug VQA-X or COCO-Saliency behind the
  existing metric interface and publish the curve alongside the synthetic
  baseline.
- gRPC alternative to the FastAPI surface for in-cluster low-latency inference.
