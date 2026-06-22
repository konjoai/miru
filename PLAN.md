# PLAN.md — Miru Roadmap

**Project:** Miru — Multimodal Reasoning Tracer  
**Current version:** v1.11.0  
**Status:** Qwen3-VL real backend (Phase 28), EU AI Act compliance harden (Phase 27), synergistic-faithfulness probe / F_syn (Phase 26), explanation alert rules (Phase 25), 767 tests passing (9 skipped without MIRU_TEST_REAL_BACKENDS=1)

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

## Researched Feature Roadmap

Prioritised features compiled from a 2026 explainability-tooling sweep
(arXiv saliency literature, EU AI Act compliance deadlines, expert
annotation studies, deployed XAI tooling). Each tier is independently
shippable.

### 🔴 P1 — Critical (sprint targets)

- **Explanation fidelity scorecard.** After generating a saliency map,
  run a deletion test: mask the top-K% most salient pixels and re-run
  inference. Score = drop in model confidence. Report `fidelity_score`
  ∈ [0, 1] alongside every heatmap; UI warns when < 0.5 ("this
  explanation may not reflect model reasoning").
- **Multi-method consensus overlay.** Run 2+ saliency methods
  simultaneously (occlusion + LIME minimum; integrated gradients when
  a gradient backend is available). Overlay shows where methods agree
  vs. disagree; disagreement regions are flagged.
  `POST /explain/consensus` with `methods=[…]`.
- **EU AI Act compliance report generator.** `GET /report/{analysis_id}/eu_ai_act`
  returns a structured report covering Article 11 (technical documentation),
  Article 13 (transparency), Article 15 (accuracy & robustness). Auto-fills
  from recorded analyses. **Compliance deadline: August 2026.**
- **Explanation export.** `GET /analysis/{id}/export?format=png|json|pdf`.
  PNG: heatmap at 2×. JSON: full saliency record. PDF: report-ready
  page (overlay + metadata header).

### 🟠 P2 — High impact, medium complexity

- **Multi-method consensus (full).** Integrated Gradients as a third
  method; consensus score = Jaccard similarity of top-20% salient
  regions across all methods.
- **Expert annotation alignment.** `POST /analyze/{id}/compare_annotation`
  accepts a ground-truth mask. Returns spatial IoU, Spearman rank
  correlation, "right answer / wrong reasoning" flag when prediction
  is correct but alignment < 0.3.
- **Dataset-level saliency analytics.** `POST /analyze/batch` over an
  array of images. Aggregate heatmap; spurious-correlation detector
  flags regions consistently salient but not semantically meaningful
  (borders, watermarks).
- **Cross-modal attention tracer.** `POST /trace` returns a per-word →
  image-region attention matrix. Click a word in the UI, see the
  region it attends to.

### 🟡 P3 — Strategic

- **Counterfactual generation.** Minimal adversarial perturbation that
  flips the verdict.
- **Concept-based explanations (TCAV-style).** Probe whether the model
  uses named concepts ("fur texture", "metallic surface", "text").
- **Saliency API SDK.** Python + JS clients; batch analysis with an
  async job queue.

---

## Phase 14 — SHAP Perturbation Explainer (v1.1.0) ✅

**Goal:** Add a SHAP-style tile-masking attribution explainer, promote `shap`
from roadmap → implemented, and ship 18 new tests.

**Delivered:**
- `miru/shap_explainer.py` — `SHAPConfig` (dataclass) + `SHAPExplainer` class.
  Implements the Shapley approximation:
  φᵢ ≈ (1/M) Σⱼ [ f(S_j ∪ {i}) − f(S_j) ] via tile-mask sampling.
  Pure NumPy + PIL; no `shap` library dependency. Attribution in [-1, 1]
  float32; bilinear upsample to full image resolution.
- `api/main.py` — `shap` promoted from `ROADMAP_METHODS` to
  `IMPLEMENTED_METHODS`; `_run_method` dispatches to `SHAPExplainer`; request
  schema extended with `shap_grid` and `shap_samples`.
- `tests/test_shap_explainer.py` — 18 tests: config defaults, shape/dtype/range
  contracts, determinism, baseline variants, masked-image invariants, registry
  and API integration.

**Ship gate:** 281/281 tests pass (18 new, 263 existing); 4 real-backend tests
still skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 15 — P1 Critical Sprint ✅

**Goal:** Ship the four critical features from the researched roadmap so
the explainability surface gives audit-quality, regulator-ready output.

**Delivered:**
- `miru/fidelity.py` — deletion test: mask top-K% salient pixels with
  per-image mean colour, re-run `backend.infer()`, compute
  `fidelity_score = max(0, (baseline_conf - masked_conf) / baseline_conf)`.
  Returns `FidelityResult{ fidelity_score, baseline_confidence,
  masked_confidence, k_pct, low_fidelity (< 0.5) }`. Pure NumPy.
- `miru/consensus.py` — multi-method agreement/disagreement.
  `compute_consensus(maps, top_pct=0.20)` returns an `agreement_grid`
  whose value is the fraction of methods including each cell in their
  top-pct, plus per-pair Jaccard scores, a mean `consensus_score`, and
  the explicit `disagreement_regions` flagged for the UI.
- `miru/eu_ai_act.py` — structured report builder mapping Article 11
  (technical documentation), Article 13 (transparency), Article 15
  (accuracy & robustness) onto fields present in a recorded analysis.
  Includes a `compliance_status` block flagging missing fields.
- `miru/export.py` — PNG (heatmap colorized at 2×), JSON (full
  recorded record), and PDF (single-page Pillow document with overlay
  + metadata header). Pure-zlib PNG path when Pillow is unavailable.
- `miru/recorder.py` — `build_record()` now generates a UUID v4
  `analysis_id` when none is supplied; new `find_record_by_id` helper
  scans recorded JSONL across files and returns the matching record.
- `api/main.py` — `POST /explain?fidelity=true` adds a `fidelity` block
  to the response; new `POST /explain/consensus`, `GET
  /report/{analysis_id}/eu_ai_act`, and `GET /analysis/{id}/export?format=…`.

---

## Phase 16 — Batch Explain + Content-Addressed Cache ✅

**Goal:** Ship the two highest-leverage P2 items: a batch endpoint so callers
don't pay HTTP overhead per image, and a content-addressed cache so repeated
analyses are instant.  Method-comparison was already shipped under
`/explain/compare` + `/explain/consensus` in earlier phases — not re-built.

**Delivered:**
- `miru/explain_cache.py` — SQLite-backed `ExplainCache` keyed on
  SHA-256 of `(image_b64, method, model_name, params)`.  Singleton
  accessor `get_cache()` reads `MIRU_CACHE_PATH` (default
  `./miru_cache.db`) and `MIRU_CACHE_ENABLED` (default on); both env
  vars are read at construction so tests can override per-fixture.
  Schema: `explanation_cache(key, payload, method, model_name,
  created_at, hit_count)` + `cache_meta(name, value)` for cumulative
  hit/miss totals.  Thread-safe via short-lived connections + a
  per-instance lock.  Corrupt payloads self-heal (deleted on first
  bad read, re-populated on next miss).
- `api/main.py` — `_run_explain_with_cache()` wraps the existing
  compute path.  Cache **misses** record + populate as before.  Cache
  **hits** still call `maybe_record()` so each `/explain` call is its
  own audit event with its own `analysis_id` — only the heavy
  computation (saliency grid, overlay, top regions, fidelity block)
  is reused.  Cache hits return the observed lookup latency, not the
  stored compute latency, so clients can see the real speedup.
  `X-Miru-Cache: hit|miss|bypass` response header makes cache state
  observable without parsing the JSON.
- `POST /explain` — new `use_cache: bool = true` query param so
  callers can force re-computation.
- `POST /explain/batch` — accepts 1..32 items, runs them sequentially
  through the cache, returns per-item results plus an `aggregate`
  block (`total`, `success_count`, `failure_count`, `cache_hits`,
  `cache_misses`, `mean_confidence`, `mean_fidelity`,
  `total_latency_ms`).  One bad item doesn't abort the batch unless
  `stop_on_error=true`; failed items return an `error` string.
- `GET /explain/cache_stats` — `{enabled, path, total_entries,
  total_hits, total_misses, hit_rate, size_bytes, per_method}`.
- `POST /explain/cache_clear` — drops every entry, resets counters.

**Tests:**
- `tests/test_explain_cache.py` — 19 unit tests: key determinism + 4
  partitioning axes, get/put round-trip, hit-count column, corrupt
  payload self-heal, uncacheable payload skip, stats initial state +
  after-traffic + per-method breakdown, clear, env-var truthy/falsy
  matrix, `None` when disabled, singleton identity, reset.
- `api/test_batch_and_cache.py` — 19 HTTP tests: first call miss,
  second call hit, partition by method, partition by param,
  `use_cache=false` bypass, env-disable bypass, stats reflects
  traffic, stats when disabled, clear endpoint, batch happy path with
  three distinct images, batch warm-cache reports all-hits, batch
  preserves order, fidelity flag propagates uniformly, batch with
  mixed methods, one bad item does not fail others, `stop_on_error`
  aborts remainder, empty `items` 422, oversized 422, single item.

**Ship gate:** 387 / 387 passing (322 baseline → +65 new); 5 skipped
(4 CLIP real-backend + 1 pre-existing).

---

## Phase 17 — Cross-Modal Attention Tracer (v1.2.0) ✅ COMPLETE

**Goal:** Ship a word→image-region attribution matrix so developers can see
which tokens in a question drive attention to which spatial regions.

**Delivered:**
- `miru/cross_modal.py` — `CrossModalTracer` + `CrossModalTrace` dataclass.
  Perturbation-based: for each whitespace token, ablate the word from the
  question, measure the positive shift in the spatial attention map, and
  min-max normalise to `[0, 1]`. Backend-agnostic, no gradients required.
  Returns `matrix: (n_words, grid_h × grid_w)` float32 + baseline
  `full_attention: (grid_h, grid_w)`.
- `POST /trace` in `api/main.py` — `TraceRequest` / `TraceResponse` schemas;
  `model_name`, `question`, `image_b64` in; `words`, `matrix`, `grid_h`,
  `grid_w`, `full_attention`, `latency_ms` out.
- `tests/test_cross_modal.py` — 22 tests: word count, matrix shape/dtype/range,
  full-attention shape/dtype/range, empty-question edge case, single-word edge
  case, determinism, inter-question variation, `_normalise_row` helper,
  API happy-path, response shape, matrix values, full-attention shape/values,
  empty question 200, unknown-model 400, bad-image 400, model-name echo,
  health regression guard.

**Ship gate:** 409/409 tests pass; all 387 prior tests still pass; 5 skipped.

---

## Phase 18 — Expert Annotation Alignment (v1.3.0) ✅ COMPLETE

**Goal:** Let developers supply a human-drawn ground-truth mask and receive
concrete spatial alignment scores and a "right answer, wrong reasoning" flag.

**Delivered:**
- `miru/annotation.py` — `compare_annotation(saliency, mask, *, answer_correct, top_pct)`
  returns `AnnotationAlignment{iou, auc_roc, spearman_r, top_pct, misaligned}`.
  IoU reuses the existing `iou_at_topk_pct` harness.  AUC-ROC reuses `auc_roc`.
  Spearman rank correlation is pure NumPy via `_rank` + `_spearman` helpers.
  `misaligned = answer_correct AND iou < MISALIGN_THRESHOLD (0.3)`.
- `POST /annotate` in `api/main.py` — `AnnotateRequest` / `AnnotateResponse`;
  full explain fields (overlay, attention_grid, top_regions, answer, confidence)
  plus an `AlignmentBlock`. Mask validated: non-empty, rectangular, ≤ 512×512.
  Returns 400 on bad image, unknown model, unknown method, empty/jagged/oversized mask.
- `tests/test_annotation.py` — 32 tests: unit (perfect/inverted/uniform alignment,
  Spearman sign, misaligned flag scenarios, resolution mismatch, error paths,
  helper unit tests) + API (happy path, alignment block presence, value ranges,
  misaligned flag, all error contracts, lime method, top_pct round-trip, health).

**Ship gate:** 441/441 tests pass; all 409 prior tests still pass; 5 skipped.

---

## Phase 19 — Dataset-Level Saliency Analytics (v1.4.0) ✅ COMPLETE

**Goal:** Aggregate saliency maps across a dataset of images, produce a
mean heatmap, and flag spurious-correlation candidates.

**Delivered:**
- `miru/dataset_analytics.py` — `aggregate_saliency()`, `detect_spurious()`,
  `analyse_dataset()` + `DatasetAnalytics` frozen dataclass.  All grids are
  bilinearly resampled to the first grid's shape.  Spurious detection:
  mean ≥ `SPURIOUS_MEAN_THRESHOLD (0.5)` AND CV < `SPURIOUS_CV_THRESHOLD (0.5)`;
  suppressed when n < `MIN_SAMPLES_FOR_SPURIOUS (3)`.
- `POST /analyze/batch` in `api/main.py` — `DatasetAnalyticsRequest` (1–64
  images, per-image question, shared model/method/params) /
  `DatasetAnalyticsResponse` (mean_grid, std_grid, cv_grid, spurious_cells,
  per_image results, latency_ms).
- `tests/test_dataset_analytics.py` — 29 tests: unit (aggregate shape/dtype/
  mean/std contracts, spurious flag logic, threshold edge cases, empty-list
  error, analyse_dataset pipeline) + API (happy path, response shape, index
  order, value ranges, single-image no-spurious, error contracts).

**Ship gate:** 470/470 tests pass; all 441 prior tests still pass; 5 skipped.

---

## Phase 20 — History · Calibration · Diff ✅ COMPLETE

**Goal:** Open up the recorded explanation store for query, surface a
quantitative trust signal (ECE), and let users diff two past analyses
post-hoc.  Three composing endpoints: history is the foundation,
calibration is one aggregation over it, diff is a two-record
operation that reuses the lookup primitives.

**Delivered:**
- `miru/history.py` — `query_records()` (filters: method, model,
  min_confidence, since; limit 1..200, offset ≥ 0; newest-first;
  drains the recorder before scanning) and `compute_calibration()`
  (ECE + per-bin reliability curve; skips records without fidelity;
  clamps out-of-range values; validates n_bins ∈ 2..50; empty
  population returns ece=0.0).
- `miru/diff.py` — `diff_records(rec_a, rec_b, top_n)`: bilinear-
  align grids, cosine similarity on raw vectors (sign + magnitude
  preserved), L2 on min-max normalised grids, signed delta grid,
  top-N changed cells, and a human-readable summary using a 3×3
  spatial grid ("A focused more on the bottom-left; B shifted toward
  the top-right").  Handles flat / degenerate grids without NaN.
- `GET /explain/history` — paginated filtered listing; strips the
  bulky `attention_grid` / `top_regions` per row (fetch full via
  `/analysis/{id}/export?format=json`).
- `GET /explain/calibration` — pulls up to `limit` recent records
  filtered by method/model, keeps those with fidelity, returns ECE
  + bins + echoed filters.
- `POST /explain/diff` — body `{analysis_id_a, analysis_id_b, top_n}`;
  400 on identical IDs, 404 on missing record.
- 52 new tests: 22 in `tests/test_history.py`, 13 in
  `tests/test_diff.py`, 17 in `api/test_history_diff_calibration.py`.

**Ship gate:** 491 / 491 passing post-rebase onto main (after the
parallel Phases 17/18/19 — cross-modal / annotation / analytics — also
landed). 5 skipped. No regressions.

**Note on numbering:** committed locally as "Phase 17" before
rebase; renumbered to **Phase 20** here because main shipped Phases
17 / 18 / 19 in parallel.  See CHANGELOG for the same renumbering.

---

## Phase 21 — Scale-Space Attention Ensemble (v1.5.0) ✅ COMPLETE

**Goal:** Run inference at multiple image scales and average the attention
maps for more robust saliency — directly addressing the discovery finding that
single-scale cross-attention captures only 52–75% of the true saliency signal
(arXiv 2509.22415).

**Delivered:**
- `miru/ensemble.py` — `AttentionEnsemble` class with configurable scales and
  weights; `_bilinear_resize_image` pure-NumPy resize helper; `EnsembleResult`
  frozen dataclass. Scales below `MIN_DIM (4px)` silently skipped; all-fail
  case returns all-zero grid with warning. Final grid re-normalised to `[0, 1]`.
- `POST /explain/ensemble` in `api/main.py` — `EnsembleRequest` (scales 1–5,
  optional weights, standard overlay/colormap/top_k params) /
  `EnsembleResponse` (ensemble_grid, per_scale, scales_used, scales_skipped,
  top_regions, overlay_b64, latency_ms). Validates scale range (0, 4] and
  weight-length match.
- `tests/test_ensemble.py` — 27 tests: resize helper (shape, dtype, range,
  too-small → None), `AttentionEnsemble` unit (single-scale, value range,
  scales_used/skipped tracking, per_scale count, custom weights, empty scales
  error, mismatch error, all-fail zeros, resolution forwarding), API (happy
  path, response fields, grid values, per_scale count, error contracts, echo).

**Ship gate:** 497/497 tests pass; all 491 prior tests still pass; 5 skipped.

---

## Phase 22 — Model Comparison · Post-hoc Consensus · Search ✅ COMPLETE

**Goal:** Three composing read-side endpoints that operate on the
recorded explanation store opened up by Phase 20. Model comparison
aggregates across models; post-hoc consensus combines existing
analyses without rerunning; search finds historical analyses with
similar attribution patterns.

**Delivered:**

- `miru/model_comparison.py` — `compare_models(models, *, limit, method)`
  pulls per-model history slices and aggregates `n_records`,
  `mean_confidence`, `mean_latency_ms`, `mean_fidelity`,
  `n_with_fidelity`, `ece` (via `compute_calibration`), and
  `method_distribution`.  Reports three winner verdicts: by mean
  confidence (higher wins), by mean fidelity (higher wins), by ECE
  (lower wins).  Each winner is `None` when no model has data for
  that metric.

- `miru/posthoc_consensus.py` — `build_consensus(records, *, weighting, top_k)`
  takes already-recorded dicts (typically obtained via
  `find_record_by_id` per ID), bilinearly aligns all attention
  grids to the max shape, computes a weighted average and a
  per-record `agreement_score` (cosine vs. the consensus). Three
  weighting modes: `fidelity` (default; missing fidelity → floor at
  population min; all-missing falls back to uniform), `confidence`,
  `uniform`. Distinct from `miru.consensus.compute_consensus`
  (Phase 13) which takes a fresh image + methods and runs them
  live; this combines records that already ran.

- `miru/search.py` — `search_by_pattern(query_grid | query_analysis_id,
  *, method, model, top_k, min_similarity, max_scan)`. Exact cosine
  search over the recorded JSONL store, newest first within the
  scan budget. Bilinearly aligns candidate grids to the query's
  shape so methods that produce different resolutions can be
  compared. Self-exclusion when querying by `analysis_id`.

- `api/main.py` — three new endpoints:
  - `GET /explain/models/compare?models=A,B&limit=50&method=...`
  - `POST /explain/consensus/by_ids` — body
    `{analysis_ids[2..16], weighting, top_k}`
  - `POST /explain/search` — body
    `{query_grid | query_analysis_id, method, model, top_k,
    min_similarity, max_scan}`

- New boundary constants in `api/main.py`: `MAX_COMPARE_MODELS=8`,
  `MAX_POSTHOC_IDS=16`, `MAX_SEARCH_TOP_K=50`, `MAX_SEARCH_SCAN=2000`.

**Tests:**
- `tests/test_model_comparison.py` — 12 unit tests: argument
  validation (empty / duplicate / invalid limit), empty store,
  single-model aggregate, fidelity/ECE round-trip, method
  distribution, multi-model winners by confidence and ECE, method
  filter, limit cap, no-data winners.
- `tests/test_posthoc_consensus.py` — 16 unit tests: argument
  validation (empty / unknown weighting / invalid top_k / missing
  grid / empty grid), math correctness (uniform = simple mean,
  identical = unit agreement, outlier has lower agreement),
  weighting modes (fidelity dominance, fallback to uniform, floor
  for missing, confidence dominance, zero-confidence fallback),
  shape alignment, top-region ordering, metadata round-trip.
- `tests/test_search.py` — 18 unit tests: argument validation
  (neither / both queries / invalid top_k / max_scan /
  min_similarity / unknown query_analysis_id), basic behaviour
  (empty source, exact match, sort order, self-exclusion), filters
  (method, model, min_similarity, top_k cap), shape alignment,
  candidate-without-grid skip, max_scan cap, metadata round-trip.
- `api/test_phase22_endpoints.py` — 20 HTTP tests covering every
  endpoint × every error contract.

**End-to-end verified** through `demo/server.py`:
- `GET /api/v2/explain/models/compare?models=mock&limit=10` →
  aggregate stats + winners across confidence, fidelity, ECE
- `POST /api/v2/explain/consensus/by_ids` → 16x16 consensus grid,
  per-record agreement scores, top-K consensus regions
- `POST /api/v2/explain/search` correctly excludes the query
  analysis_id from results, scores by cosine similarity

**Ship gate:** **615 / 615 passing** post-rebase (549 baseline → +66
new); 5 skipped.  No regressions.

---

## Phase 23 — Input Sensitivity / Robustness (v1.6.0) ✅ COMPLETE

**Goal:** Quantify whether an explanation is *trustworthy* — does it stay put
when the input barely moves? Fragile saliency that relocates under
imperceptible Gaussian noise is tracking noise, not signal (Ghorbani et al.
2019). This phase adds a method-agnostic robustness probe.

**Delivered:**
- `miru/sensitivity.py` — pure-NumPy, explainer-agnostic. `compute_sensitivity`
  takes a `saliency_fn` (image → 2-D grid) so it works uniformly across every
  method (`attention`/`lime`/`gradcam`/`shap`). Sweeps seeded Gaussian noise at
  each σ, re-runs the explainer `n_trials` times per σ, measures mean absolute
  attribution drift, and returns `stability_score = 1 − mean_drift`, the worst
  σ, and an `is_stable` verdict. `baseline_grid` short-circuit avoids a
  redundant (potentially expensive) clean-image explainer run. Fully
  deterministic under a seed.
- `POST /explain/sensitivity` (`api/main.py`) — `SensitivityRequest` carries the
  same explainer knobs as `/explain` plus `sigmas` (≤ 8, each in (0, 1]),
  `n_trials` (≤ 8, bounds backend.infer fan-out), `seed`, `stability_threshold`.
  Drives all four methods via the existing `_run_method` so the robustness
  measured is exactly the map the API would return. 400 on bad sigma / unknown
  method / unknown model / undecodable image.
- 24 new tests: `tests/test_sensitivity.py` (12 unit — drift math, blind-vs-
  image-dependent saliency, determinism, threshold, baseline short-circuit) +
  `api/test_sensitivity.py` (12 HTTP — contract, per-σ fields, mock-attention
  stability, determinism, gradcam path, all validation rejections).

**Design notes / pushback:** the standalone exploratory version of this feature
also reintroduced a history store, model-comparison, and pattern-search — but
those already shipped in Phases 20/22 (`miru/history.py`,
`miru/model_comparison.py`, `miru/search.py`). Per "don't re-implement
anything," those were dropped and only the genuinely-novel robustness probe was
kept, integrated into the existing `_run_method` dispatch rather than a parallel
explainer path.

**Ship gate:** 639/639 tests pass (24 new, 615 existing); 5 skipped (4
real-backend + 1 other). `miru/sensitivity.py` clean on ruff/radon(A)/vulture
and 150 lines.

---

## Phase 24 — ROI-targeted explanation (v1.7.0) ✅ COMPLETE

**Goal:** Let callers restrict saliency computation to a user-defined
sub-region of the image, confining attribution to the area of interest.

**Delivered:**
- `BoundingBox` Pydantic model in `api/main.py` — relative `[0, 1]`
  coordinates with cross-field `model_validator` enforcing `x2 > x1`
  and `y2 > y1`.
- `roi: BoundingBox | None = None` field on `ExplainRequest` (backward-
  compatible; defaults to `None`).
- `_apply_roi_saliency(full_image, roi, method, backend, req)` — crops
  the image to the bbox, runs the chosen explainer on the crop, resizes
  the resulting grid into the corresponding cells of a full-resolution
  zero grid.  The VLM answer/confidence always come from the full image.
  Works for all four methods (attention, lime, gradcam, shap).  Raises
  HTTP 400 when the crop maps below 4×4 pixels.
- `roi` included in `_explain_cache_key` so different sub-regions never
  collide.
- `api/test_roi.py` — 13 HTTP tests; `tests/test_roi.py` — 6 unit tests
  for `BoundingBox` validation.

**Ship gate:** 657/657 tests pass (+19 new); ruff clean.

---

## Phase 25 — Explanation Alert Rules (v1.8.0) ✅ COMPLETE

**Goal:** Fire webhooks when an `/explain` result crosses a user-defined
rule — anomaly detection over the live explanation stream.

**Delivered:**
- `miru/alerts.py` — SQLite-backed `AlertStore` with CRUD over `Rule`s
  (threshold comparisons on `confidence` / `fidelity_score` and the
  `low_fidelity` flag), evaluation into `FiredAlert`s, and asynchronous
  webhook delivery with SSRF-guarded URL validation (`validate_webhook_url`).
- `POST /explain` wires `_evaluate_and_fire_alerts` (non-blocking; delivery
  never breaks the request path). Runtime `miru_alerts.db` is git-ignored.
- `api/test_alerts.py` — rule lifecycle, evaluation, and delivery coverage.

**Ship gate:** shipped + merged (PR #13). Version bookkeeping reconciled in
Phase 26 (the commit claimed v1.8.0 but never bumped the version files).

---

## Phase 26 — Synergistic-Faithfulness Probe / F_syn (v1.9.0) ✅ COMPLETE

**Goal:** Distinguish faithful cross-modal reasoning from visual-only
salience. The deletion test (Phase 15) measures *whether* salient pixels
matter; it can't tell whether they matter *because of the question* or
regardless of it. Recent work shows deletion/insertion AUC systematically
over-credits the visual-only case (Cross-Modal Synergy benchmark,
arXiv:2509.22415 / 2605.22168).

**Delivered:**
- `miru/synergy.py` — `synergy_test()` measures the modality-level Shapley
  *interaction*: the discrete mixed second difference of model confidence
  over presence/absence of the salient visual region (`V`) and the question
  (`Q`). `interaction = f_both − f_language_only − f_vision_only + f_neither`;
  `synergy_score = clamp(interaction / max(eps, f_both), 0, 1)`; `low_synergy`
  flag below `0.3`. Pure NumPy, deterministic, no new deps; reuses
  `miru.fidelity._mask_top_k` for the visual ablation. The image-independent
  mock backend reports exactly zero synergy (honest signal). Cites Grabisch &
  Roubens 1999, Janizek et al. 2021.
- `POST /explain?synergy=true` — `SynergyBlock` on the response; three extra
  `backend.infer()` calls (off by default). Folded into `_explain_cache_key`
  and `BatchExplainRequest`. Works across all four methods.
- `api/conftest.py` — shared `png_b64` fixture (net DRY reduction).
- Reconciled the three drifting version sources (`pyproject.toml`,
  `miru/__init__.py`, `miru/config.py`) and the `/health` version assertion.
- 23 new tests: `tests/test_synergy.py` (13) + `api/test_synergy.py` (10).

**Ship gate:** 733/733 tests pass (+23 new); 5 skipped; ruff clean;
`miru/synergy.py` radon-A, zero new DRY violations.

---

## Phase 27 — EU AI Act Compliance Harden (v1.10.0) ✅ COMPLETE

**Goal:** Bring the compliance report up to the artifacts the AI Act
high-risk obligations require from **Aug 2 2026**. The report covered
Articles 11/13/15; this adds the missing record-keeping, right-to-explanation,
documented feature-importance, and robustness evidence.

**Delivered:**
- `miru/eu_ai_act.py` — added **Article 12** (record-keeping/logging) and
  **Article 86** (right to explanation: plain-language, person-facing rationale
  citing the most influential region + contestability note); **documented
  feature importance** (ranked top-5 regions) in Article 13; **synergy-aware
  robustness** in Article 15 (surfaces the Phase 26 probe; `visual_only_salience`
  risk when `low_synergy` fires). `compliance_status` extended to Arts. 12/86;
  `REPORT_VERSION = "1.1"`.
- Refactored `generate_report` C(19) → A(1) (orchestrator + per-article
  helpers); every function now grade B or better — removed a pre-existing
  complexity-gate smell while adding the features.
- `GET /report/{id}/eu_ai_act` docstring updated to list all five articles.
- 13 new tests (12 unit + 1 end-to-end); 100% coverage on `miru/eu_ai_act.py`.

**Ship gate:** 748 tests (743 passed, 5 skipped); ruff/format clean; zero new
DRY violations; radon all ≤ B.

---

## Phase 28 — Qwen3-VL Real Backend (v1.11.0) ✅ COMPLETE

**Goal:** Ship miru's first *generative* VLM backend with genuine
cross-modal attention. CLIP is a dual-encoder that only scores
image/text similarity; Qwen3-VL (Alibaba, Sept 2025) reasons over the
question tokens and image patches jointly, so its saliency is what the
synergy probe (Phase 26) and deletion test (Phase 15) are designed to
interrogate.

**Delivered:**
- `miru/models/qwen3vl.py` — `Qwen3VLBackend` mirrors the `CLIPBackend`
  lazy-load contract (weights load on first `infer()`, never at import).
  Attention is read from a **middle decoder layer** (cross-modal fusion
  peaks mid-stack — Qwen2.5-VL report arXiv:2502.13923, layers ~14-24),
  last-prompt-token → image-token attention (located via
  `config.image_token_id`), head-averaged and reshaped to a square grid.
  Confidence = first-generated-token softmax probability. `eager`
  attention impl (required for `output_attentions`).
- The verifiable numeric logic is isolated in pure helpers
  (`_select_middle_layer`, `_attention_row_to_grid`) so it is unit-tested
  fully offline; the model-load + generation path is gated behind
  `MIRU_TEST_REAL_BACKENDS=1`, exactly like CLIP.
- `miru/models/registry.py` — registers `qwen3vl` in `register_defaults()`.
- `pyproject.toml` — `[backends]` bumped to `transformers>=4.57.0` (the
  minimum that ships Qwen3-VL natively).
- 19 new tests (`tests/test_qwen3vl_backend.py`): 5 structural, 10 pure-
  helper, 4 gated real-inference.

**Ship gate:** 767 tests (758 passed, 9 skipped); ruff/format clean; zero
new DRY; radon all grade A.

---

## Phase 29 — TBD

Open candidates (P2/P3 from the researched roadmap, plus deferred items):
- ~~Qwen3-VL real backend~~ ✅ shipped in Phase 28.
- ~~EU AI Act compliance report generator~~ ✅ hardened in Phase 27.
- ~~Explanation alerts / anomaly detection~~ ✅ shipped in Phase 25.
- ~~Synergistic-faithfulness probe (F_syn)~~ ✅ shipped in Phase 26.
- ~~Region-of-interest (ROI) targeted explanation~~ ✅ shipped in Phase 24.
- ~~Input sensitivity analysis~~ ✅ shipped in Phase 23.
- ~~Expert annotation alignment (P2)~~ ✅ shipped in Phase 18.
- ~~Dataset-level saliency analytics (P2)~~ ✅ shipped in Phase 19.
- ~~Cross-modal attention tracer (P2)~~ ✅ shipped in Phase 17.
- Intra-modal + cross-modal joint attribution (informed by arXiv 2509.22415) —
  upgrade the attention extractor to sum intra-visual token interactions with
  the cross-modal signal for more faithful maps (Medium complexity).
- True Grad-CAM via torch-loaded CLIP-RN50 (P3).
- Sparse Autoencoder (SAE) concept-based explanations via Prisma (arXiv 2504.19475)
  for the EU AI Act report narrative (P3).
- Native VLM streaming backend (LLaVA / Qwen-VL, token-level attention) (P3).
- Real-image benchmark slice — VQA-X or COCO-Saliency + Saliency-Bench
  (arXiv 2310.08537v3) as a public fidelity anchor (P3).
- Counterfactual generation, TCAV / Visual-TCAV concept probes (P3).
- gRPC surface for in-cluster low-latency inference (P3).
