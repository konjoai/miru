# Changelog

All notable changes to `@miru/dashboard` are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/).

## [0.1.0] — 2026-05-08

### Added — Sprint 2: Mind of the Machine

The flagship cinematic UI for miru. Watch a vision-language model decide in real time.

- **Repository scaffold** — Vite 8 + React 19 + TypeScript + Tailwind v4 + Vitest 4. Consumes `@konjoai/ui` via `file:../../konjoai-ui`. React and motion are deduped at the resolver to share one singleton.

- **Eight views**:
  - [`<AttentionField>`](./src/views/AttentionField.tsx) — the centerpiece. A self-driven canvas that bilinearly upsamples the 16×16 attention grid every frame using a Konjo cool→hot 5-stop colormap. Owns its own `requestAnimationFrame` loop; reads the grid via ref so the parent can update at any rate. Optional breathing pulse, animated scan band, click-to-focus reticle.
  - [`<ImageStage>`](./src/views/ImageStage.tsx) — composes a sample/dropped image under the AttentionField with a breathing border ring during streaming.
  - [`<ReasoningPanel>`](./src/views/ReasoningPanel.tsx) — steps stream in with severity-tinted confidence bars. The final answer rises in beneath when inference completes.
  - [`<StepTimeline>`](./src/views/StepTimeline.tsx) — scrubber + play/pause + 0.5×/1×/2× speeds. Click any position to seek.
  - [`<SampleGallery>`](./src/views/SampleGallery.tsx) — three procedurally-painted samples (still life · landscape · diagram) so first-load is interesting without binary assets.
  - [`<ImageDropzone>`](./src/views/ImageDropzone.tsx) — drag-drop or click-to-pick image input.
  - [`<QuestionBar>`](./src/views/QuestionBar.tsx) — free-form question input with Enter submission.
  - [`<MetaInspector>`](./src/views/MetaInspector.tsx) — backend · latency · attention source (interpolated → final) · hotspot coords + coverage.

- **Library layer**:
  - [`attention.ts`](./src/lib/attention.ts) — bilinear sampling · gaussian2D · grid blending · `evolvedAttention(progress, finalGrid, tMs)` synthesis. The "scan → blob → final" temporal walk that gives the dashboard its cinematic feel during streaming.
  - [`colormap.ts`](./src/lib/colormap.ts) — Konjo cool→hot 5-stop colormap (transparent → cyan → violet → hot → amber). Per-frame sampler.
  - [`sse.ts`](./src/lib/sse.ts) — manual SSE frame splitter. POST-friendly, handles keepalive comments, mid-frame buffer continuation, and `step`/`trace`/`done`/`error` decoders.
  - [`imageEncode.ts`](./src/lib/imageEncode.ts) — `canvasToRawRgbBase64` for miru's transport quirk (raw RGB bytes, not PNG).
  - [`api.ts`](./src/lib/api.ts) — `analyzeStream` (cinematic) + `analyze` (sync). Both transparently fall back to mock fixtures when the server is unreachable.
  - [`sample.ts`](./src/lib/sample.ts) — three painted samples rendered via Canvas 2D primitives.
  - [`mock.ts`](./src/lib/mock.ts) — `MOCK_STEPS`, `MOCK_TRACE`, `mockAttentionMap` with focusable peak.

- **Honest visualization** — when streaming attention is synthesized client-side (we don't have per-step attention from miru today), the `<MetaInspector>` reports `attention: interpolated` until the final trace lands, when it flips to `final`. This is animation over real data; we don't claim per-step accuracy we don't have.

- **Tests** — 46 Vitest cases covering: colormap bounds + interpolation, bilinear sampling, gaussian peaks, grid blending, coverage, evolvedAttention invariants, SSE parsing (including mid-frame split + comment frames + JSON-parse failures), mock-fixture shape invariants, and behavioral tests for `<StepTimeline>`, `<ReasoningPanel>`, `<QuestionBar>`. All green.

- **Docs** — README, CLAUDE.md (operating rules), this changelog.

### Notes

- Sprint 2 of the 10-sprint Konjo UI Initiative — the flagship the user explicitly called out as the head-turner.
- All animation respects `prefers-reduced-motion`.
- Attention evolution is purely client-side. A future backend lift could emit per-step attention deltas in the SSE `step` event; the dashboard would consume them with no surface change.
