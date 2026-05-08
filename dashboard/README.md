# miru · Mind of the Machine

A flagship Konjo UI for **miru** — the multimodal reasoning tracer.

> 見る (miru) — to see · 観る — to watch · 視る — to look closely

Watch a vision-language model decide. Reasoning streams in token-by-token. The attention field morphs as the model focuses. Click anywhere on the image to drop a focus marker and ask *why did it look here?*

## Quick start

```bash
npm install
npm run dev      # → http://localhost:5175
npm test         # vitest (46 tests)
npm run build    # production build → dist/
```

To wire the dashboard to a live miru backend:

```bash
# Terminal 1 — start the demo server (port 8000)
cd /Users/wesleyscholl/miru
python demo/server.py

# Terminal 2 — start the dashboard (proxies /api → :8000)
cd dashboard
npm run dev
```

When the server is unreachable the dashboard transparently replays a mock trace so the cinematic experience is always whole.

## Stack

`React 19` · `TypeScript` · `Vite 8` · `Tailwind CSS v4` · `motion` · `Vitest`
Built on top of [`@konjoai/ui`](../../konjoai-ui) — the shared design system for the KonjoAI portfolio.

## What you'll see

| Panel               | What it shows                                                          |
|---------------------|------------------------------------------------------------------------|
| **Hero**            | The miru promise · cyan/violet gradient                                |
| **Image stage**     | Image + bilinearly-upsampled attention heatmap + breathing scan band   |
| **Reasoning panel** | Steps stream in with confidence bars · final answer rises in below     |
| **Step timeline**   | Scrubber + play/pause + 0.5×/1×/2× playback after stream finishes      |
| **Question bar**    | Free-form question · Enter to submit                                   |
| **Sample gallery**  | Three procedurally-painted samples (still life · landscape · diagram)  |
| **Image dropzone**  | Drag-and-drop your own image                                           |
| **Meta inspector**  | Backend · latency · attention source (interpolated → final) · hotspot  |

## Architecture

- **`<AttentionField>`** is the centerpiece — a self-driven `<canvas>` that
  bilinearly upsamples the attention grid every frame using a Konjo
  cool→hot colormap. The grid is held in a ref so the parent can update it
  at any rate; a `requestAnimationFrame` loop reads the latest on every
  paint without re-rendering React.

- **Attention evolution** ([src/lib/attention.ts](./src/lib/attention.ts)). miru
  produces one final attention grid per inference. To create the *feel* of
  attention evolving during streaming, we synthesize a walk: scan band →
  drifting blob → focused on the final hotspot. The MetaInspector reports
  `attention: interpolated` until the final trace arrives, when it flips to
  `final`. We are explicit about this — it is animation over real data, not
  a fabrication of per-step accuracy.

- **SSE parser** ([src/lib/sse.ts](./src/lib/sse.ts)). Manual fetch + frame
  splitter (the native `EventSource` forces GET; miru's stream is a POST).
  Handles `step` / `trace` / `done` / `error` events plus keepalive
  comments.

- **Image transport** ([src/lib/imageEncode.ts](./src/lib/imageEncode.ts)).
  miru's `/api/analyze` decodes `image_b64` as raw RGB bytes (3 channels),
  not as a PNG. The dashboard mirrors the demo's workaround: render to a
  16×16 canvas, drop alpha, base64-encode the bytes.

## Configuration

- `VITE_MIRU_API` — base URL of the miru API (default: `""`, leans on the
  Vite dev proxy or relative paths in production).
- The dev server proxies `/api` → `http://localhost:8000` (see [`vite.config.ts`](./vite.config.ts)).

## Tests

```bash
npm test
```

Covers: colormap bounds, bilinear sampling, gaussian peaks, grid blending,
attention coverage, evolved-attention progress invariants, SSE frame
parsing (including mid-frame buffer continuation and keepalive comments),
mock-fixture invariants, and behavioral tests for `<StepTimeline>`,
`<ReasoningPanel>`, `<QuestionBar>`. 46 tests, all green.

See [`CLAUDE.md`](./CLAUDE.md) for operating rules.
