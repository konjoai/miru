# miru/dashboard

Mind of the Machine — flagship cinematic UI for miru. Vite + React + `@konjoai/ui`. Sprint 2 of the Konjo UI Initiative.

## Stack
React 19 · TypeScript · Vite 8 · Tailwind v4 (`@theme` config) · motion · Vitest 4 · `@konjoai/ui` (file: dep)

## Commands
```bash
npm install
npm run dev          # → http://localhost:5175 (proxies /api → :8000)
npm test             # vitest (46 tests)
npm run build        # tsc -b && vite build
npm run typecheck    # tsc -b --noEmit
```

## Critical Constraints
- React, react-dom, and motion are deduped in [vite.config.ts](./vite.config.ts) so the dashboard and `@konjoai/ui` share a singleton. Don't break that.
- `@konjoai/ui` is consumed via `file:../../konjoai-ui`. Tokens come from `@konjoai/ui/styles` — don't redefine.
- Image transport quirk: miru decodes `image_b64` as RAW RGB bytes (H × W × 3), NOT a PNG. Use [`canvasToRawRgbBase64`](./src/lib/imageEncode.ts) to encode. 16×16 to match the attention grid.
- Attention evolution during streaming is synthesized client-side (scan → blob → final hotspot). The MetaInspector reports `interpolated` until the final trace arrives, then flips to `final`. Honesty over illusion.
- `<AttentionField>` owns its own `requestAnimationFrame` loop — don't add another.
- All 46 tests + the build must stay green.

## File Map
| Path | Role |
|------|------|
| `src/App.tsx` | Composition + scan state machine |
| `src/views/AttentionField.tsx` | Canvas heatmap renderer (the centerpiece) |
| `src/views/ImageStage.tsx` | Image + AttentionField overlay + breathing border |
| `src/views/ReasoningPanel.tsx` | Step-by-step trace with confidence bars + answer reveal |
| `src/views/StepTimeline.tsx` | Scrubber + play/pause/speed (0.5×/1×/2×) |
| `src/views/SampleGallery.tsx` | Procedurally-painted samples (no binary assets) |
| `src/views/ImageDropzone.tsx` | Drag-and-drop image input |
| `src/views/QuestionBar.tsx` | Question input + submit |
| `src/views/MetaInspector.tsx` | Backend · latency · attention source · hotspot |
| `src/lib/types.ts` | TS mirrors of miru/schemas.py |
| `src/lib/api.ts` | analyze + analyzeStream with mock fallback |
| `src/lib/sse.ts` | Manual SSE frame parser (POST-friendly) |
| `src/lib/attention.ts` | Bilinear sample · gaussian2D · grid blend · evolution model |
| `src/lib/colormap.ts` | Konjo cool→hot 5-stop colormap |
| `src/lib/imageEncode.ts` | Canvas → 16×16 raw RGB bytes → base64 |
| `src/lib/sample.ts` | Three painted samples (still life · landscape · diagram) |
| `src/lib/mock.ts` | MOCK_TRACE + MOCK_STEPS for offline dev |
| `src/index.css` | Imports `@konjoai/ui/styles`, adds film-grain utility |

## Backend integration
- `POST /api/analyze/stream` — SSE: `step` / `trace` / `done` / `error` events. The cinematic path.
- `POST /api/analyze` — single-shot fallback.
- Demo server CORS: open (`*`). The Vite dev proxy is purely for ergonomic relative paths.
- Future backend lift (deferred): emit per-step attention deltas in the `step` event so the heatmap is honest per-step. Today we synthesize a walk from a uniform start to the final hotspot. Document any backend change here when it lands.

## When extending
- New view? Lives in `src/views/`. Always ship a Vitest test.
- New backend shape? Mirror types in [src/lib/types.ts](./src/lib/types.ts), add a mock fixture, then add the API method to [src/lib/api.ts](./src/lib/api.ts) with a mock fallback.
- New colormap stop? Update both [src/lib/colormap.ts](./src/lib/colormap.ts) and the README. The default is intentionally on-brand cool→hot.
- New design token? Add to `@konjoai/ui` (so all flagships inherit), not here.

## Sprint context
This is **Sprint 2** of the 10-sprint Konjo UI Initiative — the flagship the user explicitly called out. Sprint 0 = `@konjoai/ui` foundation. Sprint 1 = squash Compliance Bridge (calendar critical). Sprint 3 = kairu Speed Cockpit.
