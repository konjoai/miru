/**
 * miru API client — the SSE bridge.
 *
 * Two transports:
 *   1. analyzeStream(...) — fetch + stream parser, the cinematic path.
 *   2. analyze(...)       — single-shot POST; useful for reload / replay.
 *
 * Both fall back to mocks when the server is unreachable so the dashboard
 * is always demonstrable.
 */
import type { AnalyzeRequest, ReasoningTrace, SseEvent } from "./types";
import { parseFrames, frameToSseEvent } from "./sse";
import { MOCK_STEPS, MOCK_TRACE } from "./mock";

const BASE = (import.meta.env.VITE_MIRU_API as string | undefined) ?? "";

export interface AnalyzeStreamHandle {
  /** Cancel the in-flight request. */
  cancel: () => void;
  /** Promise that resolves when the stream ends (clean or otherwise). */
  done: Promise<void>;
}

/**
 * Drive the cinematic streaming path. The caller subscribes via `onEvent`.
 * If the network fails or the server is unreachable, the function transparently
 * replays MOCK_STEPS over a synthetic timeline and emits a final mock trace.
 */
export function analyzeStream(
  req: AnalyzeRequest,
  onEvent: (e: SseEvent, opts: { fromMock: boolean }) => void,
): AnalyzeStreamHandle {
  const ctrl = new AbortController();
  let cancelled = false;

  const done = (async () => {
    try {
      const url = new URL(BASE + "/api/analyze/stream", window.location.origin);
      url.searchParams.set("overlay", "false");
      const res = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`http ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (!cancelled) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        const { frames, rest } = parseFrames(buf);
        buf = rest;
        for (const f of frames) {
          const ev = frameToSseEvent(f);
          if (ev) onEvent(ev, { fromMock: false });
        }
      }
    } catch (e) {
      if (cancelled) return;
      // Fallback — replay mock steps with a humane cadence.
      await replayMock(onEvent, () => cancelled);
    }
  })();

  return {
    cancel: () => {
      cancelled = true;
      ctrl.abort();
    },
    done,
  };
}

async function replayMock(
  onEvent: (e: SseEvent, opts: { fromMock: boolean }) => void,
  isCancelled: () => boolean,
) {
  for (const step of MOCK_STEPS) {
    if (isCancelled()) return;
    onEvent({ kind: "step", step }, { fromMock: true });
    await sleep(420);
  }
  if (isCancelled()) return;
  onEvent({ kind: "trace", trace: MOCK_TRACE }, { fromMock: true });
  onEvent({ kind: "done" }, { fromMock: true });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export async function analyze(req: AnalyzeRequest): Promise<{ trace: ReasoningTrace; fromMock: boolean }> {
  try {
    const res = await fetch(BASE + "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`http ${res.status}`);
    const trace = (await res.json()) as ReasoningTrace;
    return { trace, fromMock: false };
  } catch {
    return { trace: MOCK_TRACE, fromMock: true };
  }
}
