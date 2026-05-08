/**
 * SSE parser for miru's /api/analyze/stream endpoint.
 *
 * The native EventSource API forces GET; miru uses POST. We use fetch + a
 * manual stream parser instead.
 *
 * Frames are separated by \n\n. Each frame may include `event:` and `data:`
 * lines. Comment frames (lines starting with `:`) are keepalives; ignored.
 */
import type { SseEvent, ReasoningStep, ReasoningTrace } from "./types";

export interface ParsedFrame {
  event: string;
  data: string;
}

export function parseFrames(buffer: string): { frames: ParsedFrame[]; rest: string } {
  const frames: ParsedFrame[] = [];
  let rest = buffer;
  while (true) {
    const sep = rest.indexOf("\n\n");
    if (sep === -1) break;
    const raw = rest.slice(0, sep);
    rest = rest.slice(sep + 2);
    const lines = raw.split("\n");
    let event = "message";
    const dataLines: string[] = [];
    let isComment = true;
    for (const line of lines) {
      if (line.startsWith(":")) continue; // comment
      isComment = false;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (isComment) continue;
    frames.push({ event, data: dataLines.join("\n") });
  }
  return { frames, rest };
}

export function frameToSseEvent(f: ParsedFrame): SseEvent | null {
  switch (f.event) {
    case "step": {
      try {
        const step = JSON.parse(f.data) as ReasoningStep;
        return { kind: "step", step };
      } catch {
        return null;
      }
    }
    case "trace": {
      try {
        const trace = JSON.parse(f.data) as ReasoningTrace;
        return { kind: "trace", trace };
      } catch {
        return null;
      }
    }
    case "done":
      return { kind: "done" };
    case "error": {
      try {
        const o = JSON.parse(f.data || "{}") as { error?: string; detail?: string };
        return { kind: "error", error: o.error ?? "unknown", detail: o.detail };
      } catch {
        return { kind: "error", error: "unparseable_error_frame" };
      }
    }
    default:
      return null;
  }
}
