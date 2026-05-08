/**
 * TypeScript types mirroring miru's API contract (miru/schemas.py).
 * Source of truth lives in Python; this shadow keeps the dashboard typed.
 */

export interface ReasoningStep {
  step: number;
  description: string;
  /** [0, 1] — backend-provided. Decays per step in the default behaviour. */
  confidence: number;
}

export interface AttentionMap {
  width: number;        // typically 16
  height: number;       // typically 16
  data: number[][];     // [height][width], values [0, 1]
}

export interface ReasoningTrace {
  answer: string;
  steps: ReasoningStep[];
  attention_map: AttentionMap;
  backend: string;
  latency_ms: number;
  /** Optional server-rendered overlay PNG when overlay=true. */
  overlay_b64?: string | null;
}

export interface AnalyzeRequest {
  /** Raw RGB bytes (H × W × 3) base64-encoded. NOT a PNG. See miru routes._decode_image. */
  image_b64: string;
  question: string;
  backend?: string;
}

/** SSE event kinds emitted by /api/analyze/stream. */
export type SseEvent =
  | { kind: "step";  step: ReasoningStep }
  | { kind: "trace"; trace: ReasoningTrace }
  | { kind: "done" }
  | { kind: "error"; error: string; detail?: string };

/** Stream lifecycle the UI tracks. */
export type StreamState = "idle" | "streaming" | "done" | "error";

/** A complete sample for the gallery. */
export interface Sample {
  id: string;
  label: string;
  question: string;
  /** Function that paints the sample onto a canvas of the given size. */
  paint: (ctx: CanvasRenderingContext2D, w: number, h: number) => void;
}
