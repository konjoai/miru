/**
 * Mock fixtures for offline development. The dashboard auto-falls-back to
 * these when /api/analyze/stream is unreachable.
 */
import type { ReasoningStep, ReasoningTrace, AttentionMap } from "./types";
import { gaussian2D } from "./attention";

export function mockAttentionMap(focusX = 11.5, focusY = 6.5): AttentionMap {
  const data = gaussian2D(focusX, focusY, 2.6, 16, 16);
  // Add a softer secondary lobe for visual interest.
  const second = gaussian2D(4, 11, 1.8, 16, 16);
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      data[y][x] = Math.max(data[y][x], second[y][x] * 0.55);
    }
  }
  return { width: 16, height: 16, data };
}

export const MOCK_STEPS: ReasoningStep[] = [
  { step: 1, description: "Scanning the canvas for dominant shapes and palette.", confidence: 0.95 },
  { step: 2, description: "Identifying three foreground objects with strong specular highlights.", confidence: 0.90 },
  { step: 3, description: "Resolving the rightmost object as a fountain pen by its tapered profile.", confidence: 0.84 },
  { step: 4, description: "Cross-referencing texture cues with the surface — a wooden table.", confidence: 0.78 },
  { step: 5, description: "Final answer assembled.", confidence: 0.71 },
];

export const MOCK_TRACE: ReasoningTrace = {
  answer: "A wooden table holding a ceramic mug, a leather notebook, and a fountain pen — lit from a window on the left.",
  steps: MOCK_STEPS,
  attention_map: mockAttentionMap(),
  backend: "mock",
  latency_ms: 245.6,
  overlay_b64: null,
};
