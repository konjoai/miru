import { describe, it, expect } from "vitest";
import { mockAttentionMap, MOCK_TRACE, MOCK_STEPS } from "./mock";
import { hotspot } from "./attention";

describe("mockAttentionMap", () => {
  it("has the expected shape", () => {
    const m = mockAttentionMap();
    expect(m.width).toBe(16);
    expect(m.height).toBe(16);
    expect(m.data.length).toBe(16);
    expect(m.data[0].length).toBe(16);
  });
  it("respects the focus argument", () => {
    const m = mockAttentionMap(2, 3);
    const peak = hotspot(m.data);
    expect(peak.x).toBeGreaterThanOrEqual(1);
    expect(peak.x).toBeLessThanOrEqual(3);
    expect(peak.y).toBeGreaterThanOrEqual(2);
    expect(peak.y).toBeLessThanOrEqual(4);
  });
});

describe("MOCK_TRACE", () => {
  it("has a non-empty answer and at least 3 steps", () => {
    expect(MOCK_TRACE.answer.length).toBeGreaterThan(10);
    expect(MOCK_TRACE.steps.length).toBeGreaterThanOrEqual(3);
  });
  it("step numbers are 1..N consecutively", () => {
    MOCK_STEPS.forEach((s, i) => expect(s.step).toBe(i + 1));
  });
  it("confidences fall in [0, 1]", () => {
    for (const s of MOCK_STEPS) {
      expect(s.confidence).toBeGreaterThanOrEqual(0);
      expect(s.confidence).toBeLessThanOrEqual(1);
    }
  });
});
