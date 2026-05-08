import { describe, it, expect } from "vitest";
import { parseFrames, frameToSseEvent } from "./sse";

describe("parseFrames", () => {
  it("splits two complete frames", () => {
    const buf = "event: step\ndata: {\"step\":1}\n\nevent: done\ndata: {}\n\n";
    const { frames, rest } = parseFrames(buf);
    expect(frames.length).toBe(2);
    expect(frames[0].event).toBe("step");
    expect(frames[0].data).toBe('{"step":1}');
    expect(frames[1].event).toBe("done");
    expect(rest).toBe("");
  });

  it("returns rest when buffer ends mid-frame", () => {
    const buf = "event: step\ndata: {\"step\":1}\n\nevent: trace\ndata: {\"answer\":\"x";
    const { frames, rest } = parseFrames(buf);
    expect(frames.length).toBe(1);
    expect(rest).toBe('event: trace\ndata: {"answer":"x');
  });

  it("ignores comment frames (keepalives)", () => {
    const buf = ": keepalive\n\nevent: step\ndata: {}\n\n";
    const { frames } = parseFrames(buf);
    expect(frames.length).toBe(1);
    expect(frames[0].event).toBe("step");
  });

  it("returns empty when no full frame yet", () => {
    const { frames, rest } = parseFrames("event: step\ndata: {");
    expect(frames.length).toBe(0);
    expect(rest).toBe("event: step\ndata: {");
  });
});

describe("frameToSseEvent", () => {
  it("decodes a step frame", () => {
    const ev = frameToSseEvent({ event: "step", data: '{"step":1,"description":"x","confidence":0.9}' });
    expect(ev?.kind).toBe("step");
    if (ev?.kind === "step") expect(ev.step.step).toBe(1);
  });

  it("decodes a trace frame", () => {
    const trace = {
      answer: "x",
      steps: [],
      attention_map: { width: 16, height: 16, data: [] },
      backend: "mock",
      latency_ms: 1.0,
    };
    const ev = frameToSseEvent({ event: "trace", data: JSON.stringify(trace) });
    expect(ev?.kind).toBe("trace");
  });

  it("returns done sentinel", () => {
    const ev = frameToSseEvent({ event: "done", data: "{}" });
    expect(ev?.kind).toBe("done");
  });

  it("decodes an error frame", () => {
    const ev = frameToSseEvent({ event: "error", data: '{"error":"timeout","detail":"oh no"}' });
    expect(ev?.kind).toBe("error");
    if (ev?.kind === "error") {
      expect(ev.error).toBe("timeout");
      expect(ev.detail).toBe("oh no");
    }
  });

  it("ignores unknown event names", () => {
    expect(frameToSseEvent({ event: "garbage", data: "{}" })).toBeNull();
  });

  it("returns null on malformed JSON", () => {
    expect(frameToSseEvent({ event: "step", data: "not json" })).toBeNull();
  });
});
