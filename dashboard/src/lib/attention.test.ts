import { describe, it, expect } from "vitest";
import {
  bilinear,
  blendGrids,
  coverage,
  evolvedAttention,
  gaussian2D,
  hotspot,
  makeUniformGrid,
} from "./attention";

describe("makeUniformGrid", () => {
  it("creates a w × h grid filled with value", () => {
    const g = makeUniformGrid(4, 3, 0.5);
    expect(g.length).toBe(3);
    expect(g[0].length).toBe(4);
    expect(g[2][3]).toBe(0.5);
  });
});

describe("bilinear", () => {
  const g = [
    [0, 1],
    [1, 0],
  ];
  it("returns 0 for out-of-range", () => {
    expect(bilinear(g, -1, -1)).toBe(0);
    expect(bilinear(g, 5, 5)).toBe(0);
  });
  it("returns the corner value at integer coords", () => {
    expect(bilinear(g, 0, 0)).toBe(0);
    expect(bilinear(g, 1, 0)).toBe(1);
    expect(bilinear(g, 0, 1)).toBe(1);
    expect(bilinear(g, 1, 1)).toBe(0);
  });
  it("interpolates the centre as the mean of all four corners", () => {
    expect(bilinear(g, 0.5, 0.5)).toBeCloseTo(0.5);
  });
});

describe("gaussian2D", () => {
  it("peaks at the centre", () => {
    const g = gaussian2D(7.5, 7.5, 2);
    const { x, y, v } = hotspot(g);
    expect(v).toBeCloseTo(1, 5);
    expect(x === 7 || x === 8).toBe(true);
    expect(y === 7 || y === 8).toBe(true);
  });
  it("normalizes to [0, 1]", () => {
    const g = gaussian2D(5, 5, 2);
    let max = 0;
    for (const row of g) for (const v of row) max = Math.max(max, v);
    expect(max).toBeCloseTo(1, 5);
  });
});

describe("blendGrids", () => {
  it("returns A when t=0 and B when t=1", () => {
    const a = makeUniformGrid(4, 4, 0);
    const b = makeUniformGrid(4, 4, 1);
    expect(blendGrids(a, b, 0)).toEqual(a);
    expect(blendGrids(a, b, 1)).toEqual(b);
  });
  it("interpolates element-wise at t=0.5", () => {
    const a = makeUniformGrid(4, 4, 0);
    const b = makeUniformGrid(4, 4, 1);
    const m = blendGrids(a, b, 0.5);
    expect(m[2][2]).toBe(0.5);
  });
});

describe("coverage", () => {
  it("counts cells above threshold", () => {
    const g = [[0, 1], [0.5, 0.6]];
    expect(coverage(g, 0.4)).toBe(0.75); // 3 of 4
  });
});

describe("evolvedAttention", () => {
  it("returns the final grid when progress=1", () => {
    const final = makeUniformGrid(16, 16, 0.7);
    final[5][5] = 1;
    const out = evolvedAttention(1, final, 0);
    // Approximate equality — at progress=1 the blend collapses to target.
    expect(out[5][5]).toBeCloseTo(1);
    expect(out[0][0]).toBeCloseTo(0.7);
  });
  it("clamps progress to [0, 1]", () => {
    const a = evolvedAttention(2, null, 0);
    const b = evolvedAttention(1, null, 0);
    expect(a).toEqual(b);
  });
  it("returns a 16x16 grid by default", () => {
    const out = evolvedAttention(0.5, null, 0);
    expect(out.length).toBe(16);
    expect(out[0].length).toBe(16);
  });
});
