import { describe, it, expect } from "vitest";
import { colormap } from "./colormap";

describe("colormap", () => {
  it("returns transparent for v=0", () => {
    const [, , , a] = colormap(0);
    expect(a).toBe(0);
  });
  it("returns the warm amber endpoint for v=1", () => {
    const [r, g, b, a] = colormap(1);
    expect([r, g, b, a]).toEqual([246, 193, 119, 240]);
  });
  it("interpolates linearly between adjacent stops", () => {
    // midway between stops at v=0.0 ([10,16,28,0]) and v=0.25 ([95,179,255,110])
    const [r, , , a] = colormap(0.125);
    expect(r).toBeGreaterThan(40);
    expect(r).toBeLessThan(60);
    expect(a).toBeGreaterThan(45);
    expect(a).toBeLessThan(65);
  });
  it("clamps below 0", () => {
    expect(colormap(-1)).toEqual(colormap(0));
  });
  it("clamps above 1", () => {
    expect(colormap(2)).toEqual(colormap(1));
  });
  it("handles NaN as 0", () => {
    expect(colormap(NaN)).toEqual(colormap(0));
  });
});
