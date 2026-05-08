/**
 * Attention grid math — sampling, lerp, and the temporal evolution model.
 *
 * miru emits a single 16×16 attention grid at the end of inference. To create
 * the illusion of attention *evolving* during streaming, we synthesize a
 * walk: scan pattern → drifting blob → focused on the final hotspot.
 *
 * This is an animation over real data, not a fabrication of per-step
 * attention. We're transparent about it: the meta inspector reports
 * `attention_source: "interpolated"` until the final trace arrives.
 */
import type { AttentionMap } from "./types";

export function makeUniformGrid(w = 16, h = 16, value = 0.05): number[][] {
  return Array.from({ length: h }, () => Array.from({ length: w }, () => value));
}

/**
 * Bilinearly sample a grid at fractional indices fx, fy (in [0, w-1] × [0, h-1]).
 * Returns 0 outside the grid.
 */
export function bilinear(grid: number[][], fx: number, fy: number): number {
  const h = grid.length;
  if (h === 0) return 0;
  const w = grid[0].length;
  if (fx < 0 || fy < 0 || fx > w - 1 || fy > h - 1) return 0;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const dx = fx - x0;
  const dy = fy - y0;
  const v00 = grid[y0][x0];
  const v01 = grid[y0][x1];
  const v10 = grid[y1][x0];
  const v11 = grid[y1][x1];
  return (
    v00 * (1 - dx) * (1 - dy) +
    v01 *      dx  * (1 - dy) +
    v10 * (1 - dx) *      dy  +
    v11 *      dx  *      dy
  );
}

/**
 * 2D Gaussian falloff. cx, cy in grid coords; sigma in grid cells.
 */
export function gaussian2D(
  cx: number, cy: number, sigma: number, w = 16, h = 16,
): number[][] {
  const grid = makeUniformGrid(w, h, 0);
  const s2 = 2 * sigma * sigma;
  let max = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = Math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / s2));
      grid[y][x] = v;
      if (v > max) max = v;
    }
  }
  // normalize to [0, 1]
  if (max > 0) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        grid[y][x] = grid[y][x] / max;
      }
    }
  }
  return grid;
}

export function blendGrids(a: number[][], b: number[][], t: number): number[][] {
  if (a.length === 0 || b.length === 0) return a;
  const h = Math.min(a.length, b.length);
  const w = Math.min(a[0].length, b[0].length);
  const out = makeUniformGrid(w, h, 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y][x] = a[y][x] * (1 - t) + b[y][x] * t;
    }
  }
  return out;
}

/**
 * The walk: given streaming progress [0, 1] and a final attention grid,
 * synthesize an in-flight attention grid.
 *
 *   progress = 0     → soft horizontal scan band
 *   progress = 0.4   → drifting blob near the (eventual) hotspot
 *   progress = 1     → exact final grid
 *
 * If finalGrid is null we use a moving Gaussian for the demo.
 */
export function evolvedAttention(
  progress: number,
  finalGrid: number[][] | null,
  /** Time in ms since stream start; controls scan-band motion. */
  tMs: number,
  w = 16,
  h = 16,
): number[][] {
  const p = Math.max(0, Math.min(1, progress));

  // Scan band — a horizontal Gaussian stripe oscillating.
  const scanY = (h - 1) * 0.5 + Math.sin(tMs / 600) * (h * 0.32);
  const scan = gaussian2D(w / 2, scanY, 2.4 + Math.sin(tMs / 1200) * 0.4, w, h);

  // Hotspot from final grid (peak location), or default center.
  let cx = w / 2;
  let cy = h / 2;
  if (finalGrid) {
    let max = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (finalGrid[y][x] > max) {
          max = finalGrid[y][x];
          cx = x;
          cy = y;
        }
      }
    }
  }

  const focused = gaussian2D(cx, cy, 1.4 + (1 - p) * 1.6, w, h);
  const target = finalGrid ?? focused;

  // Three-way blend: scan → focused → exact target as p moves 0 → 1.
  if (p < 0.5) {
    const t = p / 0.5;
    return blendGrids(scan, focused, t);
  }
  const t = (p - 0.5) / 0.5;
  return blendGrids(focused, target, t);
}

/** Find the hotspot (argmax) of a grid. Returns [x, y, v]. */
export function hotspot(grid: number[][]): { x: number; y: number; v: number } {
  let mx = 0, my = 0, mv = -Infinity;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (grid[y][x] > mv) {
        mv = grid[y][x];
        mx = x;
        my = y;
      }
    }
  }
  return { x: mx, y: my, v: mv };
}

/** Coverage = fraction of cells with attention > threshold. */
export function coverage(grid: number[][], threshold = 0.4): number {
  if (grid.length === 0) return 0;
  const w = grid[0].length;
  const h = grid.length;
  let c = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (grid[y][x] > threshold) c++;
  return c / (w * h);
}

/** Convenience — coerce an AttentionMap (or absence) into a grid. */
export function gridOf(map: AttentionMap | null | undefined): number[][] | null {
  if (!map) return null;
  return map.data;
}
