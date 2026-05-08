/**
 * Konjo attention colormap.
 *
 * Maps a normalized intensity v ∈ [0, 1] to an RGBA quadruple:
 *   0.00  →  transparent black
 *   0.25  →  cool cyan          (#5fb3ff)
 *   0.55  →  Konjo violet        (#b794ff)
 *   0.80  →  Konjo hot pink      (#ff4d6d)
 *   1.00  →  warm amber          (#f6c177)
 *
 * The intent is "cool to hot" but on-brand. The first stop is fully
 * transparent so low-attention regions don't muddy the underlying image.
 */

type Stop = readonly [v: number, r: number, g: number, b: number, a: number];

const STOPS: readonly Stop[] = [
  [0.00,  10,  16,  28,   0],   // transparent — match konjo-bg in tone
  [0.25,  95, 179, 255, 110],   // cool / cyan
  [0.55, 183, 148, 255, 175],   // violet
  [0.80, 255,  77, 109, 215],   // hot
  [1.00, 246, 193, 119, 240],   // warm amber
] as const;

/**
 * Sample the colormap at intensity v.
 * Returns four bytes [r, g, b, a].
 */
export function colormap(v: number): [number, number, number, number] {
  if (!Number.isFinite(v)) v = 0;
  if (v <= STOPS[0][0]) return [STOPS[0][1], STOPS[0][2], STOPS[0][3], STOPS[0][4]];
  if (v >= STOPS[STOPS.length - 1][0]) {
    const s = STOPS[STOPS.length - 1];
    return [s[1], s[2], s[3], s[4]];
  }
  for (let i = 1; i < STOPS.length; i++) {
    const a = STOPS[i - 1];
    const b = STOPS[i];
    if (v <= b[0]) {
      const t = (v - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
        Math.round(a[3] + (b[3] - a[3]) * t),
        Math.round(a[4] + (b[4] - a[4]) * t),
      ];
    }
  }
  return [0, 0, 0, 0];
}
