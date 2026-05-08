import { useEffect, useRef } from "react";
import { bilinear } from "../lib/attention";
import { colormap } from "../lib/colormap";

export interface AttentionFieldProps {
  /** Pixel size of the canvas. Square. */
  size: number;
  /** Current attention grid — typically 16×16. */
  grid: number[][];
  /** Pulse the overlay subtly while streaming. Default false. */
  breathing?: boolean;
  /** Image to composite under the heatmap. */
  imageCanvas?: HTMLCanvasElement | null;
  /** When true, draw a soft animated scan band sweeping vertically. */
  scanning?: boolean;
  /** Click to drop a focus marker. Receives normalized [0..1] coords. */
  onClick?: (nx: number, ny: number) => void;
  /** Optional persistent focus marker (normalized coords). */
  focus?: { nx: number; ny: number } | null;
  className?: string;
}

/**
 * AttentionField — the centerpiece. A canvas that composites:
 *   1. The user's image (or a sample),
 *   2. A bilinearly-upsampled attention heatmap,
 *   3. Optionally a scanning sweep,
 *   4. Optionally a persistent focus marker.
 *
 * Renders to an offscreen pipeline at native canvas resolution and uses
 * `requestAnimationFrame` for the breathing pulse. The grid is read on
 * every frame, so the parent can update `grid` as fast as it likes — the
 * heatmap will follow.
 */
export function AttentionField({
  size,
  grid,
  breathing = false,
  imageCanvas,
  scanning = false,
  onClick,
  focus,
  className,
}: AttentionFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Hold the latest grid in a ref so the rAF loop always sees it.
  const gridRef = useRef<number[][]>(grid);
  gridRef.current = grid;
  const breathingRef = useRef(breathing);
  breathingRef.current = breathing;
  const imageRef = useRef<HTMLCanvasElement | null | undefined>(imageCanvas);
  imageRef.current = imageCanvas;
  const scanningRef = useRef(scanning);
  scanningRef.current = scanning;
  const focusRef = useRef(focus);
  focusRef.current = focus;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let stopped = false;

    const start = performance.now();

    const render = () => {
      if (stopped) return;
      const t = performance.now() - start;
      const W = canvas.width;
      const H = canvas.height;

      // 1. underlay image
      ctx.fillStyle = "#0a0c12";
      ctx.fillRect(0, 0, W, H);
      const img = imageRef.current;
      if (img) {
        ctx.drawImage(img, 0, 0, W, H);
      }

      // 2. heatmap overlay — sample the grid bilinearly per pixel
      const g = gridRef.current;
      if (g && g.length > 0) {
        const gh = g.length;
        const gw = g[0].length;
        const breathe = breathingRef.current ? 0.78 + Math.sin(t / 700) * 0.22 : 1;
        const out = ctx.createImageData(W, H);
        const dst = out.data;
        for (let y = 0; y < H; y++) {
          const fy = (y / (H - 1)) * (gh - 1);
          for (let x = 0; x < W; x++) {
            const fx = (x / (W - 1)) * (gw - 1);
            const v = bilinear(g, fx, fy) * breathe;
            const [r, gC, b, a] = colormap(v);
            const idx = (y * W + x) * 4;
            dst[idx]     = r;
            dst[idx + 1] = gC;
            dst[idx + 2] = b;
            dst[idx + 3] = a;
          }
        }
        // Use a temporary canvas so we can composite with "screen" mode.
        const tmp = document.createElement("canvas");
        tmp.width = W;
        tmp.height = H;
        tmp.getContext("2d")!.putImageData(out, 0, 0);
        ctx.globalCompositeOperation = "screen";
        ctx.drawImage(tmp, 0, 0);
        ctx.globalCompositeOperation = "source-over";
      }

      // 3. scan band
      if (scanningRef.current) {
        const cy = (Math.sin(t / 900) * 0.5 + 0.5) * H;
        const grad = ctx.createLinearGradient(0, cy - H * 0.08, 0, cy + H * 0.08);
        grad.addColorStop(0,   "rgba(122, 215, 255, 0)");
        grad.addColorStop(0.5, "rgba(122, 215, 255, 0.25)");
        grad.addColorStop(1,   "rgba(122, 215, 255, 0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, cy - H * 0.08, W, H * 0.16);
      }

      // 4. focus marker — a softly pulsing target reticle
      const fok = focusRef.current;
      if (fok) {
        const fx = fok.nx * W;
        const fy = fok.ny * H;
        const r = 14 + Math.sin(t / 400) * 3;
        ctx.strokeStyle = "rgba(122, 215, 255, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(fx, fy, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(fx - r - 5, fy);
        ctx.lineTo(fx - r + 2, fy);
        ctx.moveTo(fx + r - 2, fy);
        ctx.lineTo(fx + r + 5, fy);
        ctx.moveTo(fx, fy - r - 5);
        ctx.lineTo(fx, fy - r + 2);
        ctx.moveTo(fx, fy + r - 2);
        ctx.lineTo(fx, fy + r + 5);
        ctx.stroke();
      }

      raf = requestAnimationFrame(render);
    };

    raf = requestAnimationFrame(render);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, []);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onClick) return;
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    onClick(nx, ny);
  };

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      onClick={handleClick}
      className={className}
      style={{
        width: size,
        height: size,
        display: "block",
        cursor: onClick ? "crosshair" : "default",
        borderRadius: 12,
      }}
    />
  );
}
