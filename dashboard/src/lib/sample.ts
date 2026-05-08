/**
 * Synthetic sample images, painted onto a canvas. Used so the dashboard
 * is interesting on first load without bundling binary assets.
 */
import type { Sample } from "./types";

export const SAMPLES: Sample[] = [
  {
    id: "still-life",
    label: "Still life",
    question: "What objects are on the table?",
    paint: (ctx, w, h) => {
      // wood-ish background gradient
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "#3a2a1a");
      g.addColorStop(1, "#1a1108");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // table surface
      const t = ctx.createLinearGradient(0, h * 0.6, 0, h);
      t.addColorStop(0, "#5a3e22");
      t.addColorStop(1, "#2a1d10");
      ctx.fillStyle = t;
      ctx.fillRect(0, h * 0.62, w, h * 0.38);

      // ceramic mug (left) — main subject
      ctx.fillStyle = "#e9e7df";
      ctx.beginPath();
      ctx.ellipse(w * 0.34, h * 0.62, w * 0.07, w * 0.02, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(w * 0.27, h * 0.45, w * 0.14, h * 0.18);
      ctx.beginPath();
      ctx.ellipse(w * 0.34, h * 0.45, w * 0.07, w * 0.025, 0, 0, Math.PI * 2);
      ctx.fillStyle = "#1a1108";
      ctx.fill();

      // notebook (centre)
      ctx.fillStyle = "#7e2c2c";
      ctx.fillRect(w * 0.46, h * 0.58, w * 0.20, h * 0.06);
      ctx.fillStyle = "#3a1414";
      ctx.fillRect(w * 0.46, h * 0.58, w * 0.20, h * 0.012);

      // pen (right)
      ctx.fillStyle = "#0c0c0c";
      ctx.fillRect(w * 0.69, h * 0.61, w * 0.16, h * 0.012);
      ctx.fillStyle = "#b59c5a";
      ctx.fillRect(w * 0.83, h * 0.6075, w * 0.022, h * 0.018);

      // soft window light from the left
      const k = ctx.createRadialGradient(w * 0.1, h * 0.2, 0, w * 0.1, h * 0.2, w * 0.6);
      k.addColorStop(0, "rgba(246, 193, 119, 0.35)");
      k.addColorStop(1, "rgba(246, 193, 119, 0)");
      ctx.fillStyle = k;
      ctx.fillRect(0, 0, w, h);
    },
  },
  {
    id: "landscape",
    label: "Landscape",
    question: "What time of day is shown?",
    paint: (ctx, w, h) => {
      // sky
      const sky = ctx.createLinearGradient(0, 0, 0, h * 0.7);
      sky.addColorStop(0, "#1a2438");
      sky.addColorStop(0.6, "#a06548");
      sky.addColorStop(1, "#f6c177");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, w, h * 0.72);

      // sun
      const sun = ctx.createRadialGradient(w * 0.72, h * 0.55, 0, w * 0.72, h * 0.55, w * 0.18);
      sun.addColorStop(0, "rgba(255, 235, 160, 1)");
      sun.addColorStop(0.6, "rgba(246, 193, 119, 0.55)");
      sun.addColorStop(1, "rgba(246, 193, 119, 0)");
      ctx.fillStyle = sun;
      ctx.fillRect(0, 0, w, h);

      // hills
      ctx.fillStyle = "#13202a";
      ctx.beginPath();
      ctx.moveTo(0, h * 0.85);
      ctx.bezierCurveTo(w * 0.2, h * 0.65, w * 0.4, h * 0.9, w * 0.6, h * 0.7);
      ctx.bezierCurveTo(w * 0.8, h * 0.55, w, h * 0.85, w, h);
      ctx.lineTo(0, h);
      ctx.fill();

      ctx.fillStyle = "#0a131a";
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.bezierCurveTo(w * 0.25, h * 0.85, w * 0.55, h, w * 0.8, h * 0.85);
      ctx.lineTo(w, h);
      ctx.fill();
    },
  },
  {
    id: "diagram",
    label: "Diagram",
    question: "What does this diagram show?",
    paint: (ctx, w, h) => {
      ctx.fillStyle = "#0a0c12";
      ctx.fillRect(0, 0, w, h);

      // three nodes connected by arrows: A → B → C with feedback A ← C
      const nodes = [
        { x: w * 0.2, y: h * 0.5, label: "A" },
        { x: w * 0.5, y: h * 0.5, label: "B" },
        { x: w * 0.8, y: h * 0.5, label: "C" },
      ];

      // edges
      ctx.strokeStyle = "#7ad7ff";
      ctx.lineWidth = w * 0.005;
      const drawArrow = (x1: number, y1: number, x2: number, y2: number) => {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        const a = Math.atan2(y2 - y1, x2 - x1);
        const head = w * 0.02;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - head * Math.cos(a - 0.4), y2 - head * Math.sin(a - 0.4));
        ctx.lineTo(x2 - head * Math.cos(a + 0.4), y2 - head * Math.sin(a + 0.4));
        ctx.closePath();
        ctx.fillStyle = "#7ad7ff";
        ctx.fill();
      };
      drawArrow(nodes[0].x + w * 0.04, nodes[0].y, nodes[1].x - w * 0.04, nodes[1].y);
      drawArrow(nodes[1].x + w * 0.04, nodes[1].y, nodes[2].x - w * 0.04, nodes[2].y);

      // feedback arrow (curved from C back to A above)
      ctx.strokeStyle = "#b794ff";
      ctx.beginPath();
      ctx.moveTo(nodes[2].x, nodes[2].y - w * 0.04);
      ctx.bezierCurveTo(nodes[2].x, h * 0.15, nodes[0].x, h * 0.15, nodes[0].x, nodes[0].y - w * 0.04);
      ctx.stroke();

      // nodes
      for (const n of nodes) {
        ctx.fillStyle = "#181c27";
        ctx.beginPath();
        ctx.arc(n.x, n.y, w * 0.05, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#b794ff";
        ctx.stroke();
        ctx.fillStyle = "#e7ecf4";
        ctx.font = `${w * 0.035}px JetBrains Mono`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(n.label, n.x, n.y);
      }
    },
  },
];

/**
 * Render the named sample into the supplied canvas at full size.
 */
export function paintSample(
  sample: Sample,
  canvas: HTMLCanvasElement,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  sample.paint(ctx, canvas.width, canvas.height);
}
