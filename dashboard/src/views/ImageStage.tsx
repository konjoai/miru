import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { AttentionField } from "./AttentionField";
import type { Sample } from "../lib/types";
import { paintSample } from "../lib/sample";

export interface ImageStageProps {
  /** Sample to paint into the underlay canvas (or null to keep current). */
  sample: Sample | null;
  /** Loaded image (from drop) — replaces the sample. */
  loadedImage?: HTMLCanvasElement | null;
  /** Current attention grid. */
  grid: number[][];
  /** Stream is in flight — drives breathing + scan band. */
  streaming: boolean;
  /** Optional click-to-focus handler. */
  onFocus?: (nx: number, ny: number) => void;
  focus?: { nx: number; ny: number } | null;
  /** Visible canvas size in px. */
  size?: number;
}

/**
 * The image stage — the user's image and the AttentionField sit on top of
 * one another. Underlay is a hidden canvas painted by the active sample;
 * the AttentionField composites it as the heatmap underlay.
 */
export function ImageStage({
  sample,
  loadedImage,
  grid,
  streaming,
  onFocus,
  focus,
  size = 480,
}: ImageStageProps) {
  const underlayRef = useRef<HTMLCanvasElement | null>(null);
  const [tick, setTick] = useState(0); // forces AttentionField to pick up new underlay

  useEffect(() => {
    const c = underlayRef.current;
    if (!c) return;
    if (loadedImage) {
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(loadedImage, 0, 0, c.width, c.height);
    } else if (sample) {
      paintSample(sample, c);
    }
    setTick((t) => t + 1);
  }, [sample, loadedImage]);

  // Keep underlay canvas resolution in sync with display size for crisp draws.
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.985 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: ease.kanjo }}
      className="relative"
      style={{ width: size, height: size }}
    >
      <canvas
        ref={underlayRef}
        width={size * 2}
        height={size * 2}
        style={{ display: "none" }}
      />
      <AttentionField
        size={size}
        grid={grid}
        imageCanvas={underlayRef.current}
        breathing={streaming}
        scanning={streaming}
        onClick={onFocus}
        focus={focus}
        className="shadow-konjo-glow"
        key={tick}
      />
      {streaming && (
        <motion.div
          aria-hidden
          className="absolute -inset-1 rounded-konjo-lg pointer-events-none"
          animate={{ opacity: [0.18, 0.05, 0.18] }}
          transition={{ duration: 2, ease: ease.seishin, repeat: Infinity }}
          style={{
            border: "1px solid var(--color-konjo-accent)",
            boxShadow: "0 0 24px var(--color-konjo-glow-accent)",
          }}
        />
      )}
    </motion.div>
  );
}
