import { useEffect, useRef } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { Sample } from "../lib/types";
import { paintSample } from "../lib/sample";

export interface SampleGalleryProps {
  samples: Sample[];
  active?: string;
  onPick: (s: Sample) => void;
}

export function SampleGallery({ samples, active, onPick }: SampleGalleryProps) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {samples.map((s, i) => (
        <SampleTile
          key={s.id}
          sample={s}
          active={s.id === active}
          onPick={() => onPick(s)}
          index={i}
        />
      ))}
    </div>
  );
}

function SampleTile({
  sample,
  active,
  onPick,
  index,
}: {
  sample: Sample;
  active: boolean;
  onPick: () => void;
  index: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = canvasRef.current;
    if (c) paintSample(sample, c);
  }, [sample]);

  return (
    <motion.button
      type="button"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: ease.kanjo, delay: index * 0.05 }}
      onClick={onPick}
      aria-pressed={active}
      className={[
        "group relative rounded-konjo overflow-hidden border transition-colors text-left",
        active
          ? "border-konjo-accent shadow-konjo-glow"
          : "border-konjo-line hover:border-konjo-fg-muted",
      ].join(" ")}
      style={{ aspectRatio: "1 / 1" }}
    >
      <canvas
        ref={canvasRef}
        width={120}
        height={120}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
      <div className="absolute inset-x-0 bottom-0 px-2 py-1.5 bg-gradient-to-t from-black/85 to-transparent">
        <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">
          {sample.label}
        </div>
      </div>
    </motion.button>
  );
}
