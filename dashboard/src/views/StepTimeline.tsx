import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import type { ReasoningStep } from "../lib/types";

export interface StepTimelineProps {
  steps: ReasoningStep[];
  /** Currently selected step index (or steps.length for "after final"). */
  cursor: number;
  onCursorChange: (i: number) => void;
  /** True when the live SSE stream is in flight — disables scrubbing. */
  live?: boolean;
  /** Trace is complete — enables play/scrub. */
  finalized?: boolean;
}

const SPEEDS: { id: number; label: string }[] = [
  { id: 0.5, label: "0.5×" },
  { id: 1.0, label: "1×" },
  { id: 2.0, label: "2×" },
];

/**
 * Horizontal scrubber: dots per step + a draggable cursor. When the trace
 * is finalized, you can play it back at 0.5×/1×/2×.
 */
export function StepTimeline({
  steps,
  cursor,
  onCursorChange,
  live = false,
  finalized = false,
}: StepTimelineProps) {
  const total = steps.length;
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const playRef = useRef({ playing: false, speed: 1.0 });
  playRef.current = { playing, speed };

  useEffect(() => {
    if (!playing || !finalized || total === 0) return;
    const stepMs = 600 / speed;
    const id = setInterval(() => {
      onCursorChange(Math.min(cursor + 1, total));
      if (cursor + 1 >= total) {
        setPlaying(false);
        clearInterval(id);
      }
    }, stepMs);
    return () => clearInterval(id);
  }, [playing, speed, cursor, total, finalized, onCursorChange]);

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!finalized || total === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = (e.clientX - rect.left) / rect.width;
    onCursorChange(Math.round(Math.max(0, Math.min(1, t)) * total));
  };

  const togglePlay = () => {
    if (!finalized) return;
    if (cursor >= total) onCursorChange(0);
    setPlaying((p) => !p);
  };

  return (
    <div className="glass-konjo rounded-konjo p-3 flex items-center gap-3">
      {/* Play / Pause */}
      <button
        type="button"
        onClick={togglePlay}
        disabled={!finalized || total === 0}
        aria-label={playing ? "pause" : "play"}
        className={[
          "w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-colors",
          finalized && total > 0
            ? "bg-konjo-accent text-konjo-bg hover:brightness-110"
            : "bg-konjo-surface text-konjo-fg-faint cursor-not-allowed",
        ].join(" ")}
      >
        {playing ? <PauseGlyph /> : <PlayGlyph />}
      </button>

      {/* Track */}
      <div className="flex-1 relative h-9 cursor-pointer" onClick={handleSeek}>
        <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-konjo-line rounded-full -translate-y-1/2" />
        {/* Filled portion */}
        <motion.div
          className="absolute left-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full"
          style={{
            background: "var(--color-konjo-accent)",
            boxShadow: "0 0 8px var(--color-konjo-glow-accent)",
          }}
          animate={{ width: total > 0 ? `${(cursor / total) * 100}%` : "0%" }}
          transition={{ duration: 0.25, ease: ease.kanjo }}
        />
        {/* Dots per step */}
        {steps.map((_, i) => {
          const x = total > 0 ? (i / total) * 100 : 0;
          const passed = i < cursor;
          const active = i === Math.min(cursor, total - 1);
          return (
            <span
              key={i}
              aria-hidden
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full"
              style={{
                left: `${x}%`,
                width: active ? 10 : 7,
                height: active ? 10 : 7,
                background: passed || active ? "var(--color-konjo-accent)" : "var(--color-konjo-fg-faint)",
                boxShadow: active ? "0 0 8px var(--color-konjo-accent)" : undefined,
              }}
            />
          );
        })}
        {/* End marker */}
        {total > 0 && (
          <span
            aria-hidden
            className="absolute top-1/2 right-0 -translate-y-1/2 translate-x-1/2 rounded-full"
            style={{
              width: 7,
              height: 7,
              background: cursor >= total ? "var(--color-konjo-good)" : "var(--color-konjo-fg-faint)",
              boxShadow: cursor >= total ? "0 0 10px var(--color-konjo-good)" : undefined,
            }}
          />
        )}
      </div>

      {/* Speeds */}
      <div className="hidden sm:inline-flex items-center gap-1 p-1 rounded-konjo bg-konjo-surface border border-konjo-line">
        {SPEEDS.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setSpeed(s.id)}
            disabled={!finalized}
            aria-pressed={s.id === speed}
            className={[
              "px-2 py-1 rounded-konjo-sm text-konjo-mono text-[10px] uppercase tracking-[0.16em] transition-colors",
              s.id === speed
                ? "bg-konjo-accent text-konjo-bg"
                : "text-konjo-fg-muted hover:text-konjo-fg",
              !finalized && "opacity-40",
            ].join(" ")}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Step counter */}
      <div className="text-konjo-mono tabular-nums text-konjo-fg-muted text-[11px] min-w-[44px] text-right">
        {Math.min(cursor, total)} / {total}
        {live && (
          <div className="text-konjo-accent text-[9px] uppercase tracking-[0.18em]">live</div>
        )}
      </div>
    </div>
  );
}

function PlayGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
      <polygon points="3,2 12,7 3,12" fill="currentColor" />
    </svg>
  );
}

function PauseGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
      <rect x="3" y="2" width="3" height="10" fill="currentColor" />
      <rect x="8" y="2" width="3" height="10" fill="currentColor" />
    </svg>
  );
}
