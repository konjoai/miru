import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import type { ReasoningStep } from "../lib/types";

export interface ReasoningPanelProps {
  steps: ReasoningStep[];
  /** Final answer (revealed at the end of the stream). */
  answer?: string;
  /** Stream is in flight — drives the trailing typing caret. */
  streaming: boolean;
  /** Currently focused step index (for scrubber sync). */
  highlight?: number;
  className?: string;
}

/**
 * The model's voice. Steps stream in with a typing-style fade + slide,
 * each carrying its own confidence bar. When inference completes, the
 * final answer rises in below.
 */
export function ReasoningPanel({
  steps,
  answer,
  streaming,
  highlight,
  className,
}: ReasoningPanelProps) {
  return (
    <div
      className={[
        "glass-konjo rounded-konjo-lg p-5 grain-konjo",
        "flex flex-col gap-3",
        className,
      ].join(" ")}
    >
      <div className="flex items-center justify-between">
        <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted text-[10px]">
          reasoning trace
        </div>
        <div className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-faint">
          {steps.length} step{steps.length === 1 ? "" : "s"}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1" style={{ minHeight: 240 }}>
        {steps.length === 0 && !streaming && (
          <div className="text-konjo-fg-muted text-[13px] py-8 text-center">
            Drop an image or pick a sample to begin.
          </div>
        )}

        <AnimatePresence initial={false}>
          {steps.map((s, i) => (
            <motion.div
              key={s.step}
              initial={{ opacity: 0, y: 8, filter: "blur(4px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{ duration: 0.35, ease: ease.kanjo }}
              className={[
                "rounded-konjo p-3 border transition-colors",
                highlight === i
                  ? "border-konjo-accent bg-konjo-surface-2/70"
                  : "border-konjo-line/60 bg-konjo-surface/60",
              ].join(" ")}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span
                  className="text-konjo-mono text-[10px] tabular-nums text-konjo-fg-muted"
                  style={{ minWidth: 22 }}
                >
                  {String(s.step).padStart(2, "0")}
                </span>
                <span className="flex-1 text-konjo-fg" style={{ fontSize: 13, lineHeight: 1.5 }}>
                  {s.description}
                </span>
              </div>
              <ConfidenceBar value={s.confidence} />
            </motion.div>
          ))}
        </AnimatePresence>

        {streaming && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-konjo-fg-muted text-[12px]"
          >
            <motion.span
              aria-hidden
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, ease: ease.seishin, repeat: Infinity }}
              className="inline-block rounded-full"
              style={{
                width: 6, height: 6,
                background: "var(--color-konjo-accent)",
                boxShadow: "0 0 8px var(--color-konjo-accent)",
              }}
            />
            <span className="text-konjo-mono">thinking…</span>
          </motion.div>
        )}
      </div>

      <AnimatePresence>
        {answer && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: ease.kanjo }}
            className="border-t border-konjo-line/60 pt-3 mt-2"
          >
            <div className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-violet text-[10px] mb-1">
              answer
            </div>
            <div
              className="text-konjo-fg"
              style={{ fontSize: 15, lineHeight: 1.5, fontWeight: 500 }}
            >
              {answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  const c =
    pct >= 0.85 ? "var(--color-konjo-good)" :
    pct >= 0.65 ? "var(--color-konjo-accent)" :
    pct >= 0.45 ? "var(--color-konjo-warm)" : "var(--color-konjo-hot)";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 rounded-full bg-konjo-line/60 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct * 100}%` }}
          transition={{ duration: 0.6, ease: ease.kanjo }}
          className="h-full"
          style={{ background: c, boxShadow: `0 0 6px ${c}` }}
        />
      </div>
      <span
        className="text-konjo-mono text-[10px] tabular-nums"
        style={{ color: c, minWidth: 36 }}
      >
        {(pct * 100).toFixed(0)}%
      </span>
    </div>
  );
}
