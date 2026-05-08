import { useState } from "react";
import { ease } from "@konjoai/ui";
import { motion } from "motion/react";

export interface QuestionBarProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  submitLabel?: string;
}

export function QuestionBar({
  value, onChange, onSubmit, disabled, submitLabel = "ask",
}: QuestionBarProps) {
  const [focused, setFocused] = useState(false);
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: ease.kanjo }}
      className={[
        "flex items-center gap-2 glass-konjo rounded-konjo p-2 pl-4",
        focused ? "shadow-konjo-glow" : "",
      ].join(" ")}
    >
      <input
        type="text"
        placeholder="ask a question about the image…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onKeyDown={(e) => { if (e.key === "Enter" && !disabled) onSubmit(); }}
        className="flex-1 bg-transparent border-0 outline-none text-konjo-fg placeholder:text-konjo-fg-faint"
        style={{ fontSize: 15 }}
      />
      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled}
        className={[
          "px-4 py-2 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[11px] transition-colors",
          disabled
            ? "bg-konjo-surface text-konjo-fg-faint cursor-not-allowed"
            : "bg-konjo-accent text-konjo-bg hover:brightness-110 cursor-pointer",
        ].join(" ")}
      >
        {submitLabel}
      </button>
    </motion.div>
  );
}
