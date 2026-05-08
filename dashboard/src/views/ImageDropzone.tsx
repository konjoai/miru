import { useRef, useState } from "react";
import { ease } from "@konjoai/ui";
import { motion } from "motion/react";
import { loadFileIntoCanvas } from "../lib/imageEncode";

export interface ImageDropzoneProps {
  /** Called once a dropped image has been painted to a canvas (passed back). */
  onLoaded: (canvas: HTMLCanvasElement) => void;
  /** Visual size in px. */
  size?: number;
}

export function ImageDropzone({ onLoaded, size = 120 }: ImageDropzoneProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("not an image");
      return;
    }
    const c = canvasRef.current;
    if (!c) return;
    try {
      await loadFileIntoCanvas(file, c);
      setError(null);
      onLoaded(c);
    } catch {
      setError("decode failed");
    }
  };

  return (
    <motion.label
      htmlFor="image-drop-input"
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files[0];
        if (f) void handleFile(f);
      }}
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, ease: ease.kanjo }}
      className={[
        "relative cursor-pointer rounded-konjo overflow-hidden border-dashed border",
        "flex items-center justify-center transition-colors",
        drag
          ? "border-konjo-accent bg-konjo-surface-2/70"
          : "border-konjo-line bg-konjo-surface/40 hover:border-konjo-fg-muted",
      ].join(" ")}
      style={{ aspectRatio: "1 / 1", width: size, height: size }}
    >
      <input
        ref={inputRef}
        id="image-drop-input"
        type="file"
        accept="image/*"
        className="absolute inset-0 opacity-0 cursor-pointer"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleFile(f);
        }}
      />
      <canvas
        ref={canvasRef}
        width={size * 2}
        height={size * 2}
        style={{ width: "100%", height: "100%", display: "block", position: "absolute", inset: 0 }}
      />
      <div className="relative z-10 text-center px-2 pointer-events-none">
        <div
          className="text-konjo-mono uppercase tracking-[0.18em] text-konjo-fg-muted"
          style={{ fontSize: 9 }}
        >
          {error ?? "drop or click"}
        </div>
      </div>
    </motion.label>
  );
}
