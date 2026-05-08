import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { KonjoApp, ease } from "@konjoai/ui";
import { ImageStage } from "./views/ImageStage";
import { ReasoningPanel } from "./views/ReasoningPanel";
import { StepTimeline } from "./views/StepTimeline";
import { SampleGallery } from "./views/SampleGallery";
import { ImageDropzone } from "./views/ImageDropzone";
import { QuestionBar } from "./views/QuestionBar";
import { MetaInspector } from "./views/MetaInspector";
import { SAMPLES } from "./lib/sample";
import {
  evolvedAttention,
  makeUniformGrid,
  blendGrids,
  gaussian2D,
} from "./lib/attention";
import { canvasToRawRgbBase64 } from "./lib/imageEncode";
import { analyzeStream } from "./lib/api";
import type { ReasoningStep, ReasoningTrace, Sample } from "./lib/types";

const EXPECTED_STREAM_MS = 3500; // visual budget for the evolved-attention walk

export default function App() {
  // Image source — sample or dropped.
  const [sample, setSample] = useState<Sample>(SAMPLES[0]);
  const [loadedImage, setLoadedImage] = useState<HTMLCanvasElement | null>(null);
  const [question, setQuestion] = useState<string>(SAMPLES[0].question);

  // Stream lifecycle.
  const [streamState, setStreamState] = useState<"idle" | "streaming" | "done" | "error">("idle");
  const [steps, setSteps] = useState<ReasoningStep[]>([]);
  const [trace, setTrace] = useState<ReasoningTrace | null>(null);
  const [fromMock, setFromMock] = useState<boolean>(false);
  const [streamStartedAt, setStreamStartedAt] = useState<number>(0);

  // Display state.
  const [currentGrid, setCurrentGrid] = useState<number[][]>(() => makeUniformGrid(16, 16, 0.04));
  const [focus, setFocus] = useState<{ nx: number; ny: number } | null>(null);
  const [cursor, setCursor] = useState<number>(0);

  const cancelRef = useRef<(() => void) | null>(null);

  // Drive the evolved-attention walk while streaming.
  useEffect(() => {
    if (streamState !== "streaming") return;
    const id = setInterval(() => {
      const elapsed = Date.now() - streamStartedAt;
      const progress = Math.min(1, elapsed / EXPECTED_STREAM_MS);
      const finalGrid = trace?.attention_map.data ?? null;
      setCurrentGrid(evolvedAttention(progress, finalGrid, elapsed));
    }, 100);
    return () => clearInterval(id);
  }, [streamState, streamStartedAt, trace]);

  // After final trace arrives, snap to the canonical attention.
  useEffect(() => {
    if (streamState === "done" && trace) {
      setCurrentGrid(trace.attention_map.data);
    }
  }, [streamState, trace]);

  // When the user scrubs after completion, show a proportional snapshot.
  useEffect(() => {
    if (streamState !== "done" || !trace) return;
    const total = steps.length;
    if (total === 0) return;
    const t = total > 0 ? cursor / total : 1;
    if (t >= 1) {
      setCurrentGrid(trace.attention_map.data);
      return;
    }
    // Pre-final scrub: blend a soft blob toward the final grid.
    const final = trace.attention_map.data;
    const soft = gaussian2D(8, 8, 4 + (1 - t) * 4, 16, 16);
    setCurrentGrid(blendGrids(soft, final, t));
  }, [cursor, streamState, trace, steps.length]);

  const startScan = () => {
    cancelRef.current?.();
    setSteps([]);
    setTrace(null);
    setFocus(null);
    setCursor(0);
    setStreamState("streaming");
    setStreamStartedAt(Date.now());

    const sourceCanvas = loadedImage ?? renderSampleToCanvas(sample, 64);
    let imageB64 = "";
    try {
      imageB64 = canvasToRawRgbBase64(sourceCanvas, 16);
    } catch {
      // jsdom or unavailable — let the API client fall back to mocks
    }

    const handle = analyzeStream(
      { image_b64: imageB64, question, backend: "mock" },
      (e, opts) => {
        if (opts.fromMock) setFromMock(true);
        if (e.kind === "step") {
          setSteps((s) => [...s, e.step]);
          setCursor((c) => c + 1);
        } else if (e.kind === "trace") {
          setTrace(e.trace);
        } else if (e.kind === "done") {
          setStreamState("done");
        } else if (e.kind === "error") {
          setStreamState("error");
        }
      },
    );
    cancelRef.current = handle.cancel;
  };

  // Auto-run on mount.
  useEffect(() => {
    startScan();
    return () => cancelRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPickSample = (s: Sample) => {
    setSample(s);
    setLoadedImage(null);
    setQuestion(s.question);
  };

  const onDropImage = (canvas: HTMLCanvasElement) => {
    // Snapshot the dropped canvas so future ImageDropzone repaints don't mutate ours.
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    copy.getContext("2d")?.drawImage(canvas, 0, 0);
    setLoadedImage(copy);
  };

  const live = streamState === "streaming";
  const finalized = streamState === "done";
  const attentionSource: "interpolated" | "final" | "idle" =
    streamState === "idle" ? "idle" :
    finalized ? "final" : "interpolated";

  return (
    <KonjoApp
      product="miru"
      tagline="The mind of the machine"
      status={
        live      ? { label: "streaming", severity: "info" } :
        finalized ? { label: fromMock ? "offline · mocks" : "live", severity: fromMock ? "warn" : "ok" } :
                    { label: "idle", severity: "info" }
      }
    >
      <Hero />

      <div className="space-y-6 mt-10">
        {/* Cinema — image stage + reasoning */}
        <section className="grid lg:grid-cols-[480px_1fr] gap-6 items-start">
          <ImageStage
            sample={sample}
            loadedImage={loadedImage}
            grid={currentGrid}
            streaming={live}
            onFocus={(nx, ny) => setFocus({ nx, ny })}
            focus={focus}
            size={480}
          />

          <ReasoningPanel
            steps={steps.slice(0, finalized ? cursor || steps.length : steps.length)}
            answer={finalized && cursor >= steps.length ? trace?.answer : undefined}
            streaming={live}
            highlight={finalized ? Math.min(cursor - 1, steps.length - 1) : undefined}
            className="h-full min-h-[480px]"
          />
        </section>

        {/* Timeline scrubber */}
        <StepTimeline
          steps={steps}
          cursor={cursor}
          onCursorChange={setCursor}
          live={live}
          finalized={finalized}
        />

        {/* Question + Sample picker + Dropzone */}
        <section className="grid md:grid-cols-[1fr_auto] gap-4 items-start">
          <QuestionBar
            value={question}
            onChange={setQuestion}
            onSubmit={startScan}
            disabled={live || question.trim().length === 0}
            submitLabel={live ? "thinking…" : "ask"}
          />
          <div className="flex items-stretch gap-3">
            <SampleGallery
              samples={SAMPLES}
              active={loadedImage ? undefined : sample.id}
              onPick={onPickSample}
            />
            <ImageDropzone onLoaded={onDropImage} size={120} />
          </div>
        </section>

        {/* Meta inspector */}
        <MetaInspector
          backend={trace?.backend ?? "mock"}
          latencyMs={trace?.latency_ms}
          attentionSource={attentionSource}
          grid={currentGrid}
          fromMock={fromMock}
        />

        {/* Click hint */}
        <AnimatePresence>
          {finalized && !focus && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: ease.kanjo, delay: 0.4 }}
              className="text-konjo-mono text-konjo-fg-muted text-[11px] text-center"
            >
              click anywhere on the image to drop a focus marker
            </motion.div>
          )}
        </AnimatePresence>

        <Footer />
      </div>
    </KonjoApp>
  );
}

/** Render a Sample to a fresh canvas at the given size. */
function renderSampleToCanvas(s: Sample, size: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (ctx) s.paint(ctx, size, size);
  return c;
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet" style={{ fontSize: 11 }}>
        miru · 見る · to see
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{ fontSize: 52, fontWeight: 600, letterSpacing: "-0.025em", maxWidth: 920, lineHeight: 1.05 }}
      >
        The <span style={{ color: "var(--color-konjo-accent)" }}>mind</span> of the machine,{" "}
        <span style={{ color: "var(--color-konjo-violet)" }}>visible</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        Watch a vision-language model decide. The reasoning streams in. The attention field morphs as it focuses. Click anywhere on the image to ask <em>why did it look here?</em>
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/api/analyze/stream</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · vectro · squish · kyro · kohaku · kairu · toki · squash
        </span>
      </div>
    </footer>
  );
}
