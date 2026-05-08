import { hotspot, coverage } from "../lib/attention";

export interface MetaInspectorProps {
  backend: string;
  latencyMs?: number;
  attentionSource: "interpolated" | "final" | "idle";
  grid: number[][];
  fromMock?: boolean;
}

function StatBlock({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 rounded-konjo bg-konjo-surface/60 border border-konjo-line/60">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
        {label}
      </div>
      <div
        className="text-konjo-mono tabular-nums text-konjo-fg"
        style={{ fontSize: 13, color: accent ?? "var(--color-konjo-fg)" }}
      >
        {value}
      </div>
    </div>
  );
}

export function MetaInspector({
  backend, latencyMs, attentionSource, grid, fromMock,
}: MetaInspectorProps) {
  const peak = grid.length > 0 ? hotspot(grid) : { x: 0, y: 0, v: 0 };
  const cov = grid.length > 0 ? coverage(grid) : 0;
  const sourceColor =
    attentionSource === "final"
      ? "var(--color-konjo-good)"
      : attentionSource === "interpolated"
      ? "var(--color-konjo-accent)"
      : "var(--color-konjo-fg-muted)";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      <StatBlock label="backend" value={fromMock ? `${backend} · mock` : backend} accent={fromMock ? "var(--color-konjo-warm)" : undefined} />
      <StatBlock
        label="latency"
        value={latencyMs != null ? `${latencyMs.toFixed(0)} ms` : "—"}
      />
      <StatBlock
        label="attention"
        value={attentionSource}
        accent={sourceColor}
      />
      <StatBlock
        label="hotspot · coverage"
        value={`(${peak.x},${peak.y}) · ${(cov * 100).toFixed(0)}%`}
      />
    </div>
  );
}
