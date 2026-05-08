import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StepTimeline } from "./StepTimeline";
import type { ReasoningStep } from "../lib/types";

const STEPS: ReasoningStep[] = [
  { step: 1, description: "a", confidence: 0.9 },
  { step: 2, description: "b", confidence: 0.8 },
  { step: 3, description: "c", confidence: 0.7 },
];

describe("StepTimeline", () => {
  it("renders the step counter as cursor / total", () => {
    render(<StepTimeline steps={STEPS} cursor={2} onCursorChange={() => {}} />);
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
  });

  it("disables play when not finalized", () => {
    render(<StepTimeline steps={STEPS} cursor={0} onCursorChange={() => {}} live />);
    const play = screen.getByRole("button", { name: /play/i });
    expect(play).toBeDisabled();
  });

  it("enables play when finalized", () => {
    render(<StepTimeline steps={STEPS} cursor={3} onCursorChange={() => {}} finalized />);
    expect(screen.getByRole("button", { name: /play/i })).toBeEnabled();
  });

  it("shows a live indicator when streaming", () => {
    render(<StepTimeline steps={STEPS} cursor={1} onCursorChange={() => {}} live />);
    expect(screen.getByText("live")).toBeInTheDocument();
  });

  it("calls onCursorChange when the user clicks the track", async () => {
    const cb = vi.fn();
    const { container } = render(
      <StepTimeline steps={STEPS} cursor={0} onCursorChange={cb} finalized />,
    );
    const track = container.querySelector(".cursor-pointer") as HTMLElement;
    expect(track).toBeTruthy();
    // Force-click the track to verify the handler fires (jsdom always reports rect.width=0).
    await userEvent.click(track);
    expect(cb).toHaveBeenCalled();
  });
});
