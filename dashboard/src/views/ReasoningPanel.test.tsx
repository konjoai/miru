import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReasoningPanel } from "./ReasoningPanel";
import { MOCK_STEPS, MOCK_TRACE } from "../lib/mock";

describe("ReasoningPanel", () => {
  it("renders all step descriptions", () => {
    render(<ReasoningPanel steps={MOCK_STEPS} streaming={false} />);
    for (const s of MOCK_STEPS) expect(screen.getByText(s.description)).toBeInTheDocument();
  });

  it("renders the empty hint when no steps and not streaming", () => {
    render(<ReasoningPanel steps={[]} streaming={false} />);
    expect(screen.getByText(/drop an image or pick a sample/i)).toBeInTheDocument();
  });

  it("shows the thinking indicator while streaming", () => {
    render(<ReasoningPanel steps={MOCK_STEPS.slice(0, 2)} streaming />);
    expect(screen.getByText(/thinking/i)).toBeInTheDocument();
  });

  it("renders the answer when provided", () => {
    render(<ReasoningPanel steps={MOCK_STEPS} streaming={false} answer={MOCK_TRACE.answer} />);
    expect(screen.getByText(MOCK_TRACE.answer)).toBeInTheDocument();
  });
});
