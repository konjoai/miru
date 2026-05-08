import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QuestionBar } from "./QuestionBar";

describe("QuestionBar", () => {
  it("calls onSubmit when the button is clicked", async () => {
    const onSubmit = vi.fn();
    render(<QuestionBar value="hi" onChange={() => {}} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("calls onSubmit on Enter", async () => {
    const onSubmit = vi.fn();
    render(<QuestionBar value="hi" onChange={() => {}} onSubmit={onSubmit} />);
    await userEvent.type(screen.getByRole("textbox"), "{enter}");
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("does not call onSubmit on Enter when disabled", async () => {
    const onSubmit = vi.fn();
    render(<QuestionBar value="hi" onChange={() => {}} onSubmit={onSubmit} disabled />);
    await userEvent.type(screen.getByRole("textbox"), "{enter}");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("renders a custom submitLabel", () => {
    render(<QuestionBar value="" onChange={() => {}} onSubmit={() => {}} submitLabel="thinking…" />);
    expect(screen.getByRole("button", { name: /thinking/i })).toBeInTheDocument();
  });
});
