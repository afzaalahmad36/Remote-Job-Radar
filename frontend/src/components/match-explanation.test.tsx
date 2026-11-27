import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WhyThisScore } from "./match-explanation";

const router = { replace: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const ID = "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3b";

const REASONS = {
  cap_applied: false,
  weights: { effective: { skills: 1, embedding: 0 } },
  components: { skills: { matched: 1, total: 2, coverage: 0.5, cv_skills: 3 } },
  skills: { matched: ["Python"], missing: ["Docker"] },
  filters: [{ name: "salary", outcome: "unknown", reason: "no_preference" }],
};

beforeEach(() => {
  router.replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(cleanup);

const button = () => screen.getByRole("button", { name: "Why this score?" });

describe("WhyThisScore", () => {
  it("asks for nothing until it is opened", () => {
    render(<WhyThisScore jobId={ID} />);

    expect(button().getAttribute("aria-expanded")).toBe("false");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("opens with the stored explanation, and closes again", async () => {
    fetchMock.mockResolvedValue(
      json(200, { reasons: REASONS, note_messages: ["The score is skills only."] }),
    );
    const { container } = render(<WhyThisScore jobId={ID} />);

    fireEvent.click(button());

    await screen.findByText("How the score was made");
    expect(fetchMock.mock.calls[0]![0]).toBe(`/api/v1/matches/${ID}`);
    for (const text of [
      "Skills you have",
      "Python",
      "Skills you are missing",
      "Docker",
      "Salary: not checked (you set no preference)",
      "The score is skills only.",
    ]) {
      expect(container.textContent).toContain(text);
    }
    expect(button().getAttribute("aria-expanded")).toBe("true");

    fireEvent.click(button());

    expect(screen.queryByText("How the score was made")).toBeNull();
  });

  it.each([
    [json(200, { reasons: null, note_messages: [] }), "outdated"],
    [json(404, { code: "not_found" }), "no score any more"],
    [json(502, { code: "api_unavailable" }), "not available"],
  ])("says when there is nothing to show", async (response, words) => {
    fetchMock.mockResolvedValue(response);
    const { container } = render(<WhyThisScore jobId={ID} />);

    fireEvent.click(button());

    await vi.waitFor(() => expect(container.textContent).toContain(words));
  });

  it("sends a lost session to sign in", async () => {
    fetchMock.mockResolvedValue(json(401, { code: "not_signed_in" }));
    render(<WhyThisScore jobId={ID} />);

    fireEvent.click(button());

    await vi.waitFor(() => expect(router.replace).toHaveBeenCalledWith("/login"));
  });
});
