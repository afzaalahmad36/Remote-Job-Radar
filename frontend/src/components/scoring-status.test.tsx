import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { MatchPage } from "@/lib/matches/format";

import { POLL_DELAYS_MS, ScoringStatus } from "./scoring-status";

const router = { replace: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const onChange = vi.fn();

const scoring = (state: MatchPage["scoring"]["state"]): MatchPage["scoring"] => ({
  state,
  requested_at: null,
  finished_at: null,
  failure_count: 0,
  max_attempts: 3,
  error: null,
});
const apiSays = (state: MatchPage["scoring"]["state"]) =>
  fetchMock.mockImplementation(async () => json(200, scoring(state)));

/** Let time pass, with the requests it starts finishing. */
const pass = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));
const total = POLL_DELAYS_MS.reduce((sum, delay) => sum + delay, 0);

beforeEach(() => {
  router.replace.mockReset();
  onChange.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.useFakeTimers();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const open = (state: MatchPage["scoring"]["state"]) =>
  render(<ScoringStatus scoring={scoring(state)} onChange={onChange} />);

describe("ScoringStatus", () => {
  it.each(["up_to_date", "no_cv"] as const)("says nothing when %s", async (state) => {
    const { container } = open(state);

    await pass(total);

    expect(container.textContent).toBe("");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(["pending", "running"] as const)("waits while %s, then reloads once", async (state) => {
    apiSays(state);
    open(state);
    expect(screen.getByRole("status").textContent).toContain("updating");

    await pass(3_000);
    await pass(3_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onChange).not.toHaveBeenCalled();

    apiSays("up_to_date");
    await pass(3_000);
    await pass(60_000);

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/v1/matches/status");
  });

  it("gives up after two minutes, and can be asked to check again", async () => {
    apiSays("pending");
    open("pending");

    await pass(total + 60_000);

    expect(total).toBe(120_000);
    expect(fetchMock).toHaveBeenCalledTimes(POLL_DELAYS_MS.length);
    expect(screen.getByRole("status").textContent).toContain("longer than usual");

    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    await pass(3_000);

    expect(onChange).toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(POLL_DELAYS_MS.length + 1);
  });

  it("stops at the first failed check", async () => {
    fetchMock.mockResolvedValue(json(502, { code: "api_unavailable" }));
    open("pending");

    await pass(total);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("status").textContent).toContain("Could not check");
  });

  it("stops when the page is left", async () => {
    apiSays("pending");
    const { unmount } = open("pending");
    await pass(3_000);

    unmount();
    await pass(total);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("sends a lost session to sign in", async () => {
    fetchMock.mockResolvedValue(json(401, { code: "session_ended" }));
    open("pending");

    await pass(3_000);

    expect(router.replace).toHaveBeenCalledWith("/login?notice=session_ended");
  });

  it("offers a re-score after a failure", async () => {
    fetchMock.mockResolvedValue(json(202, { status: "queued" }));
    open("failed");

    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Try again" })));

    expect(fetchMock.mock.calls[0]).toEqual([
      "/api/v1/matches/rescore",
      { method: "POST", cache: "no-store" },
    ]);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("says why a re-score did not work", async () => {
    fetchMock.mockResolvedValue(json(429, { code: "rate_limited" }));
    const { container } = open("failed");

    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Try again" })));

    expect(container.textContent).toContain("Too many attempts");
    expect(onChange).not.toHaveBeenCalled();
  });
});
