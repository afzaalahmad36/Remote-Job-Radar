// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from "vitest";

import { callOurApi } from "./browser-api";

const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

describe("callOurApi", () => {
  it("gives the data of a good answer, and never uses a cache", async () => {
    fetchMock.mockResolvedValue(json(202, { status: "queued" }));

    const result = await callOurApi("/api/v1/matches/rescore", { method: "POST" });

    expect(result).toEqual({ kind: "ok", data: { status: "queued" } });
    expect(fetchMock.mock.calls[0]).toEqual([
      "/api/v1/matches/rescore",
      { method: "POST", cache: "no-store" },
    ]);
  });

  it.each([
    ["not_signed_in", "/login"],
    ["session_ended", "/login?notice=session_ended"],
  ])("recognises a 401 %s", async (code, signIn) => {
    fetchMock.mockResolvedValue(json(401, { code }));

    expect(await callOurApi("/api/v1/matches")).toEqual({ kind: "signed_out", signIn });
  });

  it.each([
    ["an error with a code", () => json(429, { code: "rate_limited" }), "rate_limited"],
    ["an error without one", () => new Response("<html>", { status: 502 }), null],
    ["a good status with a broken body", () => new Response("<html>"), null],
  ])("reports %s", async (_name, respond, code) => {
    fetchMock.mockResolvedValue(respond());

    expect(await callOurApi("/api/v1/matches")).toEqual({ kind: "error", code });
  });

  it("reports a lost connection", async () => {
    fetchMock.mockRejectedValue(new TypeError("offline"));

    expect(await callOurApi("/api/v1/matches")).toEqual({ kind: "error", code: null });
  });

  it("throws only when the caller gave up", async () => {
    const controller = new AbortController();
    controller.abort();
    fetchMock.mockRejectedValue(new DOMException("aborted", "AbortError"));

    await expect(
      callOurApi("/api/v1/matches", { signal: controller.signal }),
    ).rejects.toThrow("aborted");
  });
});
