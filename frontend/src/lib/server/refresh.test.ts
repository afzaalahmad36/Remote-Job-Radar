// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api/client";

import { forgetRefresh, KEEP_MS, MAX_ENTRIES, refreshTokens } from "./refresh";

const fetchMock = vi.fn<typeof fetch>();
const map = () => (globalThis as { __rjrRefresh?: Map<string, unknown> }).__rjrRefresh!;

const pair = (n: number) => ({
  access_token: `access-${n}`,
  refresh_token: `refresh-${n}`,
  token_type: "bearer",
  expires_in: 1800,
});
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const sentToken = (call: number) =>
  (JSON.parse(String(fetchMock.mock.calls[call]![1]!.body)) as { refresh_token: string })
    .refresh_token;

/** An API that gives pair 1, then pair 2, and so on. */
function apiThatRotates(): void {
  let n = 0;
  fetchMock.mockImplementation(async () => json(200, pair(++n)));
}

beforeEach(() => {
  map().clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("API_INTERNAL_URL", "http://api.test:8000");
  vi.useFakeTimers({ toFake: ["Date"] }); // only the clock: promises behave normally
});

afterEach(() => {
  vi.useRealTimers();
});

const later = (ms: number) => vi.setSystemTime(Date.now() + ms);

describe("refreshTokens", () => {
  it("makes one API call for many parallel callers, and gives them all the same pair", async () => {
    apiThatRotates();

    const results = await Promise.all(Array.from({ length: 20 }, () => refreshTokens("old")));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(new Set(results.map((tokens) => tokens.refresh_token))).toEqual(new Set(["refresh-1"]));
    expect(String(fetchMock.mock.calls[0]![0])).toBe("http://api.test:8000/api/v1/auth/refresh");
    expect(sentToken(0)).toBe("old");
  });

  it("keeps different sessions apart", async () => {
    apiThatRotates();

    const [a, b] = await Promise.all([refreshTokens("user-a"), refreshTokens("user-b")]);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(a.access_token).not.toBe(b.access_token);
  });

  it("answers a late request from memory, but only for 30 seconds", async () => {
    apiThatRotates();
    await refreshTokens("old");

    later(KEEP_MS - 1);
    expect((await refreshTokens("old")).refresh_token).toBe("refresh-1");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    later(1);
    await refreshTokens("old");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("counts the 30 seconds from the answer, not from the request", async () => {
    let answer!: (response: Response) => void;
    fetchMock.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    const first = refreshTokens("old");

    later(KEEP_MS * 2); // a very slow API: still the same call
    const second = refreshTokens("old");
    answer(json(200, pair(1)));

    expect(await second).toBe(await first);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["a 401", () => Promise.resolve(json(401, { code: "authentication_failed" })), 401],
    ["a 429", () => Promise.resolve(json(429, { code: "rate_limited" })), 429],
    ["a 503", () => Promise.resolve(json(503, {})), 503],
    ["no connection", () => Promise.reject(new TypeError("refused")), 0],
    ["a body that is not JSON", () => Promise.resolve(new Response("<html>")), 502],
    ["a body without tokens", () => Promise.resolve(json(200, { access_token: "x" })), 502],
  ])("does not remember a failure: %s", async (_name, respond, status) => {
    fetchMock.mockImplementationOnce(respond);

    await expect(refreshTokens("old")).rejects.toMatchObject({ status });

    expect(map().size).toBe(0);
    fetchMock.mockResolvedValue(json(200, pair(1)));
    await expect(refreshTokens("old")).resolves.toMatchObject({ refresh_token: "refresh-1" });
  });

  it("passes on no text from the API's answer", async () => {
    fetchMock.mockResolvedValue(json(422, { code: "validation_error", detail: "echoed secret" }));

    const error = await refreshTokens("old").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).detail).toBeNull();
    expect((error as ApiError).message).not.toContain("secret");
  });

  it("never uses a raw token as a key", async () => {
    apiThatRotates();
    await refreshTokens("very-secret-token");

    for (const key of map().keys()) expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("writes nothing to the console, whether it works or fails", async () => {
    const spies = (["log", "info", "warn", "error", "debug"] as const).map((level) =>
      vi.spyOn(console, level),
    );
    apiThatRotates();
    await refreshTokens("old");
    fetchMock.mockResolvedValue(json(401, {}));
    await refreshTokens("other").catch(() => undefined);

    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  });

  it("never holds more than the maximum, dropping the oldest finished entry", async () => {
    apiThatRotates();
    for (let i = 0; i < MAX_ENTRIES + 5; i++) await refreshTokens(`token-${i}`);

    expect(map().size).toBe(MAX_ENTRIES);
    fetchMock.mockClear();
    await refreshTokens(`token-${MAX_ENTRIES + 4}`); // newest: still there
    expect(fetchMock).not.toHaveBeenCalled();
    await refreshTokens("token-0"); // oldest: gone
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("refuses new work rather than drop a call that is still running", async () => {
    fetchMock.mockReturnValue(new Promise(() => undefined)); // the API never answers
    for (let i = 0; i < MAX_ENTRIES; i++) void refreshTokens(`token-${i}`);

    await expect(refreshTokens("one-more")).rejects.toMatchObject({ code: "web_busy" });
    void refreshTokens("token-0"); // a running one is still shared

    expect(fetchMock).toHaveBeenCalledTimes(MAX_ENTRIES);
  });

  it("is shared by a second copy of the module", async () => {
    apiThatRotates();
    await refreshTokens("old");

    vi.resetModules(); // the next import evaluates the module again
    const copy = await import("./refresh");
    await copy.refreshTokens("old");

    expect(copy.refreshTokens).not.toBe(refreshTokens);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("forgetRefresh", () => {
  it("drops the entry of a token, and the older ones that lead to it", async () => {
    apiThatRotates();
    await refreshTokens("old"); // old -> refresh-1
    await refreshTokens("refresh-1"); // refresh-1 -> refresh-2
    await refreshTokens("someone-else");

    forgetRefresh("refresh-2"); // sign-out with the newest cookie

    expect(map().size).toBe(1); // only the other session is left
  });

  it("drops the entry when sign-out comes with the old cookie", async () => {
    apiThatRotates();
    await refreshTokens("old");

    forgetRefresh("old");

    expect(map().size).toBe(0);
  });
});
