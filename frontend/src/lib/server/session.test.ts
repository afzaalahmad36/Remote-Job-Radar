// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api/client";

import { callAsUser } from "./session";

const fetchMock = vi.fn<typeof fetch>(); // only `/auth/refresh` goes through here
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const NEW = { access_token: "new-access", refresh_token: "new-refresh", expires_in: 1800 };

const ok = () => Promise.resolve(new Response("ok"));
const fail = (status: number) => () => Promise.reject(new ApiError(status, "some_code"));

beforeEach(() => {
  (globalThis as { __rjrRefresh?: Map<string, unknown> }).__rjrRefresh?.clear();
  fetchMock.mockReset();
  fetchMock.mockImplementation(async () => json(200, NEW));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("API_INTERNAL_URL", "http://api.test:8000");
});

const session = { access: "old-access", refresh: "old-refresh" };

describe("callAsUser", () => {
  it("uses the access token as it is when the API accepts it", async () => {
    const call = vi.fn(ok);

    const result = await callAsUser(session, call);

    expect(result).toMatchObject({ kind: "response", tokens: undefined });
    expect(call).toHaveBeenCalledExactlyOnceWith("old-access");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refreshes first when there is no access cookie", async () => {
    const call = vi.fn(ok);

    const result = await callAsUser({ access: undefined, refresh: "old-refresh" }, call);

    expect(result).toMatchObject({ kind: "response", tokens: NEW });
    expect(call).toHaveBeenCalledExactlyOnceWith("new-access");
  });

  it("refreshes and retries once on a 401", async () => {
    const call = vi.fn(ok).mockImplementationOnce(fail(401));

    const result = await callAsUser(session, call);

    expect(result).toMatchObject({ kind: "response", tokens: NEW });
    expect(call.mock.calls).toEqual([["old-access"], ["new-access"]]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("ends the session, without looping, when the new token is refused too", async () => {
    const call = vi.fn(fail(401));

    expect(await callAsUser(session, call)).toEqual({ kind: "ended" });
    expect(call).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("calls only once when it had to refresh before calling", async () => {
    const call = vi.fn(fail(401));

    const result = await callAsUser({ access: undefined, refresh: "old-refresh" }, call);

    expect(result).toEqual({ kind: "ended" });
    expect(call).toHaveBeenCalledTimes(1);
  });

  it("ends the session when there is nothing to refresh with", async () => {
    const call = vi.fn(fail(401));

    expect(await callAsUser({ access: undefined, refresh: undefined }, call)).toEqual({
      kind: "ended",
    });
    expect(await callAsUser({ access: "old-access", refresh: undefined }, call)).toEqual({
      kind: "ended",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 422])("ends the session when the refresh answers %i", async (status) => {
    fetchMock.mockImplementation(async () => json(status, {}));

    expect(await callAsUser(session, fail(401))).toEqual({ kind: "ended" });
  });

  it.each([429, 500, 503])("keeps the session when the refresh answers %i", async (status) => {
    fetchMock.mockImplementation(async () => json(status, {}));

    const result = await callAsUser(session, fail(401));

    expect(result).toMatchObject({ kind: "error", error: { status } });
  });

  it("keeps the session when the API cannot be reached for the refresh", async () => {
    fetchMock.mockRejectedValue(new TypeError("refused"));

    const result = await callAsUser(session, fail(401));

    expect(result).toMatchObject({ kind: "error", error: { status: 0 } });
  });

  it.each([0, 404, 409, 429, 500])("does not refresh or retry on a %i", async (status) => {
    const call = vi.fn(fail(status));

    const result = await callAsUser(session, call);

    expect(result).toMatchObject({ kind: "error", error: { status }, tokens: undefined });
    expect(call).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("still hands back the new tokens when the call fails after a refresh", async () => {
    const call = vi.fn(fail(500)).mockImplementationOnce(fail(401));

    const result = await callAsUser(session, call);

    // Losing them here would sign the user out on their next request.
    expect(result).toMatchObject({ kind: "error", error: { status: 500 }, tokens: NEW });
  });

  it("shares one refresh between parallel calls of one session", async () => {
    const call = (token: string) => (token === "old-access" ? fail(401)() : ok());

    const results = await Promise.all(Array.from({ length: 10 }, () => callAsUser(session, call)));

    expect(results.every((result) => result.kind === "response")).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not swallow a programming error", async () => {
    await expect(
      callAsUser(session, () => Promise.reject(new TypeError("a bug"))),
    ).rejects.toThrow("a bug");
  });
});
