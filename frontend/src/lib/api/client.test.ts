// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiFetch, apiGet } from "./client";

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("API_INTERNAL_URL", "http://api.test:8000");
});

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

describe("apiFetch", () => {
  it("sends the token, never caches and never follows a redirect", async () => {
    fetchMock.mockResolvedValue(json(200, { items: [] }));

    await apiGet("/api/v1/sources", "access-token");

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toBe("http://api.test:8000/api/v1/sources");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer access-token");
    expect(init?.cache).toBe("no-store");
    expect(init?.redirect).toBe("error");
    expect(init?.signal).toBeInstanceOf(AbortSignal);
  });

  it("turns the API's error shape into an ApiError whose message holds no body text", async () => {
    fetchMock.mockResolvedValue(json(422, { code: "cv_unreadable", detail: "secret CV words" }));

    const error = await apiFetch("/api/v1/cv").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, code: "cv_unreadable" });
    expect((error as ApiError).message).not.toContain("secret");
  });

  it("copes with an error body that is not JSON", async () => {
    fetchMock.mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 }));

    await expect(apiFetch("/api/v1/health")).rejects.toMatchObject({
      status: 502,
      code: "unknown_error",
    });
  });

  it("reports an unreachable API without the underlying error", async () => {
    fetchMock.mockRejectedValue(new TypeError("connect ECONNREFUSED 10.0.0.5:8000"));

    await expect(apiFetch("/api/v1/health")).rejects.toMatchObject({
      status: 0,
      code: "api_unreachable",
    });
  });

  it.each([
    "https://evil.example/api/v1/health",
    "//evil.example/api/v1/health",
    "/api/v1/../../admin",
    "/api/v1/%2e%2e/%2e%2e/admin",
    "/docs",
  ])("refuses to call %s", async (path) => {
    await expect(apiFetch(path)).rejects.toThrow("outside the API");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fails loudly when the API address is not configured", async () => {
    vi.stubEnv("API_INTERNAL_URL", "");

    await expect(apiFetch("/api/v1/health")).rejects.toThrow("API_INTERNAL_URL");
  });
});
