// @vitest-environment node
// Our `/api` route handlers, called directly, with a fake FastAPI behind `fetch`.
import { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MAX_BODY_BYTES } from "@/lib/server/body";

import { POST as login } from "./auth/login/route";
import { POST as logout } from "./auth/logout/route";
import { POST as register } from "./auth/register/route";
import { GET as forward } from "./v1/[...path]/route";

const ORIGIN = "http://127.0.0.1:3000";
const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

const TOKENS = { access_token: "access-2", refresh_token: "refresh-2", expires_in: 1800 };
const USER = { id: "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3b", email: "a@example.com" };
const CREDENTIALS = { email: "a@example.com", password: "long-enough-password" };
const SESSION = "rjr_access=access-1; rjr_refresh=refresh-1";

/** What the fake API answers, by path. A path not listed is a test failure. */
function api(answers: Record<string, () => Response>): void {
  fetchMock.mockImplementation(async (url) => {
    const path = new URL(String(url)).pathname.replace("/api/v1/", "");
    const answer = answers[path];
    if (!answer) throw new Error(`unexpected call to ${path}`);
    return answer();
  });
}
const calledPaths = () =>
  fetchMock.mock.calls.map(([url]) => new URL(String(url)).pathname.replace("/api/v1/", ""));

function post(path: string, body?: unknown, headers: Record<string, string> = {}): NextRequest {
  return new NextRequest(`${ORIGIN}${path}`, {
    method: "POST",
    headers: { origin: ORIGIN, "content-type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function me(cookie?: string, path = ["users", "me"]) {
  const request = new NextRequest(`${ORIGIN}/api/v1/${path.join("/")}`, {
    headers: cookie ? { cookie } : {},
  });
  return forward(request, { params: Promise.resolve({ path }) });
}

const setCookies = (response: Response) => response.headers.getSetCookie().join("\n");

beforeEach(() => {
  (globalThis as { __rjrRefresh?: Map<string, unknown> }).__rjrRefresh?.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("API_INTERNAL_URL", "http://api.test:8000");
  vi.stubEnv("APP_ORIGIN", ORIGIN);
});

describe.each([
  ["login", login],
  ["register", register],
  ["logout", logout],
])("POST /api/auth/%s", (name, handler) => {
  it("refuses another origin, and a missing one, without calling the API", async () => {
    for (const headers of [{ origin: "https://evil.example" }, { origin: "" }]) {
      const response = await handler(post(`/api/auth/${name}`, CREDENTIALS, headers));

      expect(response.status).toBe(403);
      expect(setCookies(response)).toBe("");
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("login", () => {
  it("puts the tokens in httpOnly cookies and in no body", async () => {
    api({ "auth/login": () => json(200, TOKENS) });

    const response = await login(post("/api/auth/login", { ...CREDENTIALS, extra: "x" }));

    expect(response.status).toBe(204);
    expect(await response.text()).toBe("");
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(setCookies(response)).toMatch(/rjr_access=access-2;.*HttpOnly/i);
    expect(setCookies(response)).toMatch(/rjr_refresh=refresh-2;.*HttpOnly/i);
    // Only the two fields are passed on.
    expect(JSON.parse(String(fetchMock.mock.calls[0]![1]!.body))).toEqual(CREDENTIALS);
  });

  it.each([
    [401, 401, "invalid_credentials"],
    [422, 422, "invalid_input"],
    [429, 429, "rate_limited"],
    [500, 502, "api_unavailable"],
  ])("turns the API's %i into %i %s, without its text", async (upstream, status, code) => {
    api({ "auth/login": () => json(upstream, { code: "x", detail: "words from the API" }) });

    const response = await login(post("/api/auth/login", CREDENTIALS));

    expect(response.status).toBe(status);
    expect(await response.json()).toEqual({ code });
    expect(setCookies(response)).toBe("");
  });

  it.each([
    ["a form post", 415, { "content-type": "application/x-www-form-urlencoded" }, CREDENTIALS],
    ["a huge body", 413, {}, { ...CREDENTIALS, extra: "x".repeat(MAX_BODY_BYTES) }],
    ["a missing password", 400, {}, { email: "a@example.com" }],
    ["a number as password", 400, {}, { email: "a@example.com", password: 12345678 }],
    ["a bare null", 400, {}, null],
  ])("refuses %s without calling the API", async (_name, status, headers, body) => {
    const response = await login(post("/api/auth/login", body, headers));

    expect(response.status).toBe(status);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("register", () => {
  it("creates the account and signs in", async () => {
    api({ "auth/register": () => json(201, USER), "auth/login": () => json(200, TOKENS) });

    const response = await register(post("/api/auth/register", CREDENTIALS));

    expect(response.status).toBe(204);
    expect(calledPaths()).toEqual(["auth/register", "auth/login"]);
    expect(setCookies(response)).toContain("rjr_refresh=refresh-2");
  });

  it("says when the email is taken, and does not try to sign in", async () => {
    api({ "auth/register": () => json(409, { code: "conflict" }) });

    const response = await register(post("/api/auth/register", CREDENTIALS));

    expect(await response.json()).toEqual({ code: "email_taken" });
    expect(calledPaths()).toEqual(["auth/register"]);
  });

  it("says that the account exists when only the sign-in fails", async () => {
    api({ "auth/register": () => json(201, USER), "auth/login": () => json(429, {}) });

    const response = await register(post("/api/auth/register", CREDENTIALS));

    expect(await response.json()).toEqual({ code: "registered_not_signed_in" });
  });
});

describe("logout", () => {
  const signOut = () => logout(post("/api/auth/logout", undefined, { cookie: SESSION }));

  it("revokes the session at the API and clears the cookies", async () => {
    api({ "auth/logout": () => new Response(null, { status: 204 }) });

    const response = await signOut();

    expect(response.status).toBe(204);
    expect(JSON.parse(String(fetchMock.mock.calls[0]![1]!.body))).toEqual({
      refresh_token: "refresh-1",
    });
    expect(setCookies(response).match(/Max-Age=0/gi)).toHaveLength(2);
  });

  it("clears the cookies even when the API cannot revoke, and says so", async () => {
    api({ "auth/logout": () => json(500, {}) });

    const response = await signOut();

    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ code: "not_revoked" });
    expect(setCookies(response).match(/Max-Age=0/gi)).toHaveLength(2);
  });

  it("needs no API call without a session", async () => {
    const response = await logout(post("/api/auth/logout"));

    expect(response.status).toBe(204);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("forwarding", () => {
  it("calls the API with the access token and passes the answer on, uncached", async () => {
    api({ "users/me": () => json(200, USER) });

    const response = await me(SESSION);

    expect(await response.json()).toEqual(USER);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(setCookies(response)).toBe("");
    const headers = new Headers(fetchMock.mock.calls[0]![1]!.headers);
    expect(headers.get("authorization")).toBe("Bearer access-1");
    expect(headers.get("cookie")).toBeNull();
  });

  it("refreshes when the access cookie is gone, and writes the new cookies", async () => {
    api({ "auth/refresh": () => json(200, TOKENS), "users/me": () => json(200, USER) });

    const response = await me("rjr_refresh=refresh-1");

    expect(response.status).toBe(200);
    expect(setCookies(response)).toContain("rjr_access=access-2");
    expect(setCookies(response)).toContain("rjr_refresh=refresh-2");
  });

  it("writes the new cookies even when the call then fails", async () => {
    api({ "auth/refresh": () => json(200, TOKENS), "users/me": () => json(500, {}) });

    const response = await me("rjr_refresh=refresh-1");

    expect(response.status).toBe(502);
    expect(setCookies(response)).toContain("rjr_refresh=refresh-2");
  });

  it("clears the cookies when the session has ended", async () => {
    api({ "auth/refresh": () => json(401, {}), "users/me": () => json(401, {}) });

    const response = await me(SESSION);

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ code: "session_ended" });
    expect(setCookies(response).match(/Max-Age=0/gi)).toHaveLength(2);
  });

  it("says 'not signed in' to a visitor without cookies, without calling the API", async () => {
    const response = await me();

    expect(await response.json()).toEqual({ code: "not_signed_in" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("answers 404 for a path that is not on the list", async () => {
    for (const path of [["cv"], ["auth", "refresh"], ["users", "me", ".."]]) {
      expect((await me(SESSION, path)).status).toBe(404);
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
