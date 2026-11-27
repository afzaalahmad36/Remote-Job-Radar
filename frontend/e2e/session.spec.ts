// Sessions in the PRODUCTION build, against a stand-in API that counts refresh calls. The unit
// tests cannot show this: whether the build really shares one refresh map is only seen here.
import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, expect, type Page, test } from "@playwright/test";

const STUB_WEB = "http://127.0.0.1:3001";
const STUB_API = "http://127.0.0.1:8765";
const PASSWORD = "long-enough-password";

test.use({ baseURL: STUB_WEB });
test.beforeEach(async ({ request }) => {
  await request.post(`${STUB_API}/__reset`);
});

const state = async (request: APIRequestContext) =>
  (await (await request.get(`${STUB_API}/__state`)).json()) as {
    refreshCalls: number;
    ended: boolean[];
  };

/** `name=value` of every cookie the answer sets. */
const setBy = (response: { headersArray(): { name: string; value: string }[] }) =>
  response
    .headersArray()
    .filter((header) => header.name.toLowerCase() === "set-cookie")
    .map((header) => header.value.split(";")[0]!)
    .sort();

/** A signed-in user; returns the refresh cookie's value. */
async function signUp(request: APIRequestContext, email: string): Promise<string> {
  const response = await request.post("/api/auth/register", {
    headers: { origin: STUB_WEB },
    data: { email, password: PASSWORD },
  });
  expect(response.status()).toBe(204);
  const cookie = setBy(response).find((c) => c.startsWith("rjr_refresh="))!;
  return cookie.slice("rjr_refresh=".length);
}

// Plain `fetch`, so that each request carries exactly the cookie we give it and nothing else.
const me = (refresh: string) =>
  fetch(`${STUB_WEB}/api/v1/users/me`, { headers: { cookie: `rjr_refresh=${refresh}` } });

const cookiesOf = (response: Response) =>
  response.headers
    .getSetCookie()
    .map((header) => header.split(";")[0]!)
    .sort();

test("20 parallel requests with one refresh cookie make one refresh", async ({ request }) => {
  const old = await signUp(request, "parallel@example.com");

  const responses = await Promise.all(Array.from({ length: 20 }, () => me(old)));

  expect(responses.map((r) => r.status)).toEqual(Array(20).fill(200));
  expect(await state(request)).toEqual({ refreshCalls: 1, ended: [false] });
  const cookies = cookiesOf(responses[0]!);
  expect(cookies).toHaveLength(2);
  for (const response of responses) expect(cookiesOf(response)).toEqual(cookies);

  // A late request that still has the old cookie gets the same new pair, without a new call.
  const late = await me(old);
  expect(cookiesOf(late)).toEqual(cookies);
  expect((await state(request)).refreshCalls).toBe(1);
});

test("after 30 seconds the old cookie is reuse: the session ends", async ({ request }) => {
  test.setTimeout(60_000);
  const old = await signUp(request, "reuse@example.com");
  expect((await me(old)).status).toBe(200);

  await new Promise((resolve) => setTimeout(resolve, 31_000));
  const response = await me(old);

  expect(response.status).toBe(401);
  expect(await response.json()).toEqual({ code: "session_ended" });
  expect(cookiesOf(response)).toEqual(["rjr_access=", "rjr_refresh="]);
  expect(await state(request)).toEqual({ refreshCalls: 2, ended: [true] });
});

test("an API that is down costs no cookies, and the session works afterwards", async ({
  request,
}) => {
  const old = await signUp(request, "down@example.com");
  await request.post(`${STUB_API}/__down`, { data: { down: true } });

  const failed = await me(old);

  expect(failed.status).toBe(502);
  expect(cookiesOf(failed)).toEqual([]);
  await request.post(`${STUB_API}/__down`, { data: { down: false } });
  expect((await me(old)).status).toBe(200);
});

test("signing out during a refresh leaves no working refresh token", async ({ request }) => {
  const old = await signUp(request, "race@example.com");

  const refreshing = me(old);
  await new Promise((resolve) => setTimeout(resolve, 100)); // the API takes 500 ms
  const signOut = await fetch(`${STUB_WEB}/api/auth/logout`, {
    method: "POST",
    headers: { origin: STUB_WEB, cookie: `rjr_refresh=${old}` },
  });
  await refreshing;

  expect(signOut.status).toBe(204);
  expect((await state(request)).ended).toEqual([true]);
  expect((await me(old)).status).toBe(401);
});

test("a request from another origin changes nothing", async ({ request }) => {
  await signUp(request, "csrf@example.com");

  const response = await request.post("/api/auth/logout", {
    headers: { origin: "https://evil.example" },
  });

  expect(response.status()).toBe(403);
  expect((await state(request)).ended).toEqual([false]);
});

test("every page has its own nonce, the security headers and no caching", async ({ request }) => {
  const nonces = new Set<string>();
  for (const path of ["/", "/", "/login", "/register", "/no-such-page"]) {
    const response = await request.get(path);
    const headers = response.headers();
    const nonce = /'nonce-([^']+)'/.exec(headers["content-security-policy"] ?? "")?.[1];

    expect(nonce, path).toBeDefined();
    nonces.add(nonce!);
    expect(headers["x-content-type-options"]).toBe("nosniff");
    expect(headers["x-frame-options"]).toBe("DENY");
    expect(headers["referrer-policy"]).toBe("same-origin");
    expect(headers["cache-control"], path).toContain("no-store");
    // Every script tag carries the nonce, or the browser would refuse to run it.
    const scripts = (await response.text()).match(/<script[^>]*>/g) ?? [];
    expect(scripts.length, path).toBeGreaterThan(0);
    for (const tag of scripts) expect(tag, path).toContain(`nonce="${nonce}"`);
  }
  expect(nonces.size).toBe(5);
});

/** Collects what the browser refused because of the security policy, and other errors. */
function watch(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (message) => {
    const text = message.text();
    // A 401 from `/api/v1/users/me` is how a visitor without a session is recognised.
    if (message.type() === "error" && !text.includes("401 (Unauthorized)")) problems.push(text);
    if (text.includes("Content Security Policy")) problems.push(text);
  });
  page.on("pageerror", (error) => problems.push(error.message));
  return problems;
}

test("in a browser: register, refresh, sign out, with nothing blocked", async ({
  page,
  context,
  request,
}) => {
  const problems = watch(page);
  const names = async () => (await context.cookies()).map((c) => c.name).sort();

  await page.goto("/register");
  await page.getByLabel("Email").fill("browser@example.com");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page.getByText("browser@example.com")).toBeVisible();

  expect(await names()).toEqual(["rjr_access", "rjr_refresh"]);
  expect((await context.cookies()).every((c) => c.httpOnly && c.sameSite === "Lax")).toBe(true);
  expect(await page.evaluate(() => document.cookie)).toBe(""); // page scripts see no token
  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);

  await context.clearCookies({ name: "rjr_access" });
  await page.reload();
  await expect(page.getByText("browser@example.com")).toBeVisible();
  expect((await state(request)).refreshCalls).toBe(1);
  expect(await names()).toEqual(["rjr_access", "rjr_refresh"]);

  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login$/);
  expect(await names()).toEqual([]);

  await page.goto("/no-such-page");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Page not found");
  await page.waitForLoadState("networkidle");
  expect(problems.filter((text) => !text.includes("404 (Not Found)"))).toEqual([]);
});
