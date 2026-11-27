import { expect, test } from "@playwright/test";

// Needs the Compose stack with the demo account seeded with --demo-password (CI does both).
// Without E2E_API_URL it is skipped, so the other tests run with no backend at all.
const API = process.env.E2E_API_URL;
// The public demo password of scripts/seed_demo.py: only ever for a throwaway local database.
const DEMO = { email: "demo@example.com", password: "demo-password-local-only" };

test("the seeded demo account can log in and has its 20 jobs", async ({ request }) => {
  test.skip(!API, "E2E_API_URL is not set");

  const login = await request.post(`${API}/api/v1/auth/login`, { data: DEMO });
  expect(login.status()).toBe(200);
  const { access_token: token } = (await login.json()) as { access_token: string };
  const jobs = await request.get(`${API}/api/v1/jobs?limit=1`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  expect(jobs.status()).toBe(200);
  expect(((await jobs.json()) as { total: number }).total).toBe(20);
});

test("a session of the web app survives losing its access token, against the real API", async ({
  page,
  context,
}) => {
  test.skip(!API, "E2E_API_URL is not set");
  // A new account each run. It stays in the throwaway database.
  const email = `web-e2e-${Date.now()}@example.com`;

  await page.goto("/register");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill("long-enough-password");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page.getByText(email)).toBeVisible();

  // The real API would end the session if these requests refreshed twice.
  await context.clearCookies({ name: "rjr_access" });
  const statuses = await page.evaluate(() =>
    Promise.all(Array.from({ length: 10 }, () => fetch("/api/v1/users/me").then((r) => r.status))),
  );
  expect(statuses).toEqual(Array(10).fill(200));
  await context.clearCookies({ name: "rjr_access" });
  await page.reload();
  await expect(page.getByText(email)).toBeVisible();

  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill("long-enough-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText(email)).toBeVisible();
});
