import { expect, test } from "@playwright/test";

// Needs the Compose stack with the demo account seeded with --demo-password (CI does both).
// Without E2E_API_URL it is skipped, so the smoke tests run with no backend at all.
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
