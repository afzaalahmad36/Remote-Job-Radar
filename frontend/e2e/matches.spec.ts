// The match list against the REAL stack: API, worker and database. Needs the demo account seeded
// with --demo-password (CI does both); skipped without E2E_API_URL.
//
// Only the invented CV of tests/fixtures/cv is used. It is uploaded through the API, because the
// web app has no CV page yet.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, expect, test } from "@playwright/test";

const API = process.env.E2E_API_URL;
const DEMO = { email: "demo@example.com", password: "demo-password-local-only" };
const CV = fileURLToPath(new URL("../../tests/fixtures/cv/alex_example.txt", import.meta.url));

async function uploadCv(request: APIRequestContext): Promise<void> {
  const login = await request.post(`${API}/api/v1/auth/login`, { data: DEMO });
  expect(login.status()).toBe(200);
  const { access_token: token } = (await login.json()) as { access_token: string };
  const upload = await request.put(`${API}/api/v1/cv`, {
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "text/plain" },
    data: await readFile(CV),
  });
  expect(upload.status()).toBe(200);
}

test("the demo account sees its ranked matches and why each got its score", async ({
  page,
  request,
}) => {
  test.skip(!API, "E2E_API_URL is not set");
  test.setTimeout(120_000);
  const blocked: string[] = [];
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) blocked.push(message.text());
  });
  page.on("pageerror", (error) => blocked.push(error.message));
  await uploadCv(request);

  await page.goto("/login");
  await page.getByLabel("Email").fill(DEMO.email);
  await page.getByLabel("Password").fill(DEMO.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  // Scoring runs in the background: the list waits for it and reads itself again.
  const cards = page.getByRole("article");
  await expect(cards.first()).toBeVisible({ timeout: 90_000 });
  await expect(page.getByText("Scores are updating")).toHaveCount(0, { timeout: 90_000 });

  // Best first, and without an embedding model every score says it is skills only.
  const scores = async () =>
    (await cards.locator("p:has-text('match score') span:first-child").allTextContents()).map(
      Number,
    );
  const all = await scores();
  expect(all.length).toBeGreaterThan(1);
  expect(all).toEqual([...all].sort((a, b) => b - a));
  await expect(cards.first().getByRole("list", { name: "Warnings" })).toContainText("Skills only");

  const first = cards.first();
  await first.getByRole("button", { name: "Why this score?" }).click();
  await expect(first.getByText("How the score was made")).toBeVisible();
  await expect(first.getByText(/^Work location: /)).toBeVisible();
  await expect(first.getByText(/^Skills: your CV has \d+ of the \d+ listed/)).toBeVisible();

  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);

  await page.getByLabel("Minimum score").fill(String(all[0]));
  await page.getByRole("button", { name: "Apply" }).click();
  await expect(page).toHaveURL(new RegExp(`\\?min=${all[0]}$`));
  await expect(cards.first()).toBeVisible();
  expect((await scores()).every((score) => score >= all[0]!)).toBe(true);

  expect(blocked).toEqual([]);
});
