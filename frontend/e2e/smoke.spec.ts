import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("the home page loads cleanly, from this origin only", async ({ page, baseURL }) => {
  const consoleErrors: string[] = [];
  const foreignRequests: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("request", (request) => {
    if (new URL(request.url()).origin !== baseURL) foreignRequests.push(request.url());
  });

  const response = await page.goto("/");
  await page.waitForLoadState("networkidle");

  expect(response?.status()).toBe(200);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Remote Job Radar");
  expect(consoleErrors).toEqual([]);
  expect(foreignRequests).toEqual([]); // no CDN, font service or analytics
});

test("the home page has no serious accessibility violation", async ({ page }) => {
  await page.goto("/");

  const { violations } = await new AxeBuilder({ page }).analyze();

  const serious = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious.map((v) => v.id)).toEqual([]);
});
