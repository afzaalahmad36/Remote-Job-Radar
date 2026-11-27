import { defineConfig, devices } from "@playwright/test";

// Always against the production build (`npm run build` first): dev mode differs in the things
// we care about, such as security headers, caching and module state.
export default defineConfig({
  testDir: "./e2e",
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:3000" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run start",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: false,
    env: { NEXT_TELEMETRY_DISABLED: "1" },
  },
});
