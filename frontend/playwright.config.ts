import { defineConfig, devices } from "@playwright/test";

// Always against the production build (`npm run build` first): dev mode differs in the things
// we care about, such as security headers, caching and module state.
const WEB = "http://127.0.0.1:3000";
const STUB_WEB = "http://127.0.0.1:3001"; // the same build, talking to `e2e/stub-api.mjs`
const STUB_API = "http://127.0.0.1:8765";

function web(origin: string, apiUrl: string) {
  return {
    command: `npx next start -H 127.0.0.1 -p ${new URL(origin).port}`,
    url: `${origin}/login`,
    reuseExistingServer: false,
    env: { NEXT_TELEMETRY_DISABLED: "1", APP_ORIGIN: origin, API_INTERNAL_URL: apiUrl },
  };
}

export default defineConfig({
  testDir: "./e2e",
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: { baseURL: WEB },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    // Without a backend, an address where nothing listens: the smoke tests need none.
    web(WEB, process.env.E2E_API_URL ?? "http://127.0.0.1:9"),
    { command: "node e2e/stub-api.mjs", port: 8765, reuseExistingServer: false },
    web(STUB_WEB, STUB_API),
  ],
});
