// Server-side only. Settings of the web app itself (`API_INTERNAL_URL` is read by the API client).

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

/**
 * The one address the site is opened at, such as `http://127.0.0.1:3000`. The CSRF check compares
 * the browser's `Origin` header with it, and https turns on Secure cookies.
 *
 * Read on every call, not at import time: `next build` imports modules without these settings.
 */
export function appOrigin(): string {
  const value = process.env.APP_ORIGIN;
  if (!value) throw new Error("APP_ORIGIN is not set");
  const url = URL.canParse(value) ? new URL(value) : null;
  // Exactly an origin (no path, no trailing slash), because it is compared as a string.
  if (!url || url.origin !== value || !["http:", "https:"].includes(url.protocol)) {
    throw new Error("APP_ORIGIN must look like https://example.com, with nothing after the host");
  }
  // Without https the session cookies would cross the network in clear text.
  if (url.protocol === "http:" && !LOOPBACK_HOSTS.has(url.hostname)) {
    throw new Error("APP_ORIGIN must use https unless it is this machine");
  }
  return value;
}

export function usesSecureCookies(): boolean {
  return appOrigin().startsWith("https://");
}
