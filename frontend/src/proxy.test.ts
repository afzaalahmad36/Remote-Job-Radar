// @vitest-environment node
import { NextRequest } from "next/server";
import { describe, expect, it, vi } from "vitest";

import { contentSecurityPolicy, proxy } from "./proxy";

describe("contentSecurityPolicy", () => {
  it("allows only this origin and this request's nonce in production", () => {
    const policy = contentSecurityPolicy("abc", false, false);

    expect(policy).toContain("script-src 'self' 'nonce-abc' 'strict-dynamic';");
    expect(policy).toContain("style-src 'self' 'nonce-abc';");
    expect(policy).toContain("connect-src 'self'");
    expect(policy).toContain("frame-ancestors 'none'");
    expect(policy).not.toMatch(/unsafe|https:|\*|upgrade-insecure-requests/);
  });

  it("upgrades http requests only when the site itself is https", () => {
    expect(contentSecurityPolicy("abc", true, false)).toContain("upgrade-insecure-requests");
  });
});

describe("proxy", () => {
  it("uses a new nonce for every request", () => {
    vi.stubEnv("APP_ORIGIN", "http://127.0.0.1:3000");
    const policy = () =>
      proxy(new NextRequest("http://127.0.0.1:3000/login")).headers.get("content-security-policy");

    expect(policy()).toMatch(/'nonce-[A-Za-z0-9+/=]{20,}'/);
    expect(policy()).not.toBe(policy());
  });
});
