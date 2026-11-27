// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import { appOrigin, usesSecureCookies } from "./config";

const set = (value: string) => vi.stubEnv("APP_ORIGIN", value);

describe("appOrigin", () => {
  it.each([
    ["http://127.0.0.1:3000", false],
    ["http://localhost:3000", false],
    ["http://[::1]:3000", false],
    ["https://jobs.example.com", true],
  ])("accepts %s", (value, secure) => {
    set(value);

    expect(appOrigin()).toBe(value);
    expect(usesSecureCookies()).toBe(secure);
  });

  it.each([
    "",
    "jobs.example.com",
    "https://jobs.example.com/",
    "https://jobs.example.com/app",
    "ftp://jobs.example.com",
    "http://jobs.example.com",
    "http://localhost.example.com:3000",
  ])("refuses %j", (value) => {
    set(value);

    expect(() => appOrigin()).toThrow("APP_ORIGIN");
  });
});
