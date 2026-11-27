// @vitest-environment node
import { describe, expect, it } from "vitest";

import { hasAllowedOrigin } from "./origin";

const ORIGIN = "http://127.0.0.1:3000";

const request = (method: string, origin?: string) =>
  new Request(`${ORIGIN}/api/x`, { method, headers: origin === undefined ? {} : { origin } });

describe("hasAllowedOrigin", () => {
  it.each(["POST", "PUT", "PATCH", "DELETE"])("lets %s through from our own origin", (method) => {
    expect(hasAllowedOrigin(request(method, ORIGIN), ORIGIN)).toBe(true);
  });

  it.each([
    ["a missing header", undefined],
    ["another site", "https://evil.example"],
    ["the literal null", "null"],
    ["another name for this machine", "http://localhost:3000"],
    ["another port", "http://127.0.0.1:3001"],
    ["a longer host", "http://127.0.0.1:3000.evil.example"],
  ])("refuses a change with %s", (_name, origin) => {
    for (const method of ["POST", "PUT", "PATCH", "DELETE"]) {
      expect(hasAllowedOrigin(request(method, origin), ORIGIN)).toBe(false);
    }
  });

  it("does not ask reads for an Origin", () => {
    expect(hasAllowedOrigin(request("GET"), ORIGIN)).toBe(true);
    expect(hasAllowedOrigin(request("HEAD"), ORIGIN)).toBe(true);
  });
});
