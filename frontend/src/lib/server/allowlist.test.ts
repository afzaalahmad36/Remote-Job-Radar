// @vitest-environment node
import { describe, expect, it } from "vitest";

import { FORWARD_RULES, type ForwardRule, forwardPath } from "./allowlist";

const ID = "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3b";
// A rule with a parameter, to test the UUID check before a later PR adds one for real.
const WITH_ID: readonly ForwardRule[] = [{ method: "GET", path: "/api/v1/matches/{job_id}" }];

describe("forwardPath", () => {
  it("allows only reading the current user for now", () => {
    expect(FORWARD_RULES).toEqual([{ method: "GET", path: "/api/v1/users/me" }]);
    expect(forwardPath("GET", ["users", "me"])).toBe("/api/v1/users/me");
  });

  it.each([
    ["POST", ["users", "me"]],
    ["GET", ["users"]],
    ["GET", ["users", "me", "extra"]],
    ["GET", ["users", "me", ""]],
    ["GET", ["Users", "me"]],
    ["GET", ["users/me"]],
    ["POST", ["auth", "login"]],
    ["POST", ["auth", "refresh"]],
    ["GET", ["cv"]],
    ["GET", []],
  ])("refuses %s %j", (method, segments) => {
    expect(forwardPath(method, segments)).toBeNull();
  });

  it("accepts a UUID where the path has a parameter, in either letter case", () => {
    for (const id of [ID, ID.toUpperCase()]) {
      expect(forwardPath("GET", ["matches", id], WITH_ID)).toBe(`/api/v1/matches/${id}`);
    }
  });

  it.each([
    "",
    ".",
    "..",
    "status",
    "1",
    `${ID}x`,
    `x${ID}`,
    `${ID}\n`,
    `${ID}?limit=1`,
    "%2e%2e",
    "..%2f..%2fauth",
    `${ID}/..`,
  ])("refuses %j as a parameter", (value) => {
    expect(forwardPath("GET", ["matches", value], WITH_ID)).toBeNull();
  });
});
