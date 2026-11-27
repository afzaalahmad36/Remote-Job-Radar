// @vitest-environment node
import { describe, expect, it } from "vitest";

import { FORWARD_RULES, forwardPath } from "./allowlist";

const ID = "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3b";
const query = (text: string) => new URLSearchParams(text);

describe("forwardPath", () => {
  it("allows exactly these calls", () => {
    // A new rule must be added here on purpose.
    expect(FORWARD_RULES.map((rule) => `${rule.method} ${rule.path}`)).toEqual([
      "GET /api/v1/users/me",
      "GET /api/v1/matches",
      "GET /api/v1/matches/status",
      "GET /api/v1/matches/{job_id}",
      "POST /api/v1/matches/rescore",
    ]);
    expect(forwardPath("GET", ["users", "me"])).toBe("/api/v1/users/me");
    expect(forwardPath("GET", ["matches"])).toBe("/api/v1/matches");
    expect(forwardPath("GET", ["matches", "status"])).toBe("/api/v1/matches/status");
    expect(forwardPath("POST", ["matches", "rescore"])).toBe("/api/v1/matches/rescore");
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
    ["POST", ["matches"]],
    ["DELETE", ["matches", ID]],
    ["POST", ["matches", ID]],
    ["GET", ["matches", "rescore"]],
    ["GET", ["jobs"]],
  ])("refuses %s %j", (method, segments) => {
    expect(forwardPath(method, segments)).toBeNull();
  });

  it("accepts a UUID where the path has a parameter, in either letter case", () => {
    for (const id of [ID, ID.toUpperCase()]) {
      expect(forwardPath("GET", ["matches", id])).toBe(`/api/v1/matches/${id}`);
    }
  });

  it.each([
    "",
    ".",
    "..",
    "1",
    `${ID}x`,
    `x${ID}`,
    `${ID}\n`,
    `${ID}?limit=1`,
    "%2e%2e",
    "..%2f..%2fauth",
    `${ID}/..`,
  ])("refuses %j as a parameter", (value) => {
    expect(forwardPath("GET", ["matches", value])).toBeNull();
  });
});

describe("forwarded query", () => {
  it("passes on the listed parameters only", () => {
    const search = query("min_score=60&limit=20&offset=40&include_filtered=true&source_id=x&a=1");

    expect(forwardPath("GET", ["matches"], search)).toBe(
      "/api/v1/matches?limit=20&offset=40&min_score=60&include_filtered=true",
    );
  });

  it("takes a repeated parameter once", () => {
    expect(forwardPath("GET", ["matches"], query("limit=5&limit=100"))).toBe(
      "/api/v1/matches?limit=5",
    );
  });

  it("encodes a value, so it cannot add a parameter or change the path", () => {
    const search = new URLSearchParams({ min_score: "1&source_id=x#/../auth" });

    expect(forwardPath("GET", ["matches"], search)).toBe(
      "/api/v1/matches?min_score=1%26source_id%3Dx%23%2F..%2Fauth",
    );
  });

  it("drops the whole query where a rule lists none", () => {
    expect(forwardPath("GET", ["users", "me"], query("limit=1"))).toBe("/api/v1/users/me");
    expect(forwardPath("POST", ["matches", "rescore"], query("x=1"))).toBe(
      "/api/v1/matches/rescore",
    );
  });
});
