// @vitest-environment node
import { describe, expect, it } from "vitest";

import { filtersAddress, matchesPath, parseFilters } from "./filters";

describe("filters", () => {
  it("default to everything on page 1", () => {
    const filters = parseFilters({});

    expect(filters).toEqual({ min: null, all: false, page: 1 });
    expect(filtersAddress(filters)).toBe("/");
    expect(matchesPath(filters)).toBe("/api/v1/matches?limit=20&offset=0");
  });

  it("go to the address and to the API", () => {
    const filters = parseFilters({ min: "60", all: "1", page: "3" });

    expect(filtersAddress(filters)).toBe("/?min=60&all=1&page=3");
    expect(matchesPath(filters)).toBe(
      "/api/v1/matches?limit=20&offset=40&min_score=60&include_filtered=true",
    );
  });

  it("keep a minimum score of 0", () => {
    expect(parseFilters({ min: "0" }).min).toBe(0);
  });

  it.each([
    [{ min: "101" }],
    [{ min: "-1" }],
    [{ min: "6.5" }],
    [{ min: "1e2" }],
    [{ min: "" }],
    [{ min: ["60", "70"] }],
    [{ all: "true" }],
    [{ page: "0" }],
    [{ page: "5001" }],
    [{ page: "abc" }],
  ])("ignore what makes no sense: %j", (params) => {
    expect(parseFilters(params)).toEqual({ min: null, all: false, page: 1 });
  });
});
