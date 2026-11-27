// @vitest-environment node
import { describe, expect, it } from "vitest";

import { describeFilter, describeParts, type FilterResult, type Reasons } from "./explain";

describe("describeFilter", () => {
  it.each<[FilterResult, string]>([
    [
      { name: "remote_policy", outcome: "fail", job: "onsite", preference: ["remote", "hybrid"] },
      "Work location: fails (the post: On site; you want: Remote, Hybrid)",
    ],
    [
      { name: "seniority", outcome: "unknown", preference: [], reason: "no_preference" },
      "Seniority: not checked (you set no preference)",
    ],
    [
      {
        name: "seniority",
        outcome: "unknown",
        preference: ["senior"],
        reason: "job_value_missing",
      },
      "Seniority: not checked (the post does not say)",
    ],
    [
      { name: "salary", outcome: "pass", job_per_year: 108000, preference_per_year: 90000 },
      "Salary: passes (the post: 108,000 a year; your minimum: 90,000 a year)",
    ],
    [
      { name: "salary", outcome: "unknown", reason: "currency_differs" },
      "Salary: not checked (different currencies are not compared)",
    ],
  ])("words %j", (filter, words) => {
    expect(describeFilter(filter)).toBe(words);
  });
});

describe("describeParts", () => {
  const reasons: Reasons = {
    cap_applied: false,
    weights: { effective: { skills: 0.6, embedding: 0.4 } },
    components: {
      skills: { matched: 3, total: 4, coverage: 0.75, cv_skills: 9 },
      embedding: { normalised: 0.5 },
    },
    skills: { matched: [], missing: [] },
    filters: [],
  };

  it("gives each part and its weight", () => {
    expect(describeParts(reasons)).toEqual([
      "Skills: your CV has 3 of the 4 listed (75%), counting for 60% of the score",
      "Similarity of the texts: 50%, counting for 40% of the score",
    ]);
  });

  it("leaves out a part that did not count, and says when the score was capped", () => {
    const lines = describeParts({
      ...reasons,
      cap_applied: true,
      weights: { effective: { skills: 1, embedding: 0 } },
    });

    expect(lines).toEqual([
      "Skills: your CV has 3 of the 4 listed (75%), counting for 100% of the score",
      "The score was capped because a filter failed",
    ]);
  });
});
