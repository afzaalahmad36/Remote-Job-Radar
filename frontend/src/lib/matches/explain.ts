// The stored explanation of one score, and how it is worded.
//
// `openapi.json` only says "an object" for `reasons`, so its shape is written here. It mirrors
// `app/matching/reasons.py` (`MatchReasons`), which the API validates against before sending it.
// Parts that do not apply are left out, not sent as null.
import type { components } from "@/lib/api/schema";

import { REMOTE, SENIORITY } from "./format";

export type MatchDetail = components["schemas"]["MatchDetailOut"];

interface Salary {
  min?: number;
  max?: number;
  currency?: string;
  period?: string;
}

export interface FilterResult {
  name: "remote_policy" | "seniority" | "salary";
  outcome: "pass" | "fail" | "unknown";
  job?: string | Salary;
  preference?: string[] | Salary;
  job_per_year?: number;
  preference_per_year?: number;
  reason?:
    | "no_preference"
    | "job_value_missing"
    | "currency_differs"
    | "salary_period_missing"
    | "open_ended_below_minimum";
}

export interface Reasons {
  cap_applied: boolean;
  weights: { effective: { skills: number; embedding: number } };
  components: {
    skills?: { matched: number; total: number; coverage: number; cv_skills: number };
    embedding?: { normalised: number };
  };
  skills: { matched: string[]; missing: string[] };
  filters: FilterResult[];
}

const FILTER_NAMES: Record<FilterResult["name"], string> = {
  remote_policy: "Work location",
  seniority: "Seniority",
  salary: "Salary",
};

const OUTCOMES: Record<FilterResult["outcome"], string> = {
  pass: "passes",
  fail: "fails",
  unknown: "not checked",
};

const WHY: Record<NonNullable<FilterResult["reason"]>, string> = {
  no_preference: "you set no preference",
  job_value_missing: "the post does not say",
  currency_differs: "different currencies are not compared",
  salary_period_missing: "the post gives no pay period",
  open_ended_below_minimum: "the post says “from” an amount below your minimum, so it may pay more",
};

const VALUES: Record<string, string> = { ...REMOTE, ...SENIORITY };
const word = (value: string) => VALUES[value] ?? value;
const number = (value: number) => Math.round(value).toLocaleString("en-US");

/** What was compared, or "" when there is nothing to compare. */
function compared(filter: FilterResult): string {
  if (filter.name === "salary") {
    const { job_per_year: offered, preference_per_year: wanted } = filter;
    if (offered === undefined || wanted === undefined) return "";
    return `the post: ${number(offered)} a year; your minimum: ${number(wanted)} a year`;
  }
  const { job, preference } = filter;
  if (typeof job !== "string" || !Array.isArray(preference) || preference.length === 0) return "";
  return `the post: ${word(job)}; you want: ${preference.map(word).join(", ")}`;
}

/** Such as "Work location: fails (the post: On site; you want: Remote)". */
export function describeFilter(filter: FilterResult): string {
  const details = [filter.reason ? WHY[filter.reason] : "", compared(filter)].filter(Boolean);
  const bracket = details.length > 0 ? ` (${details.join("; ")})` : "";
  return `${FILTER_NAMES[filter.name]}: ${OUTCOMES[filter.outcome]}${bracket}`;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

/** The parts the score was added up from, one line each. */
export function describeParts(reasons: Reasons): string[] {
  const { skills, embedding } = reasons.components;
  const weights = reasons.weights.effective;
  const parts: string[] = [];
  if (skills && weights.skills > 0) {
    parts.push(
      `Skills: your CV has ${skills.matched} of the ${skills.total} listed ` +
        `(${percent(skills.coverage)}), counting for ${percent(weights.skills)} of the score`,
    );
  }
  if (embedding && weights.embedding > 0) {
    parts.push(
      `Similarity of the texts: ${percent(embedding.normalised)}, ` +
        `counting for ${percent(weights.embedding)} of the score`,
    );
  }
  if (reasons.cap_applied) parts.push("The score was capped because a filter failed");
  return parts;
}
