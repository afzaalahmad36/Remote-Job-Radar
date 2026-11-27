// How the values of a match are worded. The `Record` types come from `openapi.json`: a new value
// in the API fails the type check here instead of showing up blank.
import type { components } from "@/lib/api/schema";

export type Match = components["schemas"]["MatchOut"];
export type MatchPage = components["schemas"]["MatchPage"];
export type Job = components["schemas"]["MatchJobOut"];
type NoteCode = NonNullable<Match["note_codes"]>[number];

export const REMOTE: Record<NonNullable<Job["remote_policy"]>, string> = {
  remote: "Remote",
  hybrid: "Hybrid",
  onsite: "On site",
};

export const SENIORITY: Record<NonNullable<Job["seniority"]>, string> = {
  intern: "Intern",
  junior: "Junior",
  mid: "Mid-level",
  senior: "Senior",
  lead: "Lead",
  principal: "Principal",
};

const PERIOD: Record<NonNullable<Job["salary_period"]>, string> = {
  year: "year",
  month: "month",
  week: "week",
  day: "day",
  hour: "hour",
};

const WARNINGS: Record<NoteCode, string> = {
  no_embedding: "Skills only",
  filters_failed: "Fails a filter",
  no_job_stack: "No technologies listed",
  no_cv_skills: "Your CV has no skills",
  not_english: "Not in English",
  skills_truncated: "Some skills not shown",
};

/** The short warnings of a list item. */
export function warnings(match: Match): string[] {
  // null: the stored explanation cannot be read, which is not the same as "nothing to say".
  if (match.note_codes === null) {
    const failed = match.filters_failed ? [WARNINGS.filters_failed] : [];
    return [...failed, "Explanation outdated"];
  }
  return match.note_codes.map((code) => WARNINGS[code]);
}

/** "USD 90,000–120,000 per year", or null when the post gives no amount. */
export function salary(job: Job): string | null {
  const { salary_min: min, salary_max: max, salary_currency: currency } = job;
  if (min === null && max === null) return null;
  const number = (value: number) => value.toLocaleString("en-US");
  let amount: string;
  if (min === null) amount = `up to ${number(max!)}`;
  else if (max === null) amount = `from ${number(min)}`;
  else amount = min === max ? number(min) : `${number(min)}–${number(max)}`;
  const period = job.salary_period ? ` per ${PERIOD[job.salary_period]}` : "";
  return `${currency ? `${currency} ` : ""}${amount}${period}`;
}

/** "5 Nov 2026" (in UTC, so the day does not depend on where it is read), or null. */
export function postedDate(value: string | null): string | null {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}
