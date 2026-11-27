// Server-side only: which API calls the browser may make through us. Everything else is refused.
import type { ApiPath } from "@/lib/api/client";

// The auth endpoints can never be listed: their answers hold tokens, which must not reach the page.
type ForwardablePath = Exclude<ApiPath, `/api/v1/auth/${string}`>;

export interface ForwardRule {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: ForwardablePath; // typed from openapi.json, so a rule for a missing endpoint fails to build
  query?: readonly string[]; // the query parameters passed on; any other one is dropped
}

export const FORWARD_RULES: readonly ForwardRule[] = [
  { method: "GET", path: "/api/v1/users/me" },
  {
    method: "GET",
    path: "/api/v1/matches",
    query: ["limit", "offset", "min_score", "include_filtered"],
  },
  { method: "GET", path: "/api/v1/matches/status" },
  { method: "GET", path: "/api/v1/matches/{job_id}" },
  { method: "POST", path: "/api/v1/matches/rescore" },
];

// Every path parameter of this API is a UUID.
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const PARAMETER = /^\{\w+\}$/; // such as {job_id}

function matches(pattern: string, path: string): boolean {
  const expected = pattern.split("/");
  const actual = path.split("/");
  return (
    expected.length === actual.length &&
    expected.every((part, i) =>
      PARAMETER.test(part) ? UUID.test(actual[i]!) : part === actual[i],
    )
  );
}

/** Only the listed names, each once. `URLSearchParams` encodes the values again. */
function allowedQuery(search: URLSearchParams, names: readonly string[]): string {
  const kept = new URLSearchParams();
  for (const name of names) {
    const value = search.get(name);
    if (value !== null) kept.set(name, value); // the API checks what the value says
  }
  const text = kept.toString();
  return text ? `?${text}` : "";
}

/**
 * The API path to call for these URL segments (what follows `/api/v1/`) and this query, or null if
 * not allowed.
 *
 * A fixed segment must be equal and a parameter must be a UUID, so "..", encoded slashes and
 * other tricks never match.
 */
export function forwardPath(
  method: string,
  segments: readonly string[],
  search: URLSearchParams = new URLSearchParams(),
  rules: readonly ForwardRule[] = FORWARD_RULES,
): string | null {
  // A "/" inside one segment would change how many parts the joined path has.
  if (segments.some((segment) => segment.includes("/"))) return null;
  const path = `/api/v1/${segments.join("/")}`;
  const rule = rules.find((r) => r.method === method && matches(r.path, path));
  return rule ? path + allowedQuery(search, rule.query ?? []) : null;
}
