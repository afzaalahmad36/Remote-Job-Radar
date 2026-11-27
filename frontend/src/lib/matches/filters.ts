// The list's filters live in the address (`/?min=60&all=1&page=2`), so reload and back work.

export const PAGE_SIZE = 20;
const MAX_PAGE = 5_000; // the API refuses an offset above 100,000

export interface Filters {
  min: number | null; // minimum score
  all: boolean; // also show jobs that fail a hard filter
  page: number; // from 1
}

type SearchParams = Record<string, string | string[] | undefined>;

/** A whole number within the range, or null. The address can say anything. */
function whole(value: unknown, low: number, high: number): number | null {
  if (typeof value !== "string" || !/^\d{1,6}$/.test(value)) return null;
  const number = Number(value);
  return number >= low && number <= high ? number : null;
}

export function parseFilters(params: SearchParams): Filters {
  return {
    min: whole(params.min, 0, 100),
    all: params.all === "1",
    page: whole(params.page, 1, MAX_PAGE) ?? 1,
  };
}

/** The page address for these filters. Defaults are left out. */
export function filtersAddress({ min, all, page }: Filters): string {
  const query = new URLSearchParams();
  if (min !== null) query.set("min", String(min));
  if (all) query.set("all", "1");
  if (page > 1) query.set("page", String(page));
  const text = query.toString();
  return text ? `/?${text}` : "/";
}

/** The same filters in the words of `GET /matches`. */
export function matchesPath({ min, all, page }: Filters): `/api/v1/${string}` {
  const query = new URLSearchParams({
    limit: String(PAGE_SIZE),
    offset: String((page - 1) * PAGE_SIZE),
  });
  if (min !== null) query.set("min_score", String(min));
  if (all) query.set("include_filtered", "true");
  return `/api/v1/matches?${query}`;
}
