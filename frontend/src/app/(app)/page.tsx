// "(app)" is a route group: it shares a layout but adds nothing to the address, so this is "/".
import { MatchFilters } from "@/components/match-filters";
import { MatchList } from "@/components/match-list";
import { filtersAddress, parseFilters } from "@/lib/matches/filters";

export default async function MatchesPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const filters = parseFilters(await searchParams);
  // A new key for new filters: both components start afresh with them.
  const key = filtersAddress(filters);
  return (
    <main className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="text-2xl font-semibold">Your matches</h1>
      <p className="mt-2 text-sm text-slate-700">
        The score ranks jobs against your CV. It is not a probability of getting the job.
      </p>
      <MatchFilters key={`filters ${key}`} filters={filters} />
      <div className="mt-6">
        <MatchList key={`list ${key}`} filters={filters} />
      </div>
    </main>
  );
}
