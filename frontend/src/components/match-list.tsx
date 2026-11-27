"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { MatchCard } from "@/components/match-card";
import { callOurApi } from "@/lib/browser-api";
import { type Filters, filtersAddress, matchesPath, PAGE_SIZE } from "@/lib/matches/filters";
import type { MatchPage } from "@/lib/matches/format";
import { messageFor } from "@/lib/messages";

type State = { page: MatchPage } | { problem: string } | null; // null: still loading

/** Why the list is empty, and what to do about it. */
function Empty({ page, filters }: { page: MatchPage; filters: Filters }) {
  if (page.scoring.state === "no_cv") {
    return (
      <p>
        Upload a CV to see matches. The CV page arrives in a later step; until then, use the API
        (see <code>docs/API_EXAMPLES.md</code>).
      </p>
    );
  }
  const narrowed = filters.min !== null || filters.page > 1;
  return (
    <p>
      No matches to show.{" "}
      {!filters.all && "Jobs that fail your preferences are hidden. "}
      {narrowed && (
        <Link className="underline" href="/">
          Clear the filters
        </Link>
      )}
    </p>
  );
}

function Pages({ filters, total }: { filters: Filters; total: number }) {
  const last = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (last === 1 && filters.page === 1) return null;
  const to = (page: number) => filtersAddress({ ...filters, page });
  return (
    <nav aria-label="Pages" className="mt-6 flex items-center gap-4">
      {filters.page > 1 && (
        <Link className="underline" href={to(filters.page - 1)}>
          Previous
        </Link>
      )}
      <span>
        Page {filters.page} of {last}
      </span>
      {filters.page < last && (
        <Link className="underline" href={to(filters.page + 1)}>
          Next
        </Link>
      )}
    </nav>
  );
}

/** Give it a `key` that changes with the filters: it then starts again from "loading". */
export function MatchList({ filters }: { filters: Filters }) {
  const router = useRouter();
  const [state, setState] = useState<State>(null);
  const [attempt, setAttempt] = useState(0);
  const path = matchesPath(filters);

  useEffect(() => {
    const controller = new AbortController(); // cancels the request if we leave the page
    callOurApi<MatchPage>(path, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return;
        if (result.kind === "signed_out") router.replace(result.signIn);
        else if (result.kind === "ok") setState({ page: result.data });
        else setState({ problem: messageFor(result.code) });
      })
      .catch(() => undefined); // only an abort throws
    return () => controller.abort();
  }, [path, router, attempt]);

  if (!state) return <p role="status">Loading matches…</p>;
  if ("problem" in state) {
    return (
      <>
        <p role="alert">{state.problem}</p>
        <button
          type="button"
          className="mt-3 underline"
          onClick={() => {
            setState(null);
            setAttempt((n) => n + 1);
          }}
        >
          Try again
        </button>
      </>
    );
  }

  const { page } = state;
  return (
    <>
      {page.items.length === 0 ? (
        <Empty page={page} filters={filters} />
      ) : (
        <>
          <p role="status" className="text-sm text-slate-600">
            {page.total} {page.total === 1 ? "match" : "matches"}
          </p>
          <ul className="mt-3 space-y-4">
            {page.items.map((match) => (
              <li key={match.job.id}>
                <MatchCard match={match} />
              </li>
            ))}
          </ul>
        </>
      )}
      <Pages filters={filters} total={page.total} />
    </>
  );
}
