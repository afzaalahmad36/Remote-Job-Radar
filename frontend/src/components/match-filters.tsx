"use client";

import { useRouter } from "next/navigation";
import type { FormEvent } from "react";

import { type Filters, filtersAddress, parseFilters } from "@/lib/matches/filters";

/** Applying puts the filters into the address; the page then reads them from there. */
export function MatchFilters({ filters }: { filters: Filters }) {
  const router = useRouter();

  function apply(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const next = parseFilters({
      min: String(form.get("min") ?? ""),
      all: form.get("all") ? "1" : undefined,
    });
    router.push(filtersAddress(next)); // back on page 1
  }

  return (
    <form onSubmit={apply} className="mt-4 flex flex-wrap items-end gap-4">
      <label className="block text-sm">
        Minimum score
        <input
          type="number"
          name="min"
          min={0}
          max={100}
          step={1}
          defaultValue={filters.min ?? ""}
          className="mt-1 block w-28 rounded border border-slate-400 px-2 py-1"
        />
      </label>
      <label className="flex items-center gap-2 py-1 text-sm">
        <input type="checkbox" name="all" defaultChecked={filters.all} />
        Show jobs that fail my filters
      </label>
      <button type="submit" className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white">
        Apply
      </button>
    </form>
  );
}
