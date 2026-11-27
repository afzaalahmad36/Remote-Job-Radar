// One ranked job. Everything from the post is shown as plain text (React escapes it).
import { WhyThisScore } from "@/components/match-explanation";
import { type Match, postedDate, REMOTE, salary, SENIORITY, warnings } from "@/lib/matches/format";
import { safeMailLink, safeWebLink } from "@/lib/matches/links";

// Colour only repeats what the number already says.
function badgeColour(score: number): string {
  if (score >= 70) return "bg-green-100 text-green-900";
  if (score >= 40) return "bg-amber-100 text-amber-900";
  return "bg-slate-100 text-slate-900";
}

function Skills({ stack, missing }: { stack: string[]; missing: string[] }) {
  if (stack.length === 0) return null;
  const lacking = new Set(missing);
  return (
    <ul aria-label="Skills the job lists" className="mt-3 flex flex-wrap gap-2 text-sm">
      {stack.map((name, index) => (
        <li
          key={`${index}-${name}`}
          className={`rounded border px-2 py-0.5 break-all ${
            lacking.has(name) ? "border-dashed border-slate-500" : "border-slate-300 bg-slate-50"
          }`}
        >
          {name}
          {lacking.has(name) && <span className="text-slate-600"> (missing)</span>}
        </li>
      ))}
    </ul>
  );
}

export function MatchCard({ match }: { match: Match }) {
  const { job } = match;
  const facts = [
    job.company,
    job.remote_policy && REMOTE[job.remote_policy],
    job.seniority && SENIORITY[job.seniority],
    salary(job),
    postedDate(job.posted_at) && `Posted ${postedDate(job.posted_at)}`,
  ].filter(Boolean);
  const links = [safeWebLink(job.apply_url), safeMailLink(job.apply_email)].filter(
    (link) => link !== null,
  );
  const notes = warnings(match);

  return (
    <article className="rounded border border-slate-300 p-4">
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-lg font-semibold break-words">{job.title ?? "Untitled job"}</h2>
        <p
          className={`shrink-0 rounded px-3 py-1 text-center ${badgeColour(match.score)}`}
        >
          <span className="block text-2xl font-semibold">{match.score}</span>
          <span className="block text-xs">match score</span>
        </p>
      </div>
      <p className="mt-1 break-words text-slate-700">{facts.join(" · ")}</p>
      {notes.length > 0 && (
        <ul aria-label="Warnings" className="mt-2 flex flex-wrap gap-2 text-sm">
          {notes.map((note) => (
            <li key={note} className="rounded bg-amber-100 px-2 py-0.5 text-amber-900">
              {note}
            </li>
          ))}
        </ul>
      )}
      <Skills stack={job.stack} missing={match.missing_skills} />
      {links.length > 0 && (
        <p className="mt-3 flex flex-wrap gap-4">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="underline break-all"
            >
              {link.label}
            </a>
          ))}
        </p>
      )}
      <WhyThisScore jobId={job.id} />
    </article>
  );
}
