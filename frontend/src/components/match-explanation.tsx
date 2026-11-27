"use client";

import { useRouter } from "next/navigation";
import { type ReactNode, useEffect, useId, useState } from "react";

import { callOurApi } from "@/lib/browser-api";
import { describeFilter, describeParts, type MatchDetail, type Reasons } from "@/lib/matches/explain";
import { messageFor } from "@/lib/messages";

type State = { detail: MatchDetail } | { problem: string } | null; // null: still loading

function Section({ title, lines }: { title: string; lines: string[] }) {
  if (lines.length === 0) return null;
  return (
    <>
      <h3 className="mt-3 font-semibold">{title}</h3>
      <ul className="mt-1 list-disc pl-5">
        {lines.map((line, index) => (
          <li key={`${index}-${line}`} className="break-words">
            {line}
          </li>
        ))}
      </ul>
    </>
  );
}

function Explanation({ detail }: { detail: MatchDetail }) {
  const reasons = detail.reasons as Reasons | null; // see `explain.ts`
  if (!reasons) {
    return <p>This explanation is outdated. It is rewritten at the next scoring run.</p>;
  }
  const skills = (names: string[]) => (names.length > 0 ? [names.join(", ")] : []);
  return (
    <>
      <Section title="How the score was made" lines={describeParts(reasons)} />
      <Section title="Skills you have" lines={skills(reasons.skills.matched)} />
      <Section title="Skills you are missing" lines={skills(reasons.skills.missing)} />
      <Section title="Your filters" lines={reasons.filters.map(describeFilter)} />
      <Section title="Notes" lines={detail.note_messages} />
    </>
  );
}

/** Loads when it appears: the list does not fetch the explanations of jobs nobody opens. */
function Loaded({ jobId }: { jobId: string }) {
  const router = useRouter();
  const [state, setState] = useState<State>(null);

  useEffect(() => {
    const controller = new AbortController();
    callOurApi<MatchDetail>(`/api/v1/matches/${jobId}`, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return;
        if (result.kind === "signed_out") router.replace(result.signIn);
        else if (result.kind === "ok") setState({ detail: result.data });
        else if (result.code === "not_found") {
          setState({ problem: "This job has no score any more. Reload the page." });
        } else setState({ problem: messageFor(result.code) });
      })
      .catch(() => undefined); // only an abort throws
    return () => controller.abort();
  }, [jobId, router]);

  if (!state) return <p role="status">Loading…</p>;
  if ("problem" in state) return <p role="alert">{state.problem}</p>;
  return <Explanation detail={state.detail} />;
}

export function WhyThisScore({ jobId }: { jobId: string }): ReactNode {
  const [open, setOpen] = useState(false);
  const panel = useId();
  return (
    <div className="mt-3">
      <button
        type="button"
        className="underline"
        aria-expanded={open}
        aria-controls={panel}
        onClick={() => setOpen(!open)}
      >
        Why this score?
      </button>
      <div id={panel} className="text-sm">
        {open && <Loaded jobId={jobId} />}
      </div>
    </div>
  );
}
