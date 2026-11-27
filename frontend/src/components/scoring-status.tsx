"use client";

// Says when the scores on screen are not the latest ones, and waits for the new ones.
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { callOurApi } from "@/lib/browser-api";
import type { MatchPage } from "@/lib/matches/format";
import { messageFor } from "@/lib/messages";

type Scoring = MatchPage["scoring"];

// How long to wait before each look: every 3 s at first, then every 10 s, two minutes in all.
export const POLL_DELAYS_MS = [
  ...Array<number>(10).fill(3_000),
  ...Array<number>(9).fill(10_000),
];

const waiting = (state: Scoring["state"]) => state === "pending" || state === "running";

const GAVE_UP = {
  slow: "Scores are still being computed. This is taking longer than usual.",
  lost: "Could not check whether the new scores are ready.",
};

/**
 * Look at `/matches/status` until scoring is over, then call `onChange` so the list is read again.
 * `scoring` is the object of the last list answer: a new one starts the waiting afresh.
 */
function Updating({ scoring, onChange }: { scoring: Scoring; onChange: () => void }) {
  const router = useRouter();
  // Which wait was given up, and why. Kept with its `scoring` and `round`, so that a newer one
  // starts clean without having to reset anything.
  const [gaveUp, setGaveUp] = useState<{ for: Scoring; round: number; why: "slow" | "lost" }>();
  const [round, setRound] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function look(step: number) {
      const result = await callOurApi<Scoring>("/api/v1/matches/status", {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (result.kind === "signed_out") router.replace(result.signIn);
      else if (result.kind === "error") setGaveUp({ for: scoring, round, why: "lost" });
      else if (!waiting(result.data.state)) onChange();
      else schedule(step + 1);
    }
    function schedule(step: number) {
      const delay = POLL_DELAYS_MS[step];
      if (delay === undefined) setGaveUp({ for: scoring, round, why: "slow" });
      else timer = setTimeout(() => void look(step).catch(() => undefined), delay);
    }

    schedule(0);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [scoring, round, router, onChange]);

  const stopped = gaveUp && gaveUp.for === scoring && gaveUp.round === round ? gaveUp : null;
  if (!stopped) {
    return <p role="status">Scores are updating. Any shown below are from before.</p>;
  }
  return (
    <p role="status">
      {GAVE_UP[stopped.why]}{" "}
      <button
        type="button"
        className="underline"
        onClick={() => {
          setRound((n) => n + 1);
          onChange();
        }}
      >
        Check again
      </button>
    </p>
  );
}

function Failed({ onChange }: { onChange: () => void }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");

  async function rescore() {
    setBusy(true);
    setProblem("");
    const result = await callOurApi("/api/v1/matches/rescore", { method: "POST" });
    if (result.kind === "signed_out") router.replace(result.signIn);
    else if (result.kind === "ok") onChange();
    else setProblem(messageFor(result.code));
    setBusy(false);
  }

  return (
    <div role="alert">
      <p>
        Scoring failed. Any scores below are from before.{" "}
        <button type="button" className="underline" disabled={busy} onClick={rescore}>
          Try again
        </button>
      </p>
      {problem && <p>{problem}</p>}
    </div>
  );
}

/** Nothing to say when the scores are up to date (or there is no CV: the empty list says it). */
export function ScoringStatus({ scoring, onChange }: { scoring: Scoring; onChange: () => void }) {
  if (scoring.state === "failed") {
    return (
      <div className="mb-4 rounded border border-amber-300 bg-amber-50 px-3 py-2">
        <Failed onChange={onChange} />
      </div>
    );
  }
  if (!waiting(scoring.state)) return null;
  return (
    <div className="mb-4 rounded border border-slate-300 bg-slate-50 px-3 py-2">
      <Updating scoring={scoring} onChange={onChange} />
    </div>
  );
}
