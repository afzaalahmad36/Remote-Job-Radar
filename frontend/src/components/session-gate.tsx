"use client";

// Everything signed-in sits inside this. It is not what protects the data (the `/api` routes do):
// it only sends a visitor without a session to the sign-in page.
import { useRouter } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import type { components } from "@/lib/api/schema";
import { errorCode, messageFor, UNAVAILABLE } from "@/lib/messages";

type User = components["schemas"]["UserRead"];
type State = { user: User } | { problem: string } | null; // null: still loading

const SIGN_OUT_FAILED = "Signing out did not work. Try again.";

async function loadUser(): Promise<{ user: User } | { problem: string } | { signIn: string }> {
  try {
    const response = await fetch("/api/v1/users/me", { cache: "no-store" });
    if (response.ok) return { user: (await response.json()) as User };
    const code = await errorCode(response);
    if (response.status === 401) {
      return { signIn: code === "session_ended" ? "/login?notice=session_ended" : "/login" };
    }
    return { problem: messageFor(code) };
  } catch {
    return { problem: UNAVAILABLE };
  }
}

/** Where to go after signing out, or null if it did not happen. */
async function requestSignOut(): Promise<string | null> {
  try {
    const response = await fetch("/api/auth/logout", { method: "POST" });
    if (response.ok) return "/login";
    // "not_revoked": the cookies are gone, only the server-side session could not be ended.
    return (await errorCode(response)) === "not_revoked" ? "/login?notice=not_revoked" : null;
  } catch {
    return null;
  }
}

function Waiting({ problem, onRetry }: { problem?: string; onRetry: () => void }) {
  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-semibold">Remote Job Radar</h1>
      {!problem ? (
        <p role="status" className="mt-3">
          Loading…
        </p>
      ) : (
        <>
          <p role="alert" className="mt-3">
            {problem}
          </p>
          <button type="button" onClick={onRetry} className="mt-3 underline">
            Try again
          </button>
        </>
      )}
    </main>
  );
}

export function SessionGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [state, setState] = useState<State>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let stale = false; // the component went away, or a newer attempt started
    void loadUser().then((result) => {
      if (stale) return;
      if ("signIn" in result) router.replace(result.signIn);
      else setState(result);
    });
    return () => {
      stale = true;
    };
  }, [router, attempt]);

  if (!state || "problem" in state) {
    const retry = () => {
      setState(null);
      setAttempt((n) => n + 1);
    };
    return <Waiting problem={state?.problem} onRetry={retry} />;
  }

  async function signOut() {
    const next = await requestSignOut();
    if (next) router.replace(next);
    else setState({ problem: SIGN_OUT_FAILED });
  }

  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-3">
        <span className="font-semibold">Remote Job Radar</span>
        <span className="flex items-center gap-3">
          <span className="break-all">{state.user.email}</span>
          <button type="button" onClick={signOut} className="underline">
            Sign out
          </button>
        </span>
      </header>
      {children}
    </>
  );
}
