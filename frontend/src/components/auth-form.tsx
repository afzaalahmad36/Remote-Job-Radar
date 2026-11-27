"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { errorCode, messageFor, UNAVAILABLE } from "@/lib/messages";

const TEXTS = {
  login: {
    title: "Sign in",
    submit: "Sign in",
    other: { href: "/register", text: "Create an account" },
  },
  register: {
    title: "Create an account",
    submit: "Create account",
    other: { href: "/login", text: "I already have an account" },
  },
} as const;

const FIELD = "mt-1 w-full rounded border border-slate-400 px-3 py-2";

export function AuthForm({ mode, notice }: { mode: keyof typeof TEXTS; notice?: string }) {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const texts = TEXTS[mode];

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.get("email"), password: form.get("password") }),
      });
      if (response.ok) {
        router.replace("/");
        return; // stay "busy" until the next page shows
      }
      setError(messageFor(await errorCode(response)));
    } catch {
      setError(UNAVAILABLE);
    }
    setBusy(false);
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-12">
      <h1 className="text-2xl font-semibold">{texts.title}</h1>
      {notice && (
        <p role="status" className="mt-3 rounded bg-amber-100 px-3 py-2">
          {notice}
        </p>
      )}
      {/* method="post": before the script has loaded, the browser must not put the password
          into the address bar. */}
      <form method="post" onSubmit={submit} className="mt-4 space-y-4">
        <label className="block">
          Email
          <input className={FIELD} type="email" name="email" autoComplete="email" required />
        </label>
        <label className="block">
          Password
          <input
            className={FIELD}
            type="password"
            name="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            minLength={mode === "register" ? 8 : undefined}
            maxLength={128}
            required
          />
        </label>
        {error && (
          <p role="alert" className="text-red-700">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded bg-slate-900 px-3 py-2 text-white disabled:opacity-60"
        >
          {texts.submit}
        </button>
      </form>
      <p className="mt-4">
        <Link className="underline" href={texts.other.href}>
          {texts.other.text}
        </Link>
      </p>
    </main>
  );
}
