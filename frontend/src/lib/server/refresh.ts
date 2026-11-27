// Server-side only, and the ONLY code that calls `/auth/refresh`.
//
// The API treats a second use of a refresh token as theft and ends the whole session. A page often
// sends several requests at once, all carrying the same old cookie, so they must share one call.
import { createHash } from "node:crypto";

import { ApiError, apiFetch } from "@/lib/api/client";

import type { Tokens } from "./cookies";

// How long a finished refresh still answers for the old token: for a request that was already on
// its way, or a response the browser never received. The cost: for this long, someone who stole
// the old token would get the new pair from us without the API noticing.
export const KEEP_MS = 30_000;
export const MAX_ENTRIES = 1_000;

interface Entry {
  result: Promise<Tokens>;
  expiresAt: number; // Infinity while the call is still running
  nextKey?: string; // the key of the new refresh token, so that sign-out can find this entry
}

// On `globalThis`, not in a module variable: the build may load this module more than once (one
// copy per route), and each copy must see the same map. It is still one map per server process.
const globalStore = globalThis as typeof globalThis & { __rjrRefresh?: Map<string, Entry> };
const entries = (globalStore.__rjrRefresh ??= new Map<string, Entry>());

/** The raw token is never a key, so it is not sitting in memory longer than needed. */
const keyOf = (refreshToken: string) => createHash("sha256").update(refreshToken).digest("hex");

function makeRoom(): void {
  for (const [key, entry] of entries) {
    if (entry.expiresAt <= Date.now()) entries.delete(key);
  }
  if (entries.size < MAX_ENTRIES) return;
  // Drop the oldest FINISHED entry. Dropping a running one could cause a second call for its token.
  for (const [key, entry] of entries) {
    if (entry.expiresAt !== Infinity) {
      entries.delete(key);
      return;
    }
  }
  throw new ApiError(503, "web_busy");
}

function isTokens(body: unknown): body is Tokens {
  const tokens = body as Partial<Tokens> | null;
  return (
    typeof tokens?.access_token === "string" &&
    typeof tokens.refresh_token === "string" &&
    typeof tokens.expires_in === "number"
  );
}

async function callRefresh(refreshToken: string): Promise<Tokens> {
  let body: unknown;
  try {
    const response = await apiFetch("/api/v1/auth/refresh", {
      method: "POST",
      contentType: "application/json",
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    body = await response.json();
  } catch (error) {
    // A new error without `detail`: nothing the API sent travels on.
    if (error instanceof ApiError) throw new ApiError(error.status, error.code);
    throw new ApiError(502, "bad_api_response");
  }
  if (!isTokens(body)) throw new ApiError(502, "bad_api_response");
  return body;
}

/**
 * The new token pair for this refresh token. Callers with the same token share one API call and
 * get the same pair. Throws ApiError; a failure is not remembered, so the next request tries again.
 */
export function refreshTokens(refreshToken: string): Promise<Tokens> {
  const key = keyOf(refreshToken);
  const existing = entries.get(key);
  if (existing && existing.expiresAt > Date.now()) return existing.result;

  try {
    makeRoom();
  } catch (error) {
    return Promise.reject(error);
  }
  // No `await` between the lookup above and `set` below: that is what makes this single-flight.
  // JavaScript runs this stretch without another request getting in between.
  const entry: Entry = { result: callRefresh(refreshToken), expiresAt: Infinity };
  entries.set(key, entry);
  entry.result.then(
    (tokens) => {
      entry.expiresAt = Date.now() + KEEP_MS;
      entry.nextKey = keyOf(tokens.refresh_token);
    },
    () => {
      if (entries.get(key) === entry) entries.delete(key);
    },
  );
  return entry.result;
}

/** For sign-out: drop every entry for this token, and every one that leads to it. */
export function forgetRefresh(refreshToken: string): void {
  forgetKey(keyOf(refreshToken));
}

function forgetKey(key: string): void {
  entries.delete(key);
  for (const [olderKey, entry] of entries) {
    if (entry.nextKey === key) forgetKey(olderKey);
  }
}
