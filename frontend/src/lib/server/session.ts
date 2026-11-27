// Server-side only: one API call made as the signed-in user.
import { ApiError } from "@/lib/api/client";

import type { Tokens } from "./cookies";
import { refreshTokens } from "./refresh";

/**
 * What the route handler has to do with the answer.
 *
 * `tokens` is set when the session was refreshed on the way. The handler MUST write them to the
 * cookies even when the call itself failed: the old refresh token is used up.
 */
export type SessionResult =
  | { kind: "response"; response: Response; tokens?: Tokens }
  | { kind: "error"; error: ApiError; tokens?: Tokens }
  | { kind: "ended" }; // clear the cookies; the user has to sign in again

interface Session {
  access: string | undefined;
  refresh: string | undefined;
}

// 401: used, expired or unknown. 422: the cookie holds something that is no token at all.
const DEAD_REFRESH = new Set([401, 422]);

/**
 * Run `call` with an access token, refreshing first if there is none, and once more if the API
 * answers 401. `call` may run twice, so it must build its request afresh each time. Retrying is
 * safe: on a 401 the API refused the request before doing anything.
 */
export async function callAsUser(
  session: Session,
  call: (accessToken: string) => Promise<Response>,
): Promise<SessionResult> {
  let access = session.access;
  let tokens: Tokens | undefined; // set once we have refreshed

  // At most two rounds: the second only after a 401 for a token we had NOT just refreshed.
  for (;;) {
    if (!access) {
      if (!session.refresh) return { kind: "ended" };
      try {
        tokens = await refreshTokens(session.refresh);
      } catch (error) {
        if (!(error instanceof ApiError)) throw error;
        if (DEAD_REFRESH.has(error.status)) return { kind: "ended" };
        return { kind: "error", error }; // 429, 5xx, timeout: keep the cookies, try later
      }
      access = tokens.access_token;
    }
    try {
      return { kind: "response", response: await call(access), tokens };
    } catch (error) {
      if (!(error instanceof ApiError)) throw error;
      if (error.status !== 401) return { kind: "error", error, tokens };
      if (tokens) return { kind: "ended" }; // even a brand-new token is refused
      access = undefined; // refresh and try once more
    }
  }
}
