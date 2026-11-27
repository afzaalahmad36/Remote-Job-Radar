// The one door from the browser to the API: only what the allow-list names, as the signed-in user.
import type { NextRequest, NextResponse } from "next/server";

import { apiFetch } from "@/lib/api/client";
import { forwardPath } from "@/lib/server/allowlist";
import { usesSecureCookies } from "@/lib/server/config";
import { clearedCookies, readSession, sessionCookies } from "@/lib/server/cookies";
import { apiFailure, failure, jsonText } from "@/lib/server/respond";
import { callAsUser } from "@/lib/server/session";

export const dynamic = "force-dynamic";

// Only GET for now. A method that changes data must also check the Origin (`hasAllowedOrigin`).
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<NextResponse> {
  const path = forwardPath("GET", (await context.params).path);
  if (!path) return failure(404, "not_found");
  const secure = usesSecureCookies();
  const session = readSession(request.cookies, secure);
  if (!session.access && !session.refresh) return failure(401, "not_signed_in");

  const result = await callAsUser(session, (token) => apiFetch(path, { token }));
  if (result.kind === "ended") return failure(401, "session_ended", clearedCookies(secure));
  // Written even on failure: after a refresh the old token is used up.
  const cookies = result.tokens ? sessionCookies(result.tokens, secure) : [];
  if (result.kind === "error") return apiFailure(result.error, cookies);
  try {
    return jsonText(result.response.status, await result.response.text(), cookies);
  } catch {
    return failure(502, "api_unavailable", cookies); // the answer broke off
  }
}
