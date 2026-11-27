// The one door from the browser to the API: only what the allow-list names, as the signed-in user.
import type { NextRequest, NextResponse } from "next/server";

import { apiFetch } from "@/lib/api/client";
import { forwardPath } from "@/lib/server/allowlist";
import { appOrigin, usesSecureCookies } from "@/lib/server/config";
import { clearedCookies, readSession, sessionCookies } from "@/lib/server/cookies";
import { hasAllowedOrigin } from "@/lib/server/origin";
import { apiFailure, failure, jsonText } from "@/lib/server/respond";
import { callAsUser } from "@/lib/server/session";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

// A request body is never passed on: none of the allowed calls takes one.
async function forward(
  method: "GET" | "POST",
  request: NextRequest,
  context: Context,
): Promise<NextResponse> {
  // Lets GET through; every other method must come from our own pages.
  if (!hasAllowedOrigin(request, appOrigin())) return failure(403, "forbidden_origin");
  const segments = (await context.params).path;
  const path = forwardPath(method, segments, request.nextUrl.searchParams);
  if (!path) return failure(404, "not_found");
  const secure = usesSecureCookies();
  const session = readSession(request.cookies, secure);
  if (!session.access && !session.refresh) return failure(401, "not_signed_in");

  const result = await callAsUser(session, (token) => apiFetch(path, { method, token }));
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

export const GET = (request: NextRequest, context: Context) => forward("GET", request, context);
export const POST = (request: NextRequest, context: Context) => forward("POST", request, context);
