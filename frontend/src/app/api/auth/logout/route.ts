import type { NextRequest, NextResponse } from "next/server";

import { ApiError, apiFetch } from "@/lib/api/client";
import { appOrigin, usesSecureCookies } from "@/lib/server/config";
import { clearedCookies, readSession } from "@/lib/server/cookies";
import { hasAllowedOrigin } from "@/lib/server/origin";
import { forgetRefresh } from "@/lib/server/refresh";
import { done, failure } from "@/lib/server/respond";

export const dynamic = "force-dynamic";

/** The cookies are cleared in every case; the answer says whether the API revoked the session. */
export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!hasAllowedOrigin(request, appOrigin())) return failure(403, "forbidden_origin");
  const secure = usesSecureCookies();
  const { refresh } = readSession(request.cookies, secure);
  if (!refresh) return done(clearedCookies(secure));
  forgetRefresh(refresh); // so the map cannot hand out a revoked pair
  try {
    await apiFetch("/api/v1/auth/logout", {
      method: "POST",
      contentType: "application/json",
      body: JSON.stringify({ refresh_token: refresh }),
    });
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    return failure(502, "not_revoked", clearedCookies(secure));
  }
  return done(clearedCookies(secure));
}
