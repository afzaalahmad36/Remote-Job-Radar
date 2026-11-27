import type { NextRequest, NextResponse } from "next/server";

import { ApiError } from "@/lib/api/client";
import { readCredentials } from "@/lib/server/body";
import { appOrigin, usesSecureCookies } from "@/lib/server/config";
import { sessionCookies } from "@/lib/server/cookies";
import { hasAllowedOrigin } from "@/lib/server/origin";
import { apiFailure, done, failure } from "@/lib/server/respond";
import { logIn, postCredentials } from "@/lib/server/sign-in";

export const dynamic = "force-dynamic";

/** Create the account, then sign in with it. */
export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!hasAllowedOrigin(request, appOrigin())) return failure(403, "forbidden_origin");
  const body = await readCredentials(request);
  if (!body.ok) return failure(body.status, body.code);
  try {
    await postCredentials("/api/v1/auth/register", body.credentials);
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    return error.status === 409 ? failure(409, "email_taken") : apiFailure(error);
  }
  try {
    return done(sessionCookies(await logIn(body.credentials), usesSecureCookies()));
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    // The account exists now, so "try registering again" would be the wrong advice.
    return failure(502, "registered_not_signed_in");
  }
}
