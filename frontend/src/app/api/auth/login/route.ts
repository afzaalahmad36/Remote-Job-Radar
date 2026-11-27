import type { NextRequest, NextResponse } from "next/server";

import { ApiError } from "@/lib/api/client";
import { readCredentials } from "@/lib/server/body";
import { appOrigin, usesSecureCookies } from "@/lib/server/config";
import { sessionCookies } from "@/lib/server/cookies";
import { hasAllowedOrigin } from "@/lib/server/origin";
import { apiFailure, done, failure } from "@/lib/server/respond";
import { logIn } from "@/lib/server/sign-in";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!hasAllowedOrigin(request, appOrigin())) return failure(403, "forbidden_origin");
  const body = await readCredentials(request);
  if (!body.ok) return failure(body.status, body.code);
  try {
    return done(sessionCookies(await logIn(body.credentials), usesSecureCookies()));
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
    // Here a 401 means a wrong email or password, not a lost session.
    return error.status === 401 ? failure(401, "invalid_credentials") : apiFailure(error);
  }
}
