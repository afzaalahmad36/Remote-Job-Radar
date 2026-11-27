// Server-side only: every answer of our own `/api` routes is built here.
import { NextResponse } from "next/server";

import type { ApiError } from "@/lib/api/client";

import type { SessionCookie } from "./cookies";

function withCookies(response: NextResponse, cookies: readonly SessionCookie[]): NextResponse {
  response.headers.set("Cache-Control", "no-store"); // one user's data: never kept anywhere
  for (const cookie of cookies) response.cookies.set(cookie);
  return response;
}

/** A success without a body. Tokens travel in cookies only, never in a body. */
export function done(cookies: readonly SessionCookie[] = []): NextResponse {
  return withCookies(new NextResponse(null, { status: 204 }), cookies);
}

/** A JSON text from the API, passed on as it is. */
export function jsonText(
  status: number,
  text: string,
  cookies: readonly SessionCookie[] = [],
): NextResponse {
  const headers = { "Content-Type": "application/json" };
  return withCookies(new NextResponse(text, { status, headers }), cookies);
}

/** `{ code }` only: the page picks its own words, and nothing the API wrote reaches it. */
export function failure(
  status: number,
  code: string,
  cookies: readonly SessionCookie[] = [],
): NextResponse {
  return withCookies(NextResponse.json({ code }, { status }), cookies);
}

const PASSED_ON: Record<number, string> = {
  404: "not_found",
  409: "conflict",
  422: "invalid_input",
  429: "rate_limited",
};

/** An API error as our own: a few statuses are passed on, everything else is "unavailable". */
export function apiFailure(error: ApiError, cookies: readonly SessionCookie[] = []): NextResponse {
  const code = PASSED_ON[error.status];
  return code ? failure(error.status, code, cookies) : failure(502, "api_unavailable", cookies);
}
