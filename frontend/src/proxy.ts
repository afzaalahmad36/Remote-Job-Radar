// Next.js runs this before a page is rendered (it was called "middleware" before Next.js 16).
// It only sets the Content Security Policy. It does no authentication: the `/api` routes do.
import { type NextRequest, NextResponse } from "next/server";

import { usesSecureCookies } from "@/lib/server/config";

/**
 * Which scripts and other things a page may load. A script runs only if it carries this request's
 * random nonce, so a script injected into the page does not run. Next.js reads the nonce from the
 * request header and puts it on its own scripts.
 */
export function contentSecurityPolicy(nonce: string, secure: boolean, dev: boolean): string {
  const policy = [
    "default-src 'self'",
    // `unsafe-eval` and inline styles are for the development server only.
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${dev ? " 'unsafe-eval'" : ""}`,
    `style-src 'self' ${dev ? "'unsafe-inline'" : `'nonce-${nonce}'`}`,
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ];
  if (secure) policy.push("upgrade-insecure-requests");
  return policy.join("; ");
}

export function proxy(request: NextRequest): NextResponse {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const dev = process.env.NODE_ENV === "development";
  const policy = contentSecurityPolicy(nonce, usesSecureCookies(), dev);

  const headers = new Headers(request.headers);
  headers.set("Content-Security-Policy", policy); // where Next.js looks for the nonce
  const response = NextResponse.next({ request: { headers } });
  response.headers.set("Content-Security-Policy", policy); // what the browser enforces
  return response;
}

export const config = {
  // Pages only: not our JSON routes, not static files, and not background prefetches.
  matcher: [
    {
      source: "/((?!api|_next/static|_next/image|favicon.ico).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};
