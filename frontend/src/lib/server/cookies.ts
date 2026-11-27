// Server-side only: the two session cookies. Page scripts can never read them (httpOnly).
import type { components } from "@/lib/api/schema";

export type Tokens = components["schemas"]["TokenOut"];

// The API does not say when a refresh token expires. 7 days is its default; if it is shorter
// there, the API answers 401 and the cookies are cleared anyway.
const REFRESH_MAX_AGE_SECONDS = 7 * 24 * 60 * 60;
// The access cookie disappears this long before the token expires, so "no access cookie" can
// simply mean "refresh first".
const ACCESS_MARGIN_SECONDS = 30;

export interface SessionCookie {
  name: string;
  value: string;
  httpOnly: true;
  sameSite: "lax";
  secure: boolean;
  path: "/";
  maxAge: number;
}

/** The `__Host-` prefix makes the browser insist on Secure, Path=/ and no Domain. */
export function cookieNames(secure: boolean): { access: string; refresh: string } {
  const prefix = secure ? "__Host-" : "";
  return { access: `${prefix}rjr_access`, refresh: `${prefix}rjr_refresh` };
}

function cookie(name: string, value: string, maxAge: number, secure: boolean): SessionCookie {
  return { name, value, httpOnly: true, sameSite: "lax", secure, path: "/", maxAge };
}

export function sessionCookies(tokens: Tokens, secure: boolean): SessionCookie[] {
  const names = cookieNames(secure);
  const accessAge = Math.max(tokens.expires_in - ACCESS_MARGIN_SECONDS, ACCESS_MARGIN_SECONDS);
  return [
    cookie(names.access, tokens.access_token, accessAge, secure),
    cookie(names.refresh, tokens.refresh_token, REFRESH_MAX_AGE_SECONDS, secure),
  ];
}

/** Max-Age 0 deletes. The other attributes stay: a `__Host-` cookie needs them even for this. */
export function clearedCookies(secure: boolean): SessionCookie[] {
  const names = cookieNames(secure);
  return [cookie(names.access, "", 0, secure), cookie(names.refresh, "", 0, secure)];
}

/** `request.cookies` of a Next.js request fits this shape. */
interface CookieReader {
  get(name: string): { value: string } | undefined;
}

export function readSession(
  cookies: CookieReader,
  secure: boolean,
): { access: string | undefined; refresh: string | undefined } {
  const names = cookieNames(secure);
  // An empty value counts as no cookie.
  return {
    access: cookies.get(names.access)?.value || undefined,
    refresh: cookies.get(names.refresh)?.value || undefined,
  };
}
