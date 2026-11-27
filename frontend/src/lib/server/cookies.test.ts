// @vitest-environment node
import { NextRequest, NextResponse } from "next/server";
import { describe, expect, it } from "vitest";

import { clearedCookies, cookieNames, readSession, sessionCookies } from "./cookies";

const tokens = {
  access_token: "access-1",
  refresh_token: "refresh-1",
  token_type: "bearer",
  expires_in: 1800,
};

/** The Set-Cookie headers Next.js really sends for these cookies. */
function headers(cookies: ReturnType<typeof sessionCookies>): string[] {
  const response = new NextResponse();
  for (const cookie of cookies) response.cookies.set(cookie);
  return response.headers.getSetCookie();
}

describe("session cookies", () => {
  it("are httpOnly, lax, for the whole site, and __Host- plus Secure under https", () => {
    const [access, refresh] = headers(sessionCookies(tokens, true));

    expect(access).toMatch(/^__Host-rjr_access=access-1;/);
    expect(refresh).toMatch(/^__Host-rjr_refresh=refresh-1;/);
    for (const header of [access!, refresh!]) {
      expect(header).toMatch(/; HttpOnly/i);
      expect(header).toMatch(/; Secure/i);
      expect(header).toMatch(/; SameSite=lax/i);
      expect(header).toMatch(/; Path=\/(;|$)/i);
      expect(header).not.toMatch(/Domain=/i);
    }
  });

  it("have no prefix and no Secure over plain http", () => {
    for (const header of headers(sessionCookies(tokens, false))) {
      expect(header).toMatch(/^rjr_(access|refresh)=/);
      expect(header).not.toMatch(/Secure|__Host-/i);
      expect(header).toMatch(/; HttpOnly/i);
    }
  });

  it("end the access cookie 30 seconds early, and keep the refresh cookie 7 days", () => {
    const [access, refresh] = sessionCookies(tokens, true);

    expect(access?.maxAge).toBe(1770);
    expect(refresh?.maxAge).toBe(604_800);
  });

  it("never give the access cookie less than 30 seconds", () => {
    expect(sessionCookies({ ...tokens, expires_in: 10 }, true)[0]?.maxAge).toBe(30);
  });

  it("are deleted with the same names and attributes", () => {
    const cleared = headers(clearedCookies(true));

    expect(cleared).toHaveLength(2);
    for (const header of cleared) {
      expect(header).toMatch(/^__Host-rjr_(access|refresh)=;/);
      expect(header).toMatch(/; Max-Age=0/i);
      expect(header).toMatch(/; Secure/i);
    }
  });
});

describe("readSession", () => {
  const request = (cookie: string) =>
    new NextRequest("http://127.0.0.1:3000/", { headers: { cookie } });

  it("reads the cookies of the current mode only", () => {
    const { access } = cookieNames(true);
    const cookies = request(`${access}=secure-one; rjr_refresh=plain-one`).cookies;

    expect(readSession(cookies, true)).toEqual({ access: "secure-one", refresh: undefined });
    expect(readSession(cookies, false)).toEqual({ access: undefined, refresh: "plain-one" });
  });

  it("treats an empty cookie as missing", () => {
    expect(readSession(request("rjr_access=").cookies, false).access).toBeUndefined();
  });
});
