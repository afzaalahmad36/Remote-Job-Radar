// What the pages say for each code of our `/api` routes. Safe to use in the browser.

const MESSAGES: Record<string, string> = {
  invalid_credentials: "Wrong email or password.",
  email_taken: "An account with this email already exists. Sign in instead.",
  invalid_input: "Check the email address. The password needs 8 to 128 characters.",
  rate_limited: "Too many attempts. Wait a minute and try again.",
  forbidden_origin: "Open the site at its configured address and try again.",
  registered_not_signed_in: "Your account was created. Please sign in.",
  session_ended: "Your session ended. Please sign in again.",
  not_revoked:
    "You are signed out here, but the server could not be reached to end the session there.",
};

export const UNAVAILABLE = "The service is not available right now. Try again shortly.";

export function messageFor(code: unknown): string {
  // `hasOwn`, so a code such as "toString" cannot pick up something every object has.
  if (typeof code !== "string" || !Object.hasOwn(MESSAGES, code)) return UNAVAILABLE;
  return MESSAGES[code] ?? UNAVAILABLE;
}

/** The `code` of an error answer, or null if it has none. */
export async function errorCode(response: Response): Promise<string | null> {
  try {
    const { code } = (await response.json()) as { code?: unknown };
    return typeof code === "string" ? code : null;
  } catch {
    return null;
  }
}
