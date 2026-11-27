// Server-side only: used by the login and register routes.
import { ApiError, apiFetch } from "@/lib/api/client";

import type { Credentials } from "./body";
import type { Tokens } from "./cookies";

export function postCredentials(path: string, credentials: Credentials): Promise<Response> {
  return apiFetch(path, {
    method: "POST",
    contentType: "application/json",
    body: JSON.stringify(credentials),
  });
}

/** Throws ApiError, also when the answer cannot be read. */
export async function logIn(credentials: Credentials): Promise<Tokens> {
  const response = await postCredentials("/api/v1/auth/login", credentials);
  try {
    return (await response.json()) as Tokens;
  } catch {
    throw new ApiError(502, "bad_api_response");
  }
}
