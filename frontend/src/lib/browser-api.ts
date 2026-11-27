// Browser side: one call to our own `/api` routes. Every signed-in page goes through here, so a
// lost session is recognised the same way everywhere.
import { errorCode } from "@/lib/messages";

export type ApiResult<T> =
  | { kind: "ok"; data: T }
  | { kind: "signed_out"; signIn: string } // where to send the user
  | { kind: "error"; code: string | null }; // null: no answer, or one we cannot read

/**
 * `data` is typed by the caller, from `openapi.json`. Never throws, except when `signal` aborts:
 * the page that asked is gone, and there is nothing to show.
 */
export async function callOurApi<T>(
  path: `/api/v1/${string}`,
  init: { method?: "GET" | "POST"; signal?: AbortSignal } = {},
): Promise<ApiResult<T>> {
  try {
    const response = await fetch(path, { ...init, cache: "no-store" });
    if (response.ok) return { kind: "ok", data: (await response.json()) as T };
    const code = await errorCode(response);
    if (response.status !== 401) return { kind: "error", code };
    const signIn = code === "session_ended" ? "/login?notice=session_ended" : "/login";
    return { kind: "signed_out", signIn };
  } catch (error) {
    if (init.signal?.aborted) throw error;
    return { kind: "error", code: null };
  }
}
