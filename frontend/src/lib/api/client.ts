// Server-side only: the browser never calls the API. `API_INTERNAL_URL` has no NEXT_PUBLIC_
// prefix, so Next.js does not put it into browser code.
import type { paths } from "./schema";

export type ApiPath = keyof paths;

/** The JSON body of a successful GET, taken from the generated schema. */
export type GetJson<P extends ApiPath> = paths[P] extends {
  get: { responses: { 200: { content: { "application/json": infer Body } } } };
}
  ? Body
  : never;

const TIMEOUT_MS = 10_000;

/** A failed call. The message is fixed: a response body can hold the user's own text. */
export class ApiError extends Error {
  constructor(
    readonly status: number, // 0 when no response arrived
    readonly code: string,
    readonly detail: unknown = null,
  ) {
    super(`API request failed: ${status} ${code}`);
    this.name = "ApiError";
  }
}

export interface ApiRequest {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  token?: string;
  body?: BodyInit;
  contentType?: string;
}

function apiUrl(path: string): URL {
  const base = process.env.API_INTERNAL_URL;
  if (!base) throw new Error("API_INTERNAL_URL is not set");
  const url = new URL(path, base);
  // Checked after parsing, so "..", "//other-host" and absolute URLs cannot leave the API.
  if (url.origin !== new URL(base).origin || !url.pathname.startsWith("/api/v1/")) {
    throw new Error("Refusing to call a path outside the API");
  }
  return url;
}

function headersFor({ token, contentType }: ApiRequest): Headers {
  const headers = new Headers({ Accept: "application/json" });
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (contentType) headers.set("Content-Type", contentType);
  return headers;
}

async function toError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { code?: unknown; detail?: unknown };
    const code = typeof body.code === "string" ? body.code : "unknown_error";
    return new ApiError(response.status, code, body.detail);
  } catch {
    return new ApiError(response.status, "unknown_error");
  }
}

/** One call to the API: never cached, never following a redirect, with a time limit. */
export async function apiFetch(path: string, request: ApiRequest = {}): Promise<Response> {
  const url = apiUrl(path);
  let response: Response;
  try {
    response = await fetch(url, {
      method: request.method ?? "GET",
      headers: headersFor(request),
      body: request.body,
      cache: "no-store", // one user's data must never be reused for another
      redirect: "error", // a redirect must not carry the token somewhere else
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch {
    throw new ApiError(0, "api_unreachable");
  }
  if (!response.ok) throw await toError(response);
  return response;
}

export async function apiGet<P extends ApiPath>(path: P, token?: string): Promise<GetJson<P>> {
  const response = await apiFetch(path, { token });
  return (await response.json()) as GetJson<P>;
}
