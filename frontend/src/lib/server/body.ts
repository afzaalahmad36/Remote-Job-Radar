// Server-side only: reading what the browser sent, with a size limit.

export const MAX_BODY_BYTES = 16 * 1024;

export interface Credentials {
  email: string;
  password: string;
}

export type BodyResult =
  | { ok: true; credentials: Credentials }
  | { ok: false; status: number; code: string };

const refused = (status: number): BodyResult => ({ ok: false, status, code: "bad_request" });

/** The body as text, or null if it is longer than the limit. Counted while it arrives. */
async function readLimited(request: Request): Promise<string | null> {
  const chunks: Uint8Array[] = [];
  let size = 0;
  // A `Content-Length` header can be missing or wrong, so the bytes themselves are counted.
  const reader = request.body?.getReader();
  while (reader) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_BODY_BYTES) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks).toString("utf-8");
}

/** Email and password from a JSON body. The API checks their format and length. */
export async function readCredentials(request: Request): Promise<BodyResult> {
  // Also a second CSRF barrier: another site's plain form cannot send this content type.
  const type = request.headers.get("content-type")?.split(";")[0]?.trim().toLowerCase();
  if (type !== "application/json") return refused(415);
  const text = await readLimited(request);
  if (text === null) return refused(413);
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return refused(400);
  }
  const { email, password } = (body ?? {}) as Partial<Record<keyof Credentials, unknown>>;
  if (typeof email !== "string" || typeof password !== "string") return refused(400);
  return { ok: true, credentials: { email, password } }; // nothing else is passed on
}
