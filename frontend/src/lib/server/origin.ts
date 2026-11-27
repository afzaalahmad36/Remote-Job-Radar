// Server-side only: the CSRF check. Cookies are sent along even when another site starts the
// request, but a browser also says where it came from, in `Origin`, which a page cannot forge.

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]); // these must never change anything

/** False for a request that changes data and does not come from our own pages. */
export function hasAllowedOrigin(request: Request, appOrigin: string): boolean {
  if (SAFE_METHODS.has(request.method)) return true;
  // A missing header is refused too, and so is the literal "null" some browsers send.
  return request.headers.get("origin") === appOrigin;
}
