import type { NextConfig } from "next";

// Sent with every response. The Content Security Policy needs a new nonce per request, so it is
// set in `src/proxy.ts` instead.
const SECURITY_HEADERS = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  // Not "no-referrer": with it a browser sends `Origin: null` on our own POSTs, and the CSRF
  // check would refuse them.
  { key: "Referrer-Policy", value: "same-origin" },
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false, // do not advertise the framework in every response
  headers: async () => [{ source: "/:path*", headers: SECURITY_HEADERS }],
};

export default nextConfig;
