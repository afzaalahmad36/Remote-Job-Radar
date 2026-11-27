import Link from "next/link";
import { connection } from "next/server";

// Our own 404 page: the built-in one is made at build time, so its scripts carry no nonce and the
// security policy would block them. `connection()` waits for a real request before rendering.
export default async function NotFound() {
  await connection();
  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="mt-3">
        <Link className="underline" href="/">
          Back to the dashboard
        </Link>
      </p>
    </main>
  );
}
