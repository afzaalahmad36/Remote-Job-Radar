// "(app)" is a route group: it shares a layout but adds nothing to the address, so this is "/".
export default function HomePage() {
  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-semibold">Your dashboard</h1>
      <p className="mt-3">Job matches arrive in a later step.</p>
    </main>
  );
}
