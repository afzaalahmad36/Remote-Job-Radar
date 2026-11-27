# Web dashboard (Phase 6)

Next.js (App Router), TypeScript and Tailwind. So far: register, sign in, sign out and the list of
ranked matches. The browser never calls the API; the Next.js server does.

Needs Node 24 (`.nvmrc`). Commands, from this folder:

```bash
npm ci                            # exact versions from the lockfile
cp .env.example .env.local
npm run dev                       # http://127.0.0.1:3000
npm run lint
npm run typecheck
npm test                          # Vitest, no network
npx playwright install chromium   # once
npm run build && npm run e2e      # Playwright against the production build
```

Open the site at exactly the address in `APP_ORIGIN` (`127.0.0.1` is not `localhost`), or signing
in is refused.

`npm run e2e` needs ports 3000, 3001 and 8765 free, and takes about a minute. Its tests:

- `smoke.spec.ts`: pages load, with no backend at all.
- `session.spec.ts`: sessions in the production build, against the stand-in `stub-api.mjs`. One
  test waits 31 seconds on purpose.
- `stack.spec.ts` and `matches.spec.ts`: against a real stack (API, worker and database). They run
  only when `E2E_API_URL` is set and the demo account was seeded with `--demo-password` (see the
  root README). They leave a `web-e2e-...` account behind, and give the demo account the invented
  CV of `tests/fixtures/cv`.

## The match list

The CV page is not built yet, so upload a CV through the API first
([docs/API_EXAMPLES.md](../docs/API_EXAMPLES.md)).

- **Score:** a rank for your CV, not a probability. The warnings under it (such as "Skills only")
  say how it was made; see [docs/MATCHING.md](../docs/MATCHING.md).
- **Not every job is listed.** A job with nothing to compare (without an embedding model, one that
  lists no technologies) has no score, and jobs that fail your preferences are hidden unless asked.
- **Filters** are kept in the address. Filtering by source comes with the sources page.
- **Scoring in progress:** the page checks for up to two minutes, then offers to check again.
- **Apply links** come from job posts, which anyone can write. Only plain web links and single email
  addresses are clickable (`src/lib/matches/links.ts`).
- The shape of a score's explanation is written by hand in `src/lib/matches/explain.ts`, because
  `openapi.json` only says "an object". Change it together with `app/matching/reasons.py`.

## How sign-in works

The API returns tokens in the response body. Our server routes (`src/app/api/`) put them in
httpOnly cookies, so page scripts cannot read them, and call the API for the browser.

- **Forwarding:** only calls listed in `src/lib/server/allowlist.ts`.
- **Refresh:** the API ends a session when a refresh token is used twice. So parallel requests share
  one refresh (`refresh.ts`), and its result is kept for 30 seconds for requests still on their way.
- **CSRF:** a request that changes something must have `Origin` equal to `APP_ORIGIN`.
- **Security policy:** `src/proxy.ts` allows only scripts carrying a per-request nonce, so every
  page is rendered per request.

Limits:

- **One server process only.** The shared refresh lives in memory. With several instances (such as
  serverless hosting), parallel requests can still sign a user out. To be solved before deployment.
- For 30 seconds, a stolen old refresh token would get the new pair without being detected.
- The API limits sign-in attempts per IP address, and all web users share the web server's.
- After sign-out, an access token already issued keeps working at the API until it expires.

## ESLint version

ESLint 10 is used. ESLint 9 reached end of life on 2026-08-06
([version support](https://eslint.org/version-support/)).

`eslint-config-next` 16.4.6 accepts ESLint 10, but three plugins inside it (`eslint-plugin-react`,
`eslint-plugin-import`, `eslint-plugin-jsx-a11y`) do not yet list it as a supported peer. So
`npm install` prints peer warnings and `npm ls eslint` reports `invalid`; `npm ci` works. Checked
on 2026-11-27 with deliberately bad files: their rules, the React hooks and Next.js rules and our
own privacy rules all report. A Next.js issue
([#89764](https://github.com/vercel/next.js/issues/89764)) describes crashes with ESLint 10 that
did not happen here. If one does after an upgrade, pin ESLint 9 again and say why here.

## API types

`openapi.json` is exported from the FastAPI app, and `src/lib/api/schema.d.ts` is generated from
it. After changing the API, run the first command from the repository root and the second here:

```bash
uv run python -m scripts.export_openapi
npm run gen:api
```

A Python test fails if `openapi.json` is stale, and CI fails if the generated types are.
