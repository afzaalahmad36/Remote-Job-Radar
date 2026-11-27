# Web dashboard (Phase 6)

Next.js (App Router), TypeScript and Tailwind. So far only the setup: one placeholder page, a
typed API wrapper and the tests. The browser will never call the API; the Next.js server does.

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

Port 3000 must be free. The test that needs a backend (`e2e/stack.spec.ts`) runs only when
`E2E_API_URL` is set and the demo account was seeded with `--demo-password` (see the root README).

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
