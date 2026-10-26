# CLAUDE.md — Remote Job Radar

## 1. Project overview

**Remote Job Radar** is an AI-powered job pipeline. It:

1. Ingests job posts from Telegram channels, RSS feeds and job boards.
2. Uses an LLM to extract structured data (title, stack, salary, remote policy, apply email/URL).
3. Scores each job against the user's CV (0-100) with written reasons.
4. Drafts a tailored application email and tracks status: `new → shortlisted → applied → replied → interview → offer / rejected`.

**Purpose:** portfolio project. Code must be clean, tested, typed and explainable in an interview. The owner must understand every line, so prefer simple and readable over clever.

**Owner context:** MERN/Next.js developer, intermediate in Python. Explain Python-specific idioms briefly when introducing them (async, Pydantic, SQLAlchemy sessions, dependency injection).

---

## 2. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| API | FastAPI (async) + Uvicorn |
| Validation/settings | Pydantic v2 + pydantic-settings |
| Database | PostgreSQL 16 + pgvector |
| ORM / migrations | SQLAlchemy 2.0 (async, typed `Mapped[]`) + Alembic |
| Background jobs | Celery + Redis (beat for schedules) |
| Telegram ingest | Telethon |
| RSS ingest | feedparser + httpx |
| LLM | Anthropic Python SDK (structured JSON output validated by Pydantic) |
| Embeddings | Configurable provider behind an interface; stored in pgvector |
| Auth | JWT (access + refresh), password hashing with argon2 |
| Frontend | Next.js + TypeScript + Tailwind (in `/frontend`) |
| Testing | pytest, pytest-asyncio, httpx `AsyncClient`, factory-boy, testcontainers or a test DB |
| Quality | ruff (lint + format), mypy (strict on `app/`), pre-commit |
| DevOps | Docker, Docker Compose, GitHub Actions |
| Deploy | API on Render/Railway, frontend on Vercel |
| Package manager | `uv` (use `pyproject.toml`, no `requirements.txt`) |

---

## 3. Repository layout

```
job-radar/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── alembic.ini
├── migrations/                 # Alembic versions
├── app/
│   ├── main.py                 # FastAPI app factory, router registration
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── security.py         # JWT, password hashing
│   │   ├── logging.py          # structured logging setup
│   │   └── exceptions.py       # domain exceptions + handlers
│   ├── db/
│   │   ├── base.py             # Declarative base
│   │   ├── session.py          # async engine + session dependency
│   ├── models/                 # SQLAlchemy models (one file per entity)
│   ├── schemas/                # Pydantic request/response/LLM-output schemas
│   ├── routers/                # Thin HTTP layer only
│   ├── services/               # Business logic (no FastAPI imports)
│   ├── repositories/           # DB queries (optional; keep services clean)
│   ├── ingestion/
│   │   ├── base.py             # Source interface
│   │   ├── telegram.py
│   │   └── rss.py
│   ├── llm/
│   │   ├── client.py           # Anthropic wrapper (retries, timeouts, logging)
│   │   ├── prompts.py          # Prompt templates (versioned)
│   │   └── extractors.py       # job extraction, email drafting
│   ├── matching/
│   │   ├── embeddings.py
│   │   └── scoring.py          # hybrid score: embeddings + LLM reasoning
│   ├── tasks/                  # Celery tasks (thin wrappers around services)
│   │   ├── celery_app.py
│   │   └── jobs.py
│   └── api_deps.py             # shared FastAPI dependencies
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── conftest.py
│   └── factories.py
├── scripts/                    # one-off CLI helpers (seed data, etc.)
└── frontend/                   # Next.js app
```

---

## 4. Architecture rules

- **Layering:** `routers → services → repositories/models`. Dependencies point inward only.
  - Routers: parse request, call a service, return a schema. No business logic, no raw queries.
  - Services: business logic. **Must not import FastAPI.** Receive the DB session as an argument.
  - Models: persistence only. Schemas: API/LLM contracts only. Never return ORM models directly from endpoints.
- **Async everywhere** in the API path. Never call blocking I/O (requests, sync DB drivers, `time.sleep`) inside `async def`. Use `httpx.AsyncClient`, async SQLAlchemy, `asyncio.sleep`.
- **Celery tasks are thin:** they call a service function. Tasks must be **idempotent** (safe to retry) and deduplicate by `source + external_id`.
- **Outbound requests to user-supplied URLs must pass `app/core/url_safety.py` (SSRF protection) immediately before every request and every redirect hop.** Checking only when the URL is saved is not enough: DNS and redirects can change later.
- **LLM calls go through `app/llm/client.py` only.** Never call the SDK from elsewhere.
- **LLM output is untrusted.** Always parse into a Pydantic schema; on validation failure retry once with the error message, then mark the job `extraction_failed`. Never `eval` or trust raw text.
- **Config via environment only** (`app/core/config.py`). No hard-coded secrets, URLs or model names.
- **Dependency injection** via FastAPI `Depends` (session, current user, settings) to keep code testable.

---

## 5. Data model (core entities)

| Table | Key fields |
|---|---|
| `users` | id, email (stored lowercase, unique), hashed_password (argon2id), created_at |
| `refresh_tokens` | id, user_id, family_id, token_hash (SHA-256, unique), expires_at, revoked_at, created_at. One `family_id` per login; each refresh revokes the used token and issues a new one in the same family; replaying a used token revokes the whole family |
| `cvs` | id, user_id, raw_text, parsed_json, embedding (vector), updated_at |
| `sources` | id, user_id (FK users, ON DELETE CASCADE, indexed), type (`telegram`/`rss`), name, config (JSONB; RSS requires `url`, unknown keys rejected), is_active, last_synced_at (last successful sync), etag + last_modified (conditional-GET validators), last_error (short, sanitized reason the last sync failed; NULL after a success). A user only ever sees their own sources (others' are 404, never 403). A user cannot have two RSS sources with the same normalized URL (lowercase host, no default port, trailing slash or fragment): 409 `duplicate_source`, checked in the service, not by a DB constraint. Prefer `is_active=false` over deleting: deleting a source cascades to its jobs |
| `jobs` | id, source_id (a job belongs to a user through its source; `GET /jobs` joins `sources.user_id`), external_id (the post's GUID, else its link, else `sha256:` of title+link+date), raw_text (plain text, never HTML), title, company, stack (array), seniority, salary_min/max/currency, remote_policy, apply_email, apply_url, posted_at, embedding (vector), extraction_status, created_at. **Unique:** (source_id, external_id) |
| `job_matches` | id, user_id, job_id, score (0-100), reasons (JSONB), missing_skills (array), created_at. **Unique:** (user_id, job_id) |
| `applications` | id, user_id, job_id, status (enum), email_draft, sent_at, notes, updated_at |
| `application_events` | id, application_id, from_status, to_status, created_at (audit trail) |

Conventions: UUID primary keys, `created_at`/`updated_at` timestamps (UTC, timezone-aware), enums as Postgres enums, indexes on all foreign keys and on `jobs.created_at`, `job_matches.score`.

---

## 6. Core pipelines

**Ingestion → Extraction → Matching**
1. `ingest_source` (`services/ingestion_service.py`) fetches a source through its `SourceFetcher` (`ingestion/`: RSS now, Telegram later) → stores only `raw_text`, `posted_at` (UTC), `external_id` and `extraction_status=pending` in `jobs`; the structured fields stay NULL until step 2. Idempotent: `INSERT … ON CONFLICT (source_id, external_id) DO NOTHING` (rows inserted in a fixed order), so re-runs and overlapping syncs insert nothing twice. Network I/O never runs inside an open DB transaction: fetch first, then write. `raw_text` is capped at 20,000 characters per post. Changing a source's config URL clears its etag, last_modified and last_error. A failing source never raises: a short sanitized message goes to `sources.last_error` (no IPs, internal hostnames or stack traces) and the other sources are unaffected. RSS fetching: we fetch with httpx and feedparser only parses the bytes (always via a file-like object, never a str/bytes path or URL); `url_safety` runs before the first request and before every redirect hop (redirects are followed manually, max 3); timeouts 5 s connect / 10 s read / 20 s overall; body capped at 2 MB (decoded) and 200 entries; conditional GET via ETag / Last-Modified. Until Celery is wired in, it runs inline from `POST /sources/{id}/sync` (rate-limited per user).
2. `extract_job` task sends `raw_text` to the LLM with the extraction prompt → validated `JobExtraction` schema → updates the job.
3. `embed_job` computes the embedding.
4. `score_job_for_user` combines: cosine similarity (pgvector) for candidate retrieval, then the LLM for a 0-100 score with reasons and missing skills. Store in `job_matches`.
5. `draft_email` generates a tailored email using the CV + job; the user edits before sending. **The app never sends emails automatically.**

**Scoring rules:** the score must be explainable. Always store `reasons` and `missing_skills`. Cache results; don't re-score unless the CV or job changed.

---

## 7. LLM guidelines

- Model names come from settings (`LLM_MODEL_EXTRACTION`, `LLM_MODEL_SCORING`), not code.
- Prompts live in `app/llm/prompts.py` as versioned constants (`EXTRACT_JOB_V1`). Bump the version when changing wording and note it in the commit.
- Request JSON-only output and validate with Pydantic. Keep temperature low for extraction.
- Set timeouts, retries with exponential backoff, and a per-user/per-day cost guard.
- Log token usage per call (no raw PII in logs).
- **Tests must never call the real API.** Mock `app/llm/client.py` with recorded fixtures.

---

## 8. Coding conventions

- **Type hints on everything**; `mypy --strict` must pass for `app/`.
- Use `ruff` for linting and formatting (line length 100). Run before every commit.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants. Schemas end with `Create`, `Update`, `Read`, `Out` as appropriate.
- Prefer small functions (<40 lines) with a single responsibility. Docstrings on all public services (Google style, brief).
- No bare `except:`. Catch specific exceptions; raise domain exceptions from services and map them to HTTP errors in `core/exceptions.py`.
- Use `logging` (structured), never `print`.
- Use `pathlib`, f-strings, `dataclasses`/Pydantic over dicts for structured data.
- Use timezone-aware UTC datetimes only.
- No wildcard imports. Keep imports sorted (ruff handles it).
- API: RESTful, versioned under `/api/v1`, consistent error shape `{ "detail": ..., "code": ... }`, pagination on list endpoints (`limit`, `offset`).

---

## 9. Testing

- `pytest` with `pytest-asyncio`; target **≥ 80% coverage** on `services/`, `matching/`, `llm/`.
- Unit tests: services, scoring logic, parsers (pure functions, no I/O).
- Integration tests: API endpoints via `httpx.AsyncClient` against a real test Postgres (transaction rollback per test).
- Use factories (factory-boy) for test data. No shared mutable state between tests.
- Write tests **alongside each feature**, not after. A feature is not done without tests.
- Mock: Anthropic API, Telegram, outbound HTTP. Never hit the network in tests.

---

## 10. Commands

```bash
# setup
uv sync
cp .env.example .env

# run everything in containers (API on http://localhost:8000; db/redis are not published)
docker compose up --build

# or run the API on the host: needs your own Postgres/Redis, set in .env (DATABASE_URL, REDIS_URL)
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --no-proxy-headers --reload
uv run celery -A app.tasks.celery_app worker -l info
uv run celery -A app.tasks.celery_app beat -l info

# quality (run before every commit)
uv run ruff check . --fix && uv run ruff format .
uv run mypy app
uv run pytest -q --cov=app

# migrations
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

---

## 11. Environment variables (`.env.example`)

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/jobradar
# Tests use this and never DATABASE_URL; the database must be named jobradar_test or the run aborts.
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/jobradar_test
REDIS_URL=redis://localhost:6379/0
# In production the app refuses to start if SECRET_KEY is "change-me" or shorter than 32 chars.
SECRET_KEY=change-me
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
# Per client IP and auth endpoint (in-process limiter, single instance only).
AUTH_RATE_LIMIT_ATTEMPTS=10
AUTH_RATE_LIMIT_WINDOW_SECONDS=60
# Manual source syncs per user (each makes outbound HTTP requests).
SYNC_RATE_LIMIT_ATTEMPTS=10
SYNC_RATE_LIMIT_WINDOW_SECONDS=60
ANTHROPIC_API_KEY=
LLM_MODEL_EXTRACTION=
LLM_MODEL_SCORING=
EMBEDDING_PROVIDER=
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_SESSION_PATH=./.sessions/telegram
CORS_ORIGINS=http://localhost:3000
ENV=development
```

Never commit `.env` or Telegram session files. Keep them in `.gitignore`.

---

## 12. Git workflow

- Branch per feature: `feat/…`, `fix/…`, `chore/…`. Never commit straight to `main`.
- **Conventional Commits:** `feat: add rss ingestion`, `fix: dedupe telegram posts`, `test: cover scoring edge cases`.
- Small, focused commits. Each commit should pass lint and tests.
- CI (GitHub Actions) runs ruff, mypy and pytest with a Postgres service. Build must be green before merge.

---

## 13. How Claude should work in this repo

1. **Plan first.** For any non-trivial task, propose a short plan (files to touch, approach, risks) and wait for approval before writing code.
2. **One feature per session.** Don't refactor unrelated code or add unrequested features.
3. **Show the reasoning for non-obvious choices** in a brief comment or in the PR summary, and explain new Python concepts in 1-2 lines.
4. **Write tests with the feature**, then run `ruff`, `mypy` and `pytest` and report the results.
5. **Ask before** adding a new dependency, changing the DB schema, or altering the public API shape.
6. **Migrations:** after changing models, generate an Alembic migration and review the SQL; never edit applied migrations.
7. **Security:** never log secrets or full CV text; validate all input; use parameterized queries only; rate-limit auth endpoints.
8. **Keep diffs small and readable** so the owner can review every line.
9. When finishing a task, summarize: what changed, how to run it, what's tested, what's next.

---

## 14. Build phases (roadmap)

| Phase | Scope | Done when |
|---|---|---|
| 1 | Skeleton: FastAPI app, config, Postgres models, Alembic, Docker Compose, health endpoint, CI | `docker compose up` works; CI green |
| 2 | Auth + sources CRUD + ingestion (RSS first, then Telegram) with dedupe | Jobs appear in DB from a live source |
| 3 | LLM extraction pipeline with validation, retries and fixtures | ≥ 90% of sample posts parse correctly in tests |
| 4 | CV upload/parsing, embeddings, pgvector matching and scoring with reasons | Ranked job list per user |
| 5 | Application tracker (status pipeline + events) and email draft generator | Full lifecycle via API |
| 6 | Next.js dashboard: filters, score badges, kanban board, CV page | Usable UI against the live API |
| 7 | Hardening: rate limits, caching, observability, README with architecture diagram + demo GIF, deploy | Public live demo |

**Current phase:** 2

---

## 15. Definition of done

- [ ] Feature works end to end and is covered by tests
- [ ] `ruff`, `mypy`, `pytest` all pass locally and in CI
- [ ] Migration included (if the schema changed) and reviewed
- [ ] Docs updated (README / docstrings / this file if conventions changed)
- [ ] No secrets, debug prints or commented-out code
- [ ] Owner can explain how it works
