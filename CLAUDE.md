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
| Telegram ingest | Telethon (deferred, see section 14) |
| RSS ingest | feedparser + httpx |
| Extraction | Deterministic rules first (`app/extraction/`), then an optional LLM for what the rules miss |
| LLM | Provider-agnostic client over each provider's REST API with httpx, no vendor SDK: Gemini (free tier) now, a local Ollama later. JSON output validated by Pydantic. No paid API key is used anywhere |
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
│   ├── extraction/
│   │   ├── rules.py            # deterministic extraction (regex/keywords/dictionary), no I/O
│   │   ├── contacts.py         # email/phone finding and masking (bounded scans, no slow regexes)
│   │   ├── salary.py           # salary amounts, currency and period
│   │   ├── stack.py            # technology dictionary
│   │   ├── seniority.py        # seniority words (shared by the job rules and the CV suggestions)
│   │   ├── types.py            # Found / Confidence
│   │   ├── merge.py            # rules + model result: the model never beats a HIGH rule value
│   │   ├── schema.py           # JobExtraction: the contract for LLM output
│   │   ├── plan.py             # rules-phase decision: state, enrichment, fixed error phrases (pure)
│   │   ├── text_hash.py        # post text normalization + raw_text_hash (one place)
│   │   └── limits.py           # time limits of one extraction run, chosen together
│   ├── cv/                     # CV handling (personal data): imports no LLM client and no HTTP library (a test checks it)
│   │   ├── text.py             # clean + mask CV text (emails, phones, links); pure, bounded regexes
│   │   ├── pdf_worker.py       # standalone script: reads ONE PDF from stdin in its own process; imports only pypdf
│   │   ├── pdf_text.py         # runs the worker with a hard timeout and kills it; page/char caps; fixed failure codes
│   │   ├── skills.py           # CV skills: the stack dictionary scan without the job-post caps; cleans the user's edited list
│   │   └── suggestions.py      # years of experience and seniority guesses, each with its kind of match; shown, never scored
│   ├── llm/
│   │   ├── base.py             # provider-neutral types, errors, the LLMClient interface
│   │   ├── client.py           # build_client + the privacy/size/logging guard; the only door
│   │   ├── providers/          # one adapter per provider (gemini, fake); client.py imports them
│   │   └── prompts.py          # Prompt templates (versioned)
│   ├── matching/
│   │   ├── embeddings.py
│   │   └── scoring.py          # hybrid score: embeddings + LLM reasoning
│   ├── tasks/                  # Celery tasks (thin wrappers around services) + Redis helpers
│   │   ├── celery_app.py
│   │   └── jobs.py
│   └── api_deps.py             # shared FastAPI dependencies
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── conftest.py
│   └── factories.py
├── scripts/                    # manual CLI helpers: try_extraction, eval_label, eval_*, recompute_extractions (never run by CI or app code), plus check_no_cv_files (run by pre-commit and by a test)
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
- **LLM calls go through `app/llm/client.py` only.** Provider adapters (`app/llm/providers/`) are imported by `client.py` and nowhere else, and no vendor SDK is used (a test checks both). Rules come first: the LLM is asked only for what `app/extraction/rules.py` could not find.
- **LLM output is untrusted.** Always parse into a Pydantic schema; on validation failure retry once with the error message (field names only, never the received values), then give up on the LLM for that job and keep the rules' result (planned: the retry belongs to the extraction service, which is not built yet). A link or a salary from a model is accepted only if it appears in the post (built: `merge.merge`). Never `eval` or trust raw text.
- **Config via environment only** (`app/core/config.py`). No hard-coded secrets, URLs or model names.
- **Dependency injection** via FastAPI `Depends` (session, current user, settings) to keep code testable.

---

## 5. Data model (core entities)

| Table | Key fields |
|---|---|
| `users` | id, email (stored lowercase, unique), hashed_password (argon2id), created_at |
| `refresh_tokens` | id, user_id, family_id, token_hash (SHA-256, unique), expires_at, revoked_at, created_at. One `family_id` per login; each refresh revokes the used token and issues a new one in the same family; replaying a used token revokes the whole family |
| `cvs` | id, user_id (FK users, ON DELETE CASCADE, **unique**: one CV per user, uploading again replaces it), text (the CV as text, after emails, phones and links were masked; the uploaded file is never stored), source_format (enum `pdf`/`text`), page_count (PDFs), char_count, content_hash (SHA-256 of `text`), skills (array, at most 100 names of at most 50 characters; read from the text on upload, then editable), skills_edited (false while the list is the extractor's, true once the user saved their own; a new upload resets both), created_at, updated_at. Built. Planned in later Phase 4 PRs: embedding (vector). Personal data: never sent to a hosted provider, never logged, a custom `__repr__` hides the text. `DELETE /cv` deletes the row. |
| `sources` | id, user_id (FK users, ON DELETE CASCADE, indexed), type (`telegram`/`rss`), name, config (JSONB; RSS requires `url`, unknown keys rejected), is_active, last_synced_at (last successful sync), etag + last_modified (conditional-GET validators), last_error (short, sanitized reason the last sync failed; NULL after a success). A user only ever sees their own sources (others' are 404, never 403). A user cannot have two RSS sources with the same normalized URL (lowercase host, no default port, trailing slash or fragment): 409 `duplicate_source`, checked in the service, not by a DB constraint. Prefer `is_active=false` over deleting: deleting a source cascades to its jobs |
| `jobs` | id, source_id (a job belongs to a user through its source; `GET /jobs` joins `sources.user_id`), external_id (the post's GUID, else its link, else `sha256:` of title+link+date), raw_text (plain text, never HTML), title, company, stack (array), seniority (enum), salary_min/max/currency, salary_period (enum `year/month/week/day/hour`; NULL when the post gives amounts without a period), remote_policy, apply_email, apply_url, posted_at, embedding (vector, Phase 4), created_at, and the extraction state (see "Extraction state" below): extraction_status, llm_enrichment, extraction_method, extraction_rules_version, extraction_prompt_version, extraction_model (NULL when rules only), extracted_at, extraction_error (sanitized, max 500), extraction_attempts, extraction_retries, raw_text_hash. **Unique:** (source_id, external_id). **Check:** `extraction_status <> 'extracted' OR title IS NOT NULL` |
| `job_preferences` | id, user_id (FK users, ON DELETE CASCADE, **unique**), seniority_levels (`seniority[]`), remote_policies (`remote_policy[]`; an empty array means no preference), min_salary + salary_currency (ISO code) + salary_period (all three or none: a check constraint), notes (free text, masked like a CV), created_at, updated_at. Written only when the user saves (`PUT /preferences`); reading before the first save creates no row. Personal data like the CV: never logged, never sent to a provider, a custom `__repr__` hides the notes. |
| `llm_usage` | id, user_id (FK users, ON DELETE CASCADE), job_id (FK jobs, ON DELETE SET NULL, indexed), provider, model, prompt_version, outcome (short text code, see below), input_tokens / output_tokens / thought_tokens (nullable), created_at. One row per provider request, no post or response text. Indexed on (user_id, created_at) and created_at, for the daily and per-minute caps. Written by `services/llm_budget.py` (called by the enrichment service, which nothing runs yet). `outcome` values: `reserved` (the request is about to be made, or the worker died before filling in the result), `abandoned` (a `reserved` row the sweep gave up on; still counts), `ok`, `rate_limited` (429), `unavailable` (5xx), `timeout`, `invalid_response` (the reply was empty or failed validation), `rejected` (the provider refused the request with a 4xx status). Every outcome counts toward the caps except a deleted row (a request that never left) |
| `job_matches` | id, user_id, job_id, score (0-100), reasons (JSONB), missing_skills (array), created_at. **Unique:** (user_id, job_id) |
| `applications` | id, user_id, job_id, status (enum), email_draft, sent_at, notes, updated_at |
| `application_events` | id, application_id, from_status, to_status, created_at (audit trail) |

Conventions: UUID primary keys, `created_at`/`updated_at` timestamps (UTC, timezone-aware), enums as Postgres enums, indexes on all foreign keys and on `jobs.created_at`, `job_matches.score`.

### Extraction state

**Built:** the columns, enums, indexes and the check constraint above, `RULES_VERSION` and `app/extraction/text_hash.py` (the one place the post text is normalized: trim and collapse whitespace, no case folding); and the whole extraction flow (steps 1 to 5 below): `app/extraction/plan.py` (pure decisions, the fixed `ExtractionError` phrases), `app/repositories/extraction_repository.py` (compare-and-set SQL, hash-reuse lookup), `app/services/extraction_service.py` (`extract_job(factory, job_id, ...)`, the rules phase), `app/services/enrichment_service.py` (the model path), `app/services/llm_budget.py` (caps, throttle, reservation rows), `app/services/cooldown.py` with its Redis implementation `app/tasks/cooldown.py`, and stack grounding in `app/extraction/merge.py`. It is run by the Celery task `extract_job` (`app/tasks/jobs.py`), queued by the post-ingest hook, the sweep and the retry endpoint (below). With the kill switch off (the default) the task still runs the rules; it only never asks a model.

Why `extract_job` takes a session *factory* and not a session: a run is several short transactions (read, write the rules' result, and later reserve, call the provider, write the answer), and none may stay open while a provider request is in flight. Each phase function (`load_snapshot`, `find_reusable`, `write_*`) does take a session.

`extraction_status`:

| Value | Meaning |
|---|---|
| `pending` | Not processed yet, or the rules found no title and the model may still be asked. "Rules phase not run" is `extraction_method IS NULL`, not this status. |
| `extracted` | A title exists (the check constraint enforces it). The job is usable. |
| `extraction_failed` | No title and no further attempt will happen on its own. `extraction_error` has a short sanitized reason. A retry can revive it. Set at once when no title was found and no enrichment can happen (provider `none` or kill switch off). |
| `not_a_job` | The model said `is_job_posting=false`. Rules cannot tell, so only enrichment sets it. Rules data is kept; `GET /jobs` hides it by default. |

`llm_enrichment`: `not_needed` (default; both key fields found by the rules), `pending` (a key field is missing and the post passes `LLM_MIN_POST_CHARS`; waits for provider, quota and kill switch), `done` (valid model answer merged, or an identical text reused), `skipped` (post under the gate, given up after `LLM_ENRICHMENT_GIVE_UP_DAYS`, or the reply failed validation twice; the reason goes in `extraction_error`; rules data is kept).

Transitions, per `extract_job(job_id)` run (all built):
1. Rules phase (built): runs only for a job that is `pending` / `not_needed`; a job already `pending` for the model is left for step 3, anything else is a no-op (`already_done`). It writes the rules' result at once with `extraction_method=rules`, the versions, `extracted_at` and `extraction_attempts + 1`, overwriting every extraction field (so a revived job loses its old model result). Title found: `extracted`. No title: stays `pending` while a model could still be asked, otherwise `extraction_failed` straight away with a fixed reason. `extraction_attempts` counts rules-phase runs and doubles as the version for step 5.
2. Enrichment decision (built): both key fields found: `not_needed`; a key field missing and the post under the gate: `skipped` (`post too short for the model`); a model client exists, or a title is already there: `pending`; no title and no client: `skipped` plus `extraction_failed` (`no title found; the model is not enabled`). Before that, a job with the same `raw_text_hash` is copied instead if it is finished (`extraction_method` set, `llm_enrichment` `done` or `not_needed`, status `extracted` or `not_a_job`) and was made by the current rules version and, if a model was used, the current prompt version; a model-enriched donor is preferred. Only extraction fields are copied, never ids, source, text or ownership. If the rules phase itself raises, the job is stored as `extraction_failed` / `skipped` (`unexpected error`) instead of raising, so a sweep cannot loop on it; the error log line holds only the job id and the exception type, never the message or a traceback (both can quote the post).
3. Model call (built, `enrichment_service.enrich_job`), only for a job that is `pending` and only when a client and a cooldown store are both given (the kill switch decides whether `build_client` returns a client at all): a valid answer is merged (`extraction_method=rules_llm`, model and prompt version set, `done`); `is_job_posting=false` gives `not_a_job`; an answer with no title anywhere gives `extraction_failed` (`no title found`). A 429, a 5xx, a timeout, a full cap or throttle, or a pause in effect changes nothing on the job (it stays `pending`; the result is `deferred` with a reason and a retry hint of at least one second). An invalid answer (or an empty or cut-off one) is retried once with field names only, then the model is given up on (`skipped`, `model reply was not valid`): at most 2 requests per run. A request refused by our own guard gives `skipped` (`request refused locally`). A request the provider refuses with a 4xx is told apart by its status (`LLMRejected.status_code`, a number, never provider text): only a **400**, which is about that job's content, gives `skipped` (`model request was rejected`, no pause, a warning log). Every other refusal (401 bad key, 403 no access, 404 unknown model, any other 4xx, or an error with no status) is a **configuration failure**, not the job's fault: the job stays `pending`, the result is `deferred` (`configuration`, retry hint 900 s), everyone pauses for 15 minutes (the longest pause we allow), and an error-level log line (job id and status only) makes it visible. The request is still counted (`rejected`). Just before spending a request the service looks once more for a finished job with identical text and copies it instead. A model-supplied stack entry is kept only if it appears in the post (case-insensitive, whole word, also by its dictionary spelling); dropped entries are counted in a log line (`field=stack count=N`, nothing else). `salary_period` stays NULL when the rules found amounts without a period: the model's period is never mixed into the rules' amounts (the salary is taken whole from one side). The model's salary group replaces a low-confidence rules guess only if the post supports **all** of it: every amount it gives (min and max) must literally appear in the post as a number (commas, dots or spaces as thousands separators; a `k` suffix means thousands, so `90k` supports 90,000 and a bare `90` does not), and a period it gives must be stated in the post in words a salary uses (per year, annually, /mo, weekly, per day, hourly, ...). Otherwise the whole group is rejected, not trimmed, and the rules' value stays.
   Reservation (built, `llm_budget.reserve`): before a request, one short transaction takes a global advisory lock, counts the user's and all `llm_usage` rows in the last 24 hours and all rows in the last minute, and inserts a row with outcome `reserved` if all three are under their limits; the provider call then runs with no transaction open, and a second short transaction fills in the outcome and tokens (for a valid answer, in the same transaction as the job's result). A worker that dies after reserving leaves a `reserved` row that keeps counting inside the 24 h window (the provider may have counted the request); `abandon_stale_reservations` (built; the sweep that calls it is planned) relabels rows older than 10 minutes `abandoned`, which also count, and logs a warning when more than 3 abandoned rows fall in one 24 h window, so a crash loop is visible. A request that never left (refused by the privacy or size guard: `LLMError.never_sent`) has its row deleted. Every other failed request keeps its row. The limits are `LLM_DAILY_REQUEST_CAP_PER_USER` and `LLM_DAILY_REQUEST_CAP_GLOBAL` (rolling 24 h) and `LLM_REQUESTS_PER_MINUTE` (rolling 60 s, over all users); a window is `created_at > now - window`, so a row exactly 24 h or 60 s old no longer counts. A refusal says which limit is full and when enough rows age out for a place to open. The defaults (100 per user and 150 overall per day, 5 per minute) are deliberately below the 500 per day and 15 per minute seen for gemini-3.5-flash-lite on the free tier, and there are no validation ceilings: limits depend on the model and the project.
   Cooldown (built: the `Cooldown` Protocol in `services/cooldown.py`, `RedisCooldown` in `tasks/cooldown.py`, `FakeCooldown` in tests): a shared Redis key `llm:cooldown:<provider>` with an expiry, started after a 429 (the `Retry-After` kept between 30 s and 900 s, 60 s without one), after a 5xx, timeout or dropped connection (30 s) and after a rejected request (900 s); a longer pause is never shortened. If Redis is unreachable or slow the check **fails closed**: no model request is made and the job stays as it is, while the rules phase keeps working because it needs no Redis. Every deferral carries a retry hint of at least one second and the "store is down" one at least 60 s, so a caller that honours the hint cannot tight-loop while Redis is down; the planned sweep makes one pass per scheduled run and has no retry loop inside it.
4. Sweep (built: `services/extraction_sweep.py` for the database steps, `sweep_extraction` in `tasks/jobs.py` for queueing; beat runs it every 5 minutes, a constant `SWEEP_INTERVAL_SECONDS`, not a setting). In order: (1) `abandon_stale_reservations`; (2) give-up: one set-based UPDATE of every job still `pending` after `LLM_ENRICHMENT_GIVE_UP_DAYS` (counted from `extracted_at`, the moment the rules phase made it wait, falling back to `created_at` only when `extracted_at` is NULL; so a row that sat in the database before the pipeline ran still gets the full period, and a retried job gets a fresh one when its new rules phase runs) to `skipped` with `gave up waiting for the model`, and `extraction_status` becomes `extraction_failed` where the job has no title (the rules' data stays); (3) queue up to 100 jobs with `extraction_method IS NULL` that are `pending` / `not_needed` (new jobs whose task was lost, and jobs a retry reset), oldest first; (4) queue jobs waiting for the model, oldest first, **only when asking could succeed**: a client exists (provider set and kill switch on), no pause is in effect, and the global budget has room, in which case at most as many jobs as there are global places left; owners who used their daily cap are left out so they cannot fill the batch and starve others. With the default provider `none`, step 4 queues nothing. The database steps (1 and 2) need no Redis and always run. If the queue is unreachable the sweep logs one warning at the first failed publish and stops queueing for that run (no loop, no retry inside the sweep: the next scheduled sweep is the retry); if the cooldown store is unreachable it logs one warning and skips step 4 only.
5. Every state change is a compare-and-set (built for the rules phase): `UPDATE jobs SET ... WHERE id = :id AND extraction_status = :seen AND llm_enrichment = :seen AND extraction_attempts = :seen`, with `extraction_attempts + 1` in the rules write. Zero rows means another worker got there first: the run logs it, writes nothing more and returns `lost_race`; the winner's result stands. Tested with two real transactions racing on one job (exactly one writes). Nothing is locked while a worker works; the planned per-job Redis lock only saves duplicate provider requests, and these writes are what keep the data correct.

Retry endpoint `POST /api/v1/jobs/{id}/extract` (built): owner only (others get the same 404 as a missing job), rate-limited per user in its own bucket (the `SYNC_RATE_LIMIT_*` numbers, separate from the sync bucket). Allowed when the job is `extraction_failed` or its `llm_enrichment` is `skipped`, otherwise 409 `not_retryable`. At most 5 retries per job, then 409 `retries_exhausted`. The count is `extraction_retries`, a separate column from `extraction_attempts`: attempts also grow with sweeps and Celery redeliveries, so using it would let the system's own re-runs use up the user's retries. A retry sets `extraction_status=pending`, `llm_enrichment=not_needed`, `extraction_method=NULL` and clears `extraction_error`, adds one to `extraction_retries`, enqueues the task and returns 202 (`queued`, or `already_running` without using a retry when a run holds the job's lock). The reset and the enqueue happen in one transaction that is committed only after the task was queued (with a 2 s delay so the commit lands first; if a worker is ever early, the sweep finds the job because `extraction_method` is NULL), so if the queue is unreachable the reset is rolled back, no retry is used and the answer is 503 `queue_unavailable`. Clearing `extraction_method` keeps "rules phase not run" true for the sweep, so a lost message cannot strand the job. **The retry endpoint is the only code that sets `extraction_method` back to NULL.** It does not bypass the length gate, the budgets or the kill switch.

Migration note: `jobs.seniority` was free text that nothing wrote to. The migration fails loudly if any row has a value, then converts the column to the `seniority` enum. Postgres cannot remove an enum value, so a downgrade leaves `not_a_job` in the `extraction_status` type (rows holding it move to `extraction_failed`).

---

## 6. Core pipelines

**Ingestion → Extraction → Matching**
1. `ingest_source` (`services/ingestion_service.py`) fetches a source through its `SourceFetcher` (`ingestion/`: RSS now, Telegram later) → stores only `raw_text`, `posted_at` (UTC), `external_id` and `extraction_status=pending` in `jobs`; the structured fields stay NULL until step 2. Idempotent: `INSERT … ON CONFLICT (source_id, external_id) DO NOTHING` (rows inserted in a fixed order), so re-runs and overlapping syncs insert nothing twice. Network I/O never runs inside an open DB transaction: fetch first, then write. `raw_text` is capped at 20,000 characters per post. Changing a source's config URL clears its etag, last_modified and last_error. A failing source never raises: a short sanitized message goes to `sources.last_error` (no IPs, internal hostnames or stack traces) and the other sources are unaffected. RSS fetching: we fetch with httpx and feedparser only parses the bytes (always via a file-like object, never a str/bytes path or URL); `url_safety` runs before the first request and before every redirect hop (redirects are followed manually, max 3); timeouts 5 s connect / 10 s read / 20 s overall; body capped at 2 MB (decoded) and 200 entries; conditional GET via ETag / Last-Modified. It runs in the background as Celery tasks (`app/tasks/`, thin wrappers around the services; Redis broker, JSON only, no result backend, `acks_late`, soft 60 s / hard 90 s limits, prefetch 1): `ingest_source(source_id)` (re-loads the source, skips it if deleted, own NullPool engine per run because each task has its own event loop), `ingest_all_active_sources()` (beat, every `INGEST_INTERVAL_MINUTES`: one task per active source, spread over a few seconds), and `purge_expired_refresh_tokens()` (beat, daily: only tokens expired more than 7 days ago, so a rotated-but-unexpired token is always kept for reuse detection). Two runs of one source never overlap: a Redis lock with an expiry (token-owned, 120 s TTL) per source; a run that finds it held is skipped, not queued. `POST /sources/{id}/sync` (owner only, rate-limited per user) just enqueues the task and returns 202 (`queued`, or `already_running` if a sync is in progress); the result appears in `sources.last_synced_at` / `last_error` and `GET /jobs`. Every post a sync actually inserts is then queued for extraction by the ingest task (`IngestionSummary.new_job_ids`, the ids `INSERT ... RETURNING` returned, spread over a few seconds); a failed publish is only logged and never fails the sync, because the extraction sweep is the safety net. The Celery task `extract_job(job_id)` (soft 60 s / hard 90 s from `limits.py`, retrying only database and Redis errors and never a provider outcome, which the service decides) takes a per-job Redis lock (token-owned, 120 s TTL; an optimisation against double requests, the compare-and-set writes stay the correctness guarantee), builds the real client (`build_client`) and the Redis cooldown inside the task, and uses its own `NullPool` engine like the ingest task.
2. **Extraction (Phase 3, being built step by step).** Built: the deterministic rules (`app/extraction/rules.py`: emails, links, salary and period, remote policy, seniority, stack, title, company), pure functions with no I/O that never raise and run in milliseconds even on hostile 20,000-character input (a timed test enforces it); the `JobExtraction` schema; `merge.merge`; the `extract_job_v1` prompt; the provider-agnostic client with its privacy guard and the Gemini adapter; and the time limits (`app/extraction/limits.py`). The key fields (`merge.KEY_FIELDS`) are a title and a stack; LOW-confidence guesses count as missing; the apply contact, company, remote policy, salary and seniority are optional and never trigger a request (measured on two real feeds: a model cannot supply an apply link that the post's text does not contain). `merge.merge` combines the rules' result with the model's, and a model never overrides a HIGH-confidence rule value. A post shorter than `LLM_MIN_POST_CHARS` (default 500, `merge.is_too_short_to_enrich`) is never sent: it is a headline-only stub and stays rules-only (measured: stubs were 172-268 characters, real posts at least 1,945). `extract_job(job_id)` (built) runs the rules first and writes that result at once, so a job is usable as soon as a title is found; only if a key field is still missing is the model asked, through `app/llm/client.py`, for the remaining fields. Enrichment is a separate, optional state that waits for the provider, quotas and the kill switch, and is given up after `LLM_ENRICHMENT_GIVE_UP_DAYS` with the rules' data kept (built: the sweep). Each run makes at most 2 provider requests and has an overall deadline (`limits.py`; Celery soft 60 s / hard 90 s; the per-job lock outlives the hard limit), enforced by the service and the task. The rules' output is heuristic: expect misses and the occasional wrong guess, which is why a confidence is kept per field. `rules_v2` (`RULES_VERSION`) added general rules found by diagnosing the first accuracy run: a post naming more than 12 distinct technologies reports no stack (a recruiter's list); a technology only named in a customer list or an "X alternative" phrase, and an ambiguous English word (swift, rust, ruby, ...) written in lower case, are not stack; money in a benefit sentence (401(k) match, stipend, allowance, bonus) with no pay word is not a salary, and `$` with `CAN`/`CAD` in the sentence is CAD; a title loses a leading "FULL TIME:" and a trailing parenthetical holding a price, "welcome", WFH or training (seniority is read from the cleaned title); `WFH` is remote unless its line makes it an option, a policy or a few days a week, and links are ignored when reading the remote policy. `rules_v3` added one board-specific rule, written from a diagnosis of all 200 stored posts: when the text itself gives mixed remote signals (a remote word next to a hybrid or on-site word, which `_remote_policy` leaves undecided) and the post's `Headquarters:` header (first 6 non-empty lines) names remote, anywhere or worldwide with no hybrid, on-site or in-office word in that line, the policy is remote (HIGH); a body statement with a concrete obligation ("3 days a week", "must live within", "within N miles", "relocate") keeps the result at none, and a header with only a place decides nothing. It cannot change the parse-correct share, because a missing remote policy never fails a post. The version only decides whether an old result may be reused for identical text: it does not by itself make the sweep recompute stored extractions. Identical post text reuses an existing extraction instead of making a new provider request, through a `raw_text_hash` column on `jobs` (SHA-256 of the normalized text) and a lookup before any provider call; the same post is stored once per source, so the two public feeds measured so far held 99 rows but only 55 distinct texts. The `raw_text_hash` column exists now (schema step, see section 5); the lookup and the reuse are built.
3. `embed_job` computes the embedding.
4. `score_job_for_user` combines: cosine similarity (pgvector) for candidate retrieval, then a 0-100 hybrid score (skill overlap, hard filters, embedding similarity) with reasons and missing skills, **computed without any LLM** (Phase 4 uses none: the CV must not reach a hosted provider, and a local model is not needed for this). Store in `job_matches`.
5. `draft_email` generates a tailored email using the CV + job; the user edits before sending. **The app never sends emails automatically.** Because the CV never reaches a hosted provider, this draft must be **template-based, or use a local model** (Ollama), never a hosted free-tier one. `HOSTED_ALLOWED_PURPOSES` stays pinned to `{job_post_extraction, connectivity_check}`: do not add a purpose to it for the draft (a test fails if it changes).

### CV upload (built in Phase 4, PR 1)

- `PUT /api/v1/cv`: the **request body is the file itself** (not multipart), with `Content-Type` `application/pdf`, `text/plain` or `application/octet-stream`; the content decides the format (`%PDF-` at the start means PDF, anything else must be UTF-8 text without NUL bytes), so a wrong header or file name changes nothing. It creates or replaces the caller's one CV and returns `CvRead` (id, `source_format`, `page_count`, `char_count`, the stored masked `text`, timestamps). `GET /api/v1/cv` returns it, `DELETE /api/v1/cv` deletes it (204). There is no CV id in the API, so a user only ever reaches their own; a missing CV is 404. Every endpoint needs a login (401).
- Limits (settings `CV_*`): 2 MB per upload (413 `cv_too_large`, enforced while the body streams in, and from `Content-Length` when it is sent), 10 PDF pages (checked from the page tree **before** any text is extracted), 30,000 characters of text (a longer CV is refused, never cut), 5 uploads per user per 60 s (429). Other errors: 415 `cv_unsupported_type`, 422 `cv_unreadable` (damaged, password-protected, no text such as a scanned PDF, too many pages, too long, PDF read timed out), 503 `cv_busy` (all PDF-reading slots in use). Every message is a fixed phrase.
- A PDF is read by `app/cv/pdf_worker.py` in a **separate Python process** (`python -I`, bytes in on stdin, one JSON line out, no file written, `setrlimit(RLIMIT_AS)` memory cap, `CV_PDF_MEMORY_LIMIT_MB`, 512 MB by default: enforced on Linux, not on Windows, ignored by macOS), started with a hard timeout (`CV_PDF_TIMEOUT_SECONDS`, 10 s) and **killed** when it expires; a thread cannot be killed, so a hostile PDF could pin one. At most 2 PDFs are read at the same moment per API process. **Both the upload rate limiter and this 2-PDF slot limit are per process** (memory of one API process, not shared): with N API processes or instances the real limits are N times higher, and both reset on restart. Moving the rate limiter to Redis is planned for Phase 7; until then, run a single API process, or divide the numbers. The reader is `pypdf` (BSD): keep it updated, because PDF parsers regularly get denial-of-service fixes. Two-column layouts may come out in the wrong order, and scanned PDFs have no text (no OCR).
- Stored text goes through `clean_cv_text` (NFC, line endings, control characters, runs of spaces and blank lines) and `mask_cv_text` (emails, phone numbers, links such as a GitHub or LinkedIn URL). Uploading again updates the same row in one `INSERT ... ON CONFLICT (user_id) DO UPDATE`. Curl examples (a raw body with `--data-binary` and a `Content-Type`, not `-F`) are in `docs/API_EXAMPLES.md`.

### CV skills, suggestions and job preferences (built in Phase 4, PR 2)

- **Skills.** `app/cv/skills.py::find_skills` is `app/extraction/stack.py::find_technologies` (the scanner behind `find_stack`) with the two job-post safeguards off: no 12-technology cap (a recruiter's list is a job-post problem; a CV may name 30) and no customer-list/"X alternative" rule. The dictionary, the aliases and the ambiguous-word rule (swift, rust, ruby... count only when capitalised) are shared, so a CV skill and a job's stack entry are spelled the same, which is what scoring compares. **List context** (CV only; the job rules keep the strict rule): "Go", "R", "C" and the ambiguous words count only as a whole item of a skills-style list, either a line in a section under a heading (Skills, Technologies, Tech stack, Languages, Tools; the section runs to the next blank line, section title or 8 lines) or a comma/slash/"and" line of short items (at most 3 words each; a lead-in clause may stick to the first item) naming at least two other known technologies. So `Skills: Python, Go, SQL` gives all three, while "Go-To-Market" and "go to production" never do. A skill the dictionary lacks is not found; the user adds those by hand. `save_cv` fills `cvs.skills` from the masked text in the same upsert and sets `skills_edited=false`.
- `GET /api/v1/cv/skills` returns `{skills, edited, suggestions, updated_at}`; `PUT /api/v1/cv/skills` with `{"skills": [...]}` replaces the list (one UPDATE, `skills_edited=true`). Names are cleaned (whitespace collapsed, control characters removed), a name the dictionary knows gets its spelling, duplicates ignoring case are dropped, at most 100 skills of 50 characters; a bad body is a 422 with a fixed message. No CV: 404. Uploading a new CV re-extracts and discards the edit. A CV stored before the migration has an empty list until saved or re-uploaded.
- **Suggestions** (`app/cv/suggestions.py`) are computed from the stored text on every read, not stored. Years: the largest believable "N years of experience" phrase (the word "experience" must follow, so "5 years of Python" is ignored); kind of match `years phrase`. Seniority: exactly one level named in the first 3 digit-free lines (`title keyword in the first lines`), else in lines that run until "present/current" (`title keyword in the current role`); bullets and long lines are skipped, two levels in one place give no guess. **They are only shown to the user: nothing scores, filters or ranks with them, and they are not stored or part of the preferences (a test scans `app/matching` and any scoring module and fails if one reads them)** (a heuristic guess must not silently decide what a person sees). The seniority words live in `app/extraction/seniority.py`, shared with the job rules (the CV code cannot import `rules.py`, which reaches the network layer).
- **Preferences.** `GET /api/v1/preferences` (empty defaults with `id`/`updated_at` null before the first save, no row created) and `PUT /api/v1/preferences` (saves explicitly; the body replaces everything, a field left out is cleared). Fields: `seniority_levels`, `remote_policies`, `min_salary` with `salary_currency` and `salary_period` (all three or none, else 422; currency upper-cased, 3 letters), `notes` (at most 2,000 characters, cleaned and masked like a CV, blank becomes null). Lists drop duplicates and keep order. No other code writes this table; nothing infers preferences from behaviour.

### Extraction accuracy evaluation (the pass rule)

This rule is fixed BEFORE any post is labelled, so the target cannot be tuned to the result. It is
measured by hand (`scripts/eval_label.py` to label, `scripts/try_extraction.py --eval` to score);
CI never labels or scores real posts, and the scoring never makes a provider request.

**What is labelled.** N=30 posts (`--n`, a fixed `--seed`) from one source, with distinct
`raw_text_hash`, labelled blind (the system's extraction is never shown). A blank label field means
"the post does not say". Stack: list every technology the post names, required or optional.
Salary follows the rules' conventions: a single amount is min = max, "up to X" is max only, "from X"
or "X+" is min only.

**What counts as a stack label.** An entry counts only if it is the name of a product, language,
framework, library, database, cloud service or protocol that someone would list on a CV. Practices,
methods, domains, standards (SOC 2, ISO 27001) and abstract nouns are excluded, and the lists in
`scripts/eval_clean_policy.toml` encode the borderline calls (`scripts/eval_clean_labels.py --policy`
applies them). The lists were fixed before scoring and without looking at any extraction: the cleaner
reads only the labels file.

**Two outputs are scored the same way:** the stored extraction (the job's database columns) and the
rules recomputed offline from `raw_text`.

**Outcome of one field**, in one of five buckets: both blank = correct (correctly absent); same
value = correct; different value = wrong; label has a value and the system is blank = missing; label
is blank and the system has a value = invented.

**Normalisation** (both sides): Unicode NFKC, casefold, collapse whitespace, strip surrounding
punctuation and emoji. Title: also remove gender markers such as `(m/f/d)` and treat `- – / | ,` as
spaces; the match is then exact (no fuzzy match, no "contains"). Company: the same, plus legal
suffixes at the end (inc, llc, ltd, gmbh, ag, bv, sa, plc, corp, co). Stack entries go through
`canonical_name()` (`app/extraction/stack.py`); an entry the dictionary does not know is compared as
its casefolded, whitespace-collapsed text. Enums compare as enum values (`on-site` is `onsite`);
currency is an upper-case ISO code; amounts are exact integers.

**Title convention.** The title label is the role title without the company prefix, and the company
goes in the company label: a headline "Acme Corp: Senior Backend Engineer" is title `Senior Backend
Engineer`, company `Acme Corp`. This is the scoring's normalisation applied as written, not a
loosening of it: it removes gender markers and turns `- – — / | ,` into spaces, but it does not
remove a company prefix and does not treat `:` as a separator, so a title that still carries
"Acme Corp: " never matches a title without it. The rules split a "Company: Role" headline the same
way. The labels were corrected from the post text alone, before any scoring, by
`scripts/eval_fix_labels.py` (headline split, seniority word in the title, salary amounts and period
words found in the text, stack entries not in the text removed and none added; `remote_policy` is
left as labelled). It reads only each post's id and text, never an extraction. The earlier versions
are kept as `labels.raw.json` (before the stack cleaner) and `labels.clean-v1.json` (before this fix).

**Second pass over the labels.** After the first accuracy run was diagnosed, the labels were completed
a second time by `scripts/eval_complete_labels.py`, under rules fixed in the script before anything was
written and described in its docstring (the stack rules were revised twice after dry-runs showed
false adds, such as a "Go-To-Market" read as Go or an employer's name ("Databricks", "Maven" for
"Maven Clinic") read as a technology, which is now ignored everywhere in the post; the labels were not
written before either revision, and the script has a `--max-adds-per-post` safety limit): a missing salary amount or currency is filled only when the post states it
(`CAN`/`CAD` gives CAD), and a stack entry is added only when it is on the allow-list or in the stack
dictionary, appears as a whole word in the post outside a customer list or an "X alternative" phrase,
is not on the deny list, and the post names 12 or fewer technologies (swift, rust, ruby, unreal, dart,
flask, rails, celery and jest need a capital letter, and are flagged in the change table). It reads
only each post's id and text, never imports the extractor's rules, and keeps the first-pass labels as
`labels.pass1.json`. **The first-pass score is kept and reported next to the second-pass score**
(`docs/ACCURACY.md`), because the first-pass labels left out technologies and salary parts the post
states, so part of the first run's errors were label omissions, but the second-pass rules were written
after seeing the diagnosis and mirror some of the extractor's `rules_v2` rules; a score on the
completed labels is therefore not independent evidence, and the first-pass score stays the baseline.

**A post parses correctly when all of these hold:**
1. Title: the system has one and it matches the label.
2. Company: it matches, or the label has none (an invented company does not fail the post, but it
   is counted as an invented value for the gate below).
3. Stack: the system has no entry outside the label and at least 70% of the label is found (an empty
   label passes recall trivially; any system entry then fails).
4. Remote policy, salary and seniority: missing or correct. A wrong or invented value fails the post.
   Salary is correct only when every non-blank component (min, max, currency, period) matches; a
   blank component never fails.

**Target: at least 90% of the labelled posts parse correctly, on this sample** (30 posts, one source,
a fixed seed). It is not a claim about accuracy in general. The result is printed with a Wilson 95%
interval, because 27 of 30 still leaves a wide range; the target is judged on the point estimate.

**Invented-values gate:** if more than 10% of the labelled posts contain any invented value (company,
a salary component, seniority or remote policy), the evaluation reports FAIL whatever the
parse-correct share is. An invented company counts here even though it does not fail the post.

**Reported next to the verdict** (not part of the pass rule): per-field accuracy, and per-field recall
on non-blank labels (how often a non-blank label value is found correctly); counts of wrong, missing
and invented values; stack precision, recall and F1 (micro-averaged); the stack dictionary-gap count
(label entries unknown to the dictionary, which cost recall without being a rule miss); and the
number of posts excluded, with the reason (stale hash, blank title label, skipped). Excluded posts
are not scored; the share is computed over the posts that remain.

**Scoring rules:** the score must be explainable. Always store `reasons` and `missing_skills`. Cache results; don't re-score unless the CV or job changed.

---

## 7. LLM guidelines

- **No paid API is used anywhere**, and no key for one belongs in this repo. Providers are a free Gemini key (billing left off on that project) or a local Ollama. `LLM_PROVIDER` is `none` (rules only, the default), `gemini`, `ollama` (reserved) or `fake` (development only; refused in production). `LLM_EXTRACTION_ENABLED` is a kill switch and is off by default.
- **Privacy: only public job-post text may be sent to a hosted free-tier provider. A CV and any other personal data are never sent to one.** A free tier may use what it receives to improve its products and humans may read it. Emails and phone numbers are removed from a post before it is sent. The rule is enforced in code: every request declares a `Purpose`, and `GuardedClient` refuses to send personal data or an unlisted purpose to a provider that is not `local`. The only other hosted purpose is `connectivity_check` (a fixed harmless sentence for the script's `--ping`); it is also the only one allowed to leave the provider's retention default (`store=None`), so `GuardedClient` accepts it only when the input equals the fixed `PING_INPUT` exactly, the system text is empty or `PING_SYSTEM`, and there is no schema: job text can never travel under it. CV work may use only a local provider, and Phase 4 uses no LLM on a CV at all. The rule has four layers: (1) a test pins `HOSTED_ALLOWED_PURPOSES` to exactly two purposes; (2) `Purpose.CV_ANALYSIS` exists but is on no allow-list, and a test shows `GuardedClient` refuses every non-allowed purpose for a hosted provider; (3) a test checks that `app/cv/`, `services/cv_service.py`, `services/preferences_service.py` and the CV and preferences models/schemas (and, directly, both routers) import no `app.llm`, `app.tasks`, HTTP client or LLM SDK, directly or through other `app` modules; (4) a canary test plants a unique marker in a CV, in the edited skills and in the preferences notes, and checks it never appears in an LLM call, a log record or an error response. **Storing a CV:** text only (emails, phones and links masked; names cannot be masked reliably and stay), the original file is discarded and never written to disk (the upload is read as the raw request body with a hard byte cap, not as multipart, because Starlette spools multipart files over 1 MB to a temporary file; a test sets the temp directory to an empty folder and checks it stays empty). **Never logged:** logs carry user id, format, page count, character count and timing; error messages are fixed phrases chosen by the service (a PDF parser's message can quote the file). **Backups and run records:** a `pg_dump` backup contains the `cvs` table, and any run record or script output may too; keep them outside the repository (`.local/` is git-ignored, `*.pdf` is git-ignored, and `scripts/check_no_cv_files.py` rejects a PDF or a CV-named non-code file outside `tests/fixtures/cv`, in pre-commit and in the test suite). Only invented CVs (`tests/fixtures/cv`) are used in tests, docs and screenshots.
- Model names come from settings (`LLM_MODEL_EXTRACTION`), not code. They change often: check the provider's current model list.
- Prompts live in `app/llm/prompts.py` as versioned constants (`EXTRACT_JOB_V1`). Bump the version when changing wording and note it in the commit.
- Request JSON-only output with a schema and validate with Pydantic. Do not send sampling parameters such as `temperature`: current models ignore or reject them.
- Budgets are counted in provider **requests**, not money: per-user and global daily caps, a per-minute throttle, and a shared cooldown after a 429. A 429 never fails a job; it stays pending. Counted from the `llm_usage` table through a reservation row made before each call (section 5, "Extraction state"); the defaults are small on purpose and have no ceiling tied to one model. Built: one adapter call is exactly one HTTP request with no hidden retries, so every request can be counted.
- Timeouts everywhere (one request 20 s by default, at most 25 s; a run makes at most 2 requests and has an overall deadline, both enforced by the enrichment service). Adapters never follow redirects, send the key in a header only, and never log or store request text, response text or provider error bodies. Log provider, model, token counts, timing and outcome.
- **Tests must never call a real provider.** Use `FakeLLMClient` (`app/llm/providers/fake.py`) or an `httpx.MockTransport`; the network guard enforces it.

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
- Mock: LLM providers, Telegram, outbound HTTP. Never hit the network in tests.
- **Tests never use a real Redis or broker** (nor any real network service except the test DB). Use `FakeTaskQueue` / `FakeSourceLocks` / `FakeCooldown` (`tests/fakes.py`) through dependency overrides or arguments, and point any Redis setting at `UNREACHABLE_REDIS_URL` (a port where nothing listens). `tests/network_guard.py` enforces this for every test: any other connection raises `NetworkAccessBlockedError`. CI has no Redis service on purpose, and every CI job has a `timeout-minutes`.

---

## 10. Commands

```bash
# setup
uv sync
cp .env.example .env

# run everything in containers: api (the only one that migrates and publishes a port,
# http://localhost:8000), worker, beat, db and redis (db/redis are not published)
docker compose up --build
# never scale beat: exactly one scheduler must run (`--scale worker=N` is fine)

# or run on the host: needs your own Postgres/Redis, set in .env (DATABASE_URL, REDIS_URL)
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --no-proxy-headers --reload
uv run celery -A app.tasks.celery_app worker -l info    # runs the tasks (Linux/macOS)
uv run celery -A app.tasks.celery_app beat -l info      # sends them on schedule (one instance only)
# Beat also sends sweep_extraction every 5 minutes; the worker runs extract_job. The worker needs
# the LLM_* / GEMINI_* settings like the api: in Docker all three read the same .env (env_file),
# on the host export them for the worker too. Keep LLM_EXTRACTION_ENABLED=false until you mean
# to ask a provider: with it off the worker still runs the rules and never contacts a model.
# Windows host-run worker: Celery's default prefork pool does not work on Windows, use solo:
uv run celery -A app.tasks.celery_app worker -l info -P solo
# NOTE: with -P solo neither the soft (60 s) nor the hard (90 s) task time limit is enforced and
# only one task runs at a time (CELERY_WORKER_CONCURRENCY is ignored). A feed fetch is still
# bounded by its own 20 s deadline. The Docker worker runs on Linux with prefork, so it has both.

# quality (run before every commit)
uv run ruff check . --fix && uv run ruff format .
uv run mypy app
uv run pytest -q --cov=app

# migrations
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head

# manual extraction smoke test (scripts/try_extraction.py: run by hand, never by CI or app code).
# Reads jobs from DATABASE_URL, read-only (the connection refuses writes); writes nothing.
uv run python -m scripts.try_extraction                      # rules only, no network (default)
uv run python -m scripts.try_extraction --source-id <uuid> --limit 100
# A real provider call needs BOTH flags. N is mandatory and capped at 10 in the script. The run
# lists the job ids and sizes and asks for confirmation (--yes skips the question), stops at the
# budget or on the first 429 or rejected request (never retries), sends only posts that need a
# request and pass the length gate (emails/phones masked), and saves a run record without post
# text or key under .local/ (git-ignored). Needs LLM_PROVIDER, LLM_MODEL_EXTRACTION and
# GEMINI_API_KEY in .env; it turns the kill switch on for that process only, never in .env.
uv run python -m scripts.try_extraction --allow-provider --max-requests 3
# Accuracy evaluation (section 6, "Extraction accuracy evaluation" has the pass rule). Label first,
# blind: N=30 posts (--n) of ONE source with distinct text, a fixed --seed gives the same pick again.
# It never shows what the system extracted, reads the database read-only (same guard as above) and
# saves .local/eval/labels.json after every post, so you can /quit and run it again to resume.
# Conventions are printed in the script's header: blank = the post does not say; a single salary
# amount is min = max; list every technology the post names.
uv run python -m scripts.eval_label --source-name "My Feed" --n 30 --seed 1
# Optional one-off fix of the labels from the post text alone (title/company from the headline,
# seniority, salary, stack). Reads only each post's id and text (read-only, same guard), never an
# extraction, no provider request. Copies the file to labels.clean-v1.json first (never overwritten),
# prints every change and asks "Write these changes? [y/N]" before writing.
# --dry-run prints the same table and writes nothing (no backup, no question). Optional per-post
# overrides in .local/eval/label_overrides.json (git-ignored, keyed by the post's position: title,
# company, stack, seniority, or "keep": [fields] to leave a field as labelled) win over the rules.
uv run python -m scripts.eval_fix_labels
# Then score: the stored extraction and the rules recomputed offline, against your labels. It reads
# the database, makes ZERO provider requests (it refuses --allow-provider and builds no client) and
# writes a run record without post text or labels under .local/eval/runs/.
uv run python -m scripts.try_extraction --eval .local/eval/labels.json
# Second pass over the labels (rules in the script's docstring and in section 6). Same read-only
# guard, id and text only, no provider request. --dry-run prints the change table and writes nothing;
# without it the script shows the table, asks "Write these changes? [y/N]", copies the labels to
# labels.pass1.json (never overwritten) and then writes. Score the first-pass labels first.
uv run python -m scripts.eval_complete_labels --dry-run
# One-off after a rules version change: reset old rules-only rows (never rows the model touched) so
# the extraction sweep recomputes them. It refuses to write without --confirm-backup-taken (its
# --help shows the pg_dump commands and says to check the file size), only touches a local database
# named jobradar, resets in batches (logged) and leaves extraction_retries alone. --dry-run prints
# counts only and needs no flag. Beat and a worker must run for the sweep to recompute the rows.
uv run python -m scripts.recompute_extractions --dry-run
# Connectivity ping: ONE real request with fixed text (no database, no stored post) to check the key,
# endpoint and request shape. It shows the body field names and asks for confirmation. Levels:
# 1 = model + input + generation_config; 2 = + system_instruction + store=false; 3 = + the JSON-schema
# response_format with the real prompt and a fictional job post. Use the lowest level first.
uv run python -m scripts.try_extraction --ping --ping-level 1 --allow-provider
# The script prints the database it reads (host:port/name, never the password) and refuses any
# database not named `jobradar` or without a `jobs` table (so never the test database).
# The Compose db publishes no port. To reach it from the host, create a LOCAL, git-ignored
# docker-compose.override.yml (a pre-commit hook rejects it even with `git add -f`):
#   services:
#     db:
#       ports: ["127.0.0.1:5434:5432"]
# then `docker compose up -d db` recreates only the db service (data stays in its volume) and:
#   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5434/jobradar
#   uv run python -m scripts.try_extraction
# Compose merges the override into every `docker compose` command, so delete the file to go back.
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
# How often beat syncs every active source (minimum 5, default 30).
INGEST_INTERVAL_MINUTES=30
# Tasks one Celery worker runs at once (1-16, default 2). docker compose passes it to
# `--concurrency`. Ignored by the solo pool (-P solo), which always runs one task at a time.
CELERY_WORKER_CONCURRENCY=2
# none = rules only (default, no network). gemini | fake (development only) | ollama (reserved).
LLM_PROVIDER=none
# Kill switch. Off by default; on requires a provider, a model and (for gemini) a key.
LLM_EXTRACTION_ENABLED=false
# No default: model names change, so look up a current one in the provider's docs. For Gemini
# both "gemini-3.8-flash" and "models/gemini-3.8-flash" work: the adapter sends the bare id.
LLM_MODEL_EXTRACTION=
# A free key from Google AI Studio. Leave billing off for that project. Never commit it.
GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
# low | medium | high; empty leaves it out (the model's own default: medium on gemini-3.8-flash,
# which was too slow for one request). gemini-3.8-flash rejects "minimal", so it is not allowed.
GEMINI_THINKING_LEVEL=low
# One provider request (at most 25 s: a run makes at most 2 and must fit Celery's 60 s soft limit).
LLM_REQUEST_TIMEOUT_SECONDS=20
# Includes the model's thinking tokens: too small a value cuts the answer off.
LLM_MAX_OUTPUT_TOKENS=2048
# Characters of a post sent to a provider (longer posts keep their start and end).
LLM_MAX_INPUT_CHARS=12000
# A post shorter than this skips the provider and stays rules-only (0 = off). 500 comes from 55
# real posts: every headline-only stub was 172-268 characters, every real post at least 1,945.
LLM_MIN_POST_CHARS=500
# Budgets in provider REQUESTS, counted from the llm_usage table (rolling 24 hours per user and
# overall, and the last minute overall). Limits depend on the model and on your project: read your
# real ones in the provider's console and change these deliberately. For reference,
# gemini-3.5-flash-lite showed 15 per minute and 500 per day on the free tier; these defaults stay
# well below that. (The give-up setting below is still planned: the sweep that uses it is not built.)
LLM_DAILY_REQUEST_CAP_PER_USER=100
LLM_DAILY_REQUEST_CAP_GLOBAL=150
LLM_REQUESTS_PER_MINUTE=5
# A job still waiting for the LLM after this many days keeps its rules data and is skipped.
LLM_ENRICHMENT_GIVE_UP_DAYS=7
# CV upload: caps per upload (bytes, PDF pages checked before any text is read, characters kept),
# the hard time limit for the PDF-reading process (it is killed after this many seconds), and
# uploads per user per window.
CV_MAX_FILE_BYTES=2097152
CV_MAX_PAGES=10
CV_MAX_CHARS=30000
CV_PDF_TIMEOUT_SECONDS=10
CV_PDF_MEMORY_LIMIT_MB=512
CV_RATE_LIMIT_ATTEMPTS=5
CV_RATE_LIMIT_WINDOW_SECONDS=60
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
7. **Security:** never log secrets or full CV text; validate all input; use parameterized queries only; rate-limit auth endpoints. **Never send a CV or other personal data to a hosted free-tier LLM provider** (section 7).
8. **Keep diffs small and readable** so the owner can review every line.
9. When finishing a task, summarize: what changed, how to run it, what's tested, what's next.

---

## 14. Build phases (roadmap)

| Phase | Scope | Done when |
|---|---|---|
| 1 | Skeleton: FastAPI app, config, Postgres models, Alembic, Docker Compose, health endpoint, CI | `docker compose up` works; CI green |
| 2 | Auth + sources CRUD + ingestion (RSS first, then Telegram) with dedupe | Jobs appear in DB from a live source |
| 3 | Rules-first extraction with an optional free-tier LLM pass, validation, request budgets and fixtures | Pipeline proven with fake providers in tests; accuracy on a labeled sample (≥ 90% target) measured with the manual `--eval` script, since CI never calls a provider |
| 4 | CV upload/parsing, embeddings, pgvector matching and scoring with reasons | Ranked job list per user |
| 5 | Application tracker (status pipeline + events) and email draft generator | Full lifecycle via API |
| 6 | Next.js dashboard: filters, score badges, kanban board, CV page | Usable UI against the live API |
| 7 | Hardening: rate limits, caching, observability, README with architecture diagram + demo GIF, deploy | Public live demo |

**Deferred: Telegram ingestion.** It needs a Telegram API application and a user session, and the owner has only a personal account for now. The `SourceFetcher` interface and the `telegram` source type are in place so it can be added later without touching the rest of the pipeline. Phase 2 is otherwise complete.

**Current phase:** 4 — CV, embeddings and matching (no LLM is used on a CV). Plan in 6 PRs: (1) `feat/cv-storage`: CV upload (`PUT/GET/DELETE /api/v1/cv`, PDF or plain text as the raw request body, PDFs read in a killable subprocess), the privacy tests; (2) `feat/cv-skills-preferences`: CV skills (extracted, then editable), suggestions, job preferences (built, see "CV skills, suggestions and job preferences"); (3) embeddings with fastembed (with a memory gate); (4) scoring; (5) match API and caching; (6) evaluation and demo. Phase 3 (below) is finished and merged.

**Phase 3 summary:** rules-first extraction. Done: rules, merge, schema, provider-agnostic client with the Gemini adapter, the manual `scripts/try_extraction.py` (its `--ping` levels 1-3 were run against the real Gemini API by the owner with gemini-3.5-flash-lite, level 3 validated against `JobExtraction`). The DB state (extraction columns, `llm_usage`, migration), the service's rules phase (hash reuse, compare-and-set writes), the model path (request budgets, throttle, reservation rows, shared cooldown, validation retry, stack grounding), the Celery task `extract_job`, the post-ingest hook, the extraction sweep and `POST /jobs/{id}/extract` are built and tested with fakes only; the kill switch is off, so no real provider has been asked by the pipeline. The accuracy evaluation tooling is built (`scripts/eval_label.py` to label blind, `try_extraction.py --eval` to score, the pass rule in section 6); 30 posts have been labelled and scored (numbers and run records in `docs/ACCURACY.md`): the 90% target was not met, and `rules_v2` came out of the diagnosis of that run. `scripts/recompute_extractions.py` resets old rules-only rows so the sweep redoes them, and `scripts/eval_complete_labels.py` completes the labels in a second pass. Next: the owner takes a backup, runs the recompute, scores the first-pass labels again and then the completed ones, fills the table in `docs/ACCURACY.md`, and decides when to turn the kill switch on (a `.env` edit) and compares; an optional read-only change to hide `not_a_job` in `GET /jobs`. The Ollama adapter comes on request.

---

## 15. Definition of done

- [ ] Feature works end to end and is covered by tests
- [ ] `ruff`, `mypy`, `pytest` all pass locally and in CI
- [ ] Migration included (if the schema changed) and reviewed
- [ ] Docs updated (README / docstrings / this file if conventions changed)
- [ ] No secrets, debug prints or commented-out code
- [ ] Owner can explain how it works
