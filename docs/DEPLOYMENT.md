# Deployment checklist

Work through this before the first deploy of a new environment, and again when a migration adds a
requirement (this file grows with the project). Items that are already automatic in
`docker compose` say so; the point of the list is what is **not** automatic on another host.

## 1. Database (PostgreSQL 16 with pgvector)

- [ ] **The role that runs migrations can create the `vector` extension.** Migration
      `c3d1a7e95b20` runs `CREATE EXTENSION IF NOT EXISTS vector`. pgvector is **not** a
      "trusted" extension, so an ordinary role fails with
      `permission denied to create extension "vector"` and the migration (and with it the API
      container's start) stops. Either run the migrations as a superuser (local Compose and CI do:
      the `postgres` user), or have an administrator run this once on the target database, before
      the first migration:

      ```sql
      CREATE EXTENSION IF NOT EXISTS vector;
      ```

      Managed Postgres services differ in who may do this and in which pgvector version they
      ship, so check on the real instance rather than assuming:

      ```sql
      SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name = 'vector';
      ```

      Expect a row (it is tested against pgvector 0.8.7 in `pgvector/pgvector:pg16`). No row means
      the host does not ship pgvector and this application cannot run there.
- [ ] After restoring a `pg_dump` backup into a fresh database, create the extension first (a
      plain dump of the tables does not create it) and keep the backup **outside the repository**:
      the `cvs` table holds CV text.
- [ ] A downgrade past `c3d1a7e95b20` runs `DROP EXTENSION IF EXISTS vector`. Do not downgrade a
      database that other applications share.

## 2. Secrets and configuration

- [ ] `ENV=production`, and a `SECRET_KEY` of at least 32 random characters (the app refuses to
      start otherwise).
- [ ] `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=fake` are refused in production (by design).
- [ ] `CORS_ORIGINS` lists the real front-end origin only.
- [ ] `.env` is never committed. `LLM_EXTRACTION_ENABLED` stays `false` until you mean to ask a
      provider (a deliberate edit).

## 3. Embeddings (the `worker-embed` service)

- [ ] **Build the deployed image with the model baked in:**
      `docker build --build-arg EMBEDDING_MODEL_DOWNLOAD=1 .` (Compose does this by default). The
      CI image is built with `0` and cannot embed: never deploy it for `EMBEDDING_PROVIDER=fastembed`.
- [ ] `EMBEDDING_MODEL_REVISION` is the same commit hash as the Dockerfile's
      `EMBEDDING_MODEL_REVISION` (a test checks the defaults agree). Changing the model or the
      revision re-embeds everything: the sweep does it gradually, because each vector stores
      `model@revision` and a vector from another one counts as stale.
- [ ] A model with a different vector length needs a new migration (`vector(384)` is fixed).
- [ ] `worker-embed` runs **one** task at a time (`--concurrency=1`) and is given about 512 MB. The
      memory it needs was measured at about 330 MB; see the PR for the numbers.
- [ ] `worker-embed` has **no route to the internet**: in Compose it is on an `internal: true`
      network with only the database and Redis. On another platform, give it the same property
      (no outbound access); it needs none, because the weights are in the image and
      `HF_HUB_OFFLINE=1`.
- [ ] The main worker is started with `--queues=celery` and `worker-embed` with `--queues=embed`.
      Only `worker-embed` reads the `embed` queue; if no `worker-embed` runs, embedding tasks wait
      in Redis (nothing is lost, nothing is embedded).
- [ ] Exactly one `beat`.

## 4. Process limits (known, documented)

- [ ] The auth, sync and CV-upload rate limiters and the 2-PDF slot limit live in the API process:
      run one API process, or divide the limits by the number of processes.
- [ ] PDF reading memory cap (`CV_PDF_MEMORY_LIMIT_MB`) is enforced on Linux only.
