# syntax=docker/dockerfile:1

# ---- Stage 1: build the virtualenv -------------------------------------------------------
FROM python:3.12-slim AS builder

# uv is copied from its official image (pin the version so builds are reproducible).
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

# UV_PYTHON_DOWNLOADS=never: use this image's Python 3.12, don't download another.
# UV_COMPILE_BYTECODE=1: precompile .pyc files so the app starts faster.
# UV_LINK_MODE=copy: real copies (hardlinks into uv's cache won't survive into the next stage).
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Only the dependency files come first, so Docker reuses this (slow) layer until they change.
COPY pyproject.toml uv.lock .python-version ./
# --locked: fail if uv.lock is out of date. --no-dev: leave out pytest, ruff, mypy, ...
RUN uv sync --locked --no-dev

# ---- Stage 2: the small runtime image ----------------------------------------------------
FROM python:3.12-slim

# Python writes no .pyc files at runtime (the non-root user can't write into /app anyway)
# and prints logs immediately instead of buffering them.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# The app runs as an unprivileged user, so a bug in it cannot act as root in the container.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# The venv must live at the same path as in the builder stage (/app/.venv).
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app

USER app

EXPOSE 8000

# Apply migrations, then start the API. `exec` makes uvicorn PID 1 so it receives
# `docker stop` signals and shuts down cleanly.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000"]
