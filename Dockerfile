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

# ---- Stage 2: the embedding model's weights (only when asked for) ---------------------------
# EMBEDDING_MODEL_DOWNLOAD=1 downloads the pinned revision of the model from Hugging Face into
# /models, so the running containers never need the internet. 0 (the default, used by the CI
# build) downloads nothing and /models stays empty: CI must never contact Hugging Face.
# The revision is a commit hash, not a branch, so the build always gets the same bytes. Keep it
# equal to EMBEDDING_MODEL_REVISION in app/core/config.py (a test checks). fastembed maps the name
# BAAI/bge-small-en-v1.5 to this repo, an int8 build of the same model (about 66 MB).
FROM builder AS model
ARG EMBEDDING_MODEL_DOWNLOAD=0
ARG EMBEDDING_MODEL_REPO=Qdrant/bge-small-en-v1.5-onnx-Q
ARG EMBEDDING_MODEL_REVISION=aa8f8b060edb00e03bfdd08813a2949946c8ba55
RUN mkdir /models \
    && if [ "$EMBEDDING_MODEL_DOWNLOAD" = "1" ]; then \
         REPO="$EMBEDDING_MODEL_REPO" REV="$EMBEDDING_MODEL_REVISION" /app/.venv/bin/python -c \
         "import os; from huggingface_hub import snapshot_download; \
          snapshot_download(os.environ['REPO'], revision=os.environ['REV'], cache_dir='/models')" \
         && test -d "/models/models--$(echo "$EMBEDDING_MODEL_REPO" | sed 's|/|--|')/snapshots/$EMBEDDING_MODEL_REVISION"; \
       fi \
    && chmod -R a+rX /models

# ---- Stage 3: the small runtime image ----------------------------------------------------
FROM python:3.12-slim

# Python writes no .pyc files at runtime (the non-root user can't write into /app anyway)
# and prints logs immediately instead of buffering them.
#
# Offline by default: nothing in the running containers downloads a model (HF_HUB_OFFLINE=1), and
# fastembed reads the weights from /models (FASTEMBED_CACHE_PATH).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=1 \
    FASTEMBED_CACHE_PATH=/models

# The app runs as an unprivileged user, so a bug in it cannot act as root in the container.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# The venv must live at the same path as in the builder stage (/app/.venv).
COPY --from=builder /app/.venv /app/.venv
COPY --from=model /models /models
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app

USER app

EXPOSE 8000

# Apply migrations, then start the API. `exec` makes uvicorn PID 1 so it receives
# `docker stop` signals and shuts down cleanly.
# --no-proxy-headers: ignore X-Forwarded-For, so clients cannot choose their own IP (the rate
# limiter keys on it). Behind a real reverse proxy, enable --proxy-headers together with
# --forwarded-allow-ips=<the proxy's address> explicitly.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:create_app --factory --no-proxy-headers --host 0.0.0.0 --port 8000"]
