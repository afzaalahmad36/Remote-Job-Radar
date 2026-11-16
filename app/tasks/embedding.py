"""Celery tasks that compute embeddings. They run in the `worker-embed` service only.

`embed_job` and `embed_cv` go to the `embed` queue (see `celery_app.EMBED_QUEUE`), which only that
service reads, so the model is loaded in one small process and nowhere else. This module never
imports the model: `build_embedder` (called inside a task, in that service) does, and only when
`EMBEDDING_PROVIDER` asks for it. The API and the main worker import this module just to publish
task names, and stay free of onnxruntime (a test imports them in a clean interpreter and checks).

Everything the tasks do is safe to repeat: the service skips unchanged text and writes with a
compare-and-set. The embedding path imports no LLM client and no HTTP library (a test checks it)
because CV text goes through it.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.matching.embeddings import Embedder
from app.matching.factory import build_embedder
from app.services import embedding_service
from app.services.embedding_service import EmbedOutcome
from app.tasks.celery_app import (
    HARD_TIME_LIMIT_SECONDS,
    SOFT_TIME_LIMIT_SECONDS,
    celery_app,
)
from app.tasks.common import ENQUEUE_ERRORS, RETRYABLE_ERRORS, session_factory

logger = logging.getLogger(__name__)

DISABLED = "disabled"  # EMBEDDING_PROVIDER=none: nothing is embedded

SWEEP_SOFT_TIME_LIMIT_SECONDS = 120
SWEEP_HARD_TIME_LIMIT_SECONDS = 150

# The model is loaded once per worker process, the first time a task needs it, and kept: loading
# is cheap (about 0.1 s from the page cache) but the weights are 65 MB. An injected embedder (a
# test's fake) takes precedence.
_cached: dict[str, Embedder] = {}


def get_embedder(settings: Settings) -> Embedder | None:
    """The embedder for this process (built on first use), or None when embeddings are off."""
    if settings.embedding_provider == "none":
        return None
    key = settings.embedding_model_id
    if key not in _cached:
        built = build_embedder(settings)
        if built is None:
            return None
        _cached[key] = built
    return _cached[key]


# --- publishing -------------------------------------------------------------------------------


def enqueue_embed_job(job_id: uuid.UUID, delay: float = 0.0) -> None:
    """Queue the embedding of one job, unless embeddings are off. May raise ENQUEUE_ERRORS."""
    if get_settings().embedding_provider == "none":
        return
    embed_job.apply_async(args=[str(job_id)], countdown=delay)


def enqueue_embed_cv(user_id: uuid.UUID) -> None:
    """Queue the embedding of a user's CV, unless embeddings are off. May raise ENQUEUE_ERRORS."""
    if get_settings().embedding_provider == "none":
        return
    embed_cv.apply_async(args=[str(user_id)])


# --- embed one job -----------------------------------------------------------------------------


def run_embed_job(
    job_id: uuid.UUID, *, settings: Settings | None = None, embedder: Embedder | None = None
) -> str:
    """Embed one job and say what happened (`embedded`, `unchanged`, `copied`, ...)."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    if embedder is None:
        return DISABLED
    return asyncio.run(_embed_job(settings, job_id, embedder)).value


async def _embed_job(settings: Settings, job_id: uuid.UUID, embedder: Embedder) -> EmbedOutcome:
    async with session_factory(settings) as factory:
        return await embedding_service.embed_job(factory, job_id, embedder)


@celery_app.task(
    name="embed_job",
    soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    time_limit=HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def embed_job(job_id: str) -> str:
    """Celery task: compute and store the embedding of one extracted job."""
    return run_embed_job(uuid.UUID(job_id))


# --- embed the CV ------------------------------------------------------------------------------


def run_embed_cv(
    user_id: uuid.UUID, *, settings: Settings | None = None, embedder: Embedder | None = None
) -> str:
    """Embed a user's CV and say what happened."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    if embedder is None:
        return DISABLED
    return asyncio.run(_embed_cv(settings, user_id, embedder)).value


async def _embed_cv(settings: Settings, user_id: uuid.UUID, embedder: Embedder) -> EmbedOutcome:
    async with session_factory(settings) as factory:
        return await embedding_service.embed_cv(factory, user_id, embedder)


@celery_app.task(
    name="embed_cv",
    soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    time_limit=HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def embed_cv(user_id: str) -> str:
    """Celery task: compute and store the embedding of one user's CV."""
    return run_embed_cv(uuid.UUID(user_id))


# --- the sweep ---------------------------------------------------------------------------------


def run_sweep_embeddings(
    *,
    settings: Settings | None = None,
    enqueue_job: Callable[[uuid.UUID], None] | None = None,
    enqueue_cv: Callable[[uuid.UUID], None] | None = None,
) -> dict[str, int]:
    """One pass of the safety net: queue what has no vector, or a stale one.

    It reads the database and publishes messages; it does not load the model, so the main worker
    runs it. If the queue is down it logs once and stops (the next sweep, five minutes later, is
    the retry: there is no loop here).
    """
    settings = settings or get_settings()
    if settings.embedding_provider == "none":
        return {"jobs_due": 0, "cvs_due": 0, "queued": 0}
    work = asyncio.run(_collect(settings))
    publish_job = enqueue_job or enqueue_embed_job
    publish_cv = enqueue_cv or enqueue_embed_cv
    queued = 0
    try:
        for job_id in work.job_ids:
            publish_job(job_id)
            queued += 1
        for user_id in work.cv_user_ids:
            publish_cv(user_id)
            queued += 1
    except ENQUEUE_ERRORS:
        due = len(work.job_ids) + len(work.cv_user_ids)
        logger.warning(
            "The queue is unavailable: %d of %d embedding tasks were not queued; "
            "the next sweep will pick them up",
            due - queued,
            due,
        )
    return {"jobs_due": len(work.job_ids), "cvs_due": len(work.cv_user_ids), "queued": queued}


async def _collect(settings: Settings) -> embedding_service.SweepWork:
    async with session_factory(settings) as factory:
        return await embedding_service.collect_work(factory, model_id=settings.embedding_model_id)


@celery_app.task(
    name="sweep_embeddings",
    soft_time_limit=SWEEP_SOFT_TIME_LIMIT_SECONDS,
    time_limit=SWEEP_HARD_TIME_LIMIT_SECONDS,
)
def sweep_embeddings() -> dict[str, int]:
    """Celery task, run by beat every five minutes: the safety net for embeddings."""
    return run_sweep_embeddings()
