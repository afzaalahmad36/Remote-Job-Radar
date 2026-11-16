import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.extraction.limits import (
    EXTRACTION_HARD_TIME_LIMIT_SECONDS,
    EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
)
from app.ingestion.base import SourceFetcher
from app.llm.base import LLMClient
from app.llm.client import build_client
from app.models import Source, SourceType
from app.services import auth_service, extraction_service, extraction_sweep, ingestion_service
from app.services.cooldown import Cooldown
from app.services.extraction_types import ExtractionOutcome, ExtractionResult
from app.tasks.celery_app import HARD_TIME_LIMIT_SECONDS, SOFT_TIME_LIMIT_SECONDS, celery_app
from app.tasks.common import ENQUEUE_ERRORS, RETRYABLE_ERRORS
from app.tasks.common import session_factory as _session_factory
from app.tasks.cooldown import redis_cooldown_for
from app.tasks.embedding import enqueue_embed_job
from app.tasks.locking import SourceLocks, redis_job_locks_for, redis_locks_for

logger = logging.getLogger(__name__)

# Fan-out: the n-th source's task is delayed by n * step seconds, so a big batch of syncs does
# not start at the same instant. Spacing 1 s apart, but never spreading wider than the window.
SPREAD_STEP_SECONDS = 1.0
SPREAD_WINDOW_SECONDS = 30.0

# The sweep reads the database and publishes up to 200 messages. No automatic retry: if it fails,
# the next scheduled sweep (five minutes later) does the same work.
SWEEP_SOFT_TIME_LIMIT_SECONDS = 120
SWEEP_HARD_TIME_LIMIT_SECONDS = 150

PURGE_SOFT_TIME_LIMIT_SECONDS = 240  # a big purge may take a while; it commits in batches
PURGE_HARD_TIME_LIMIT_SECONDS = 300

# ENQUEUE_ERRORS and RETRYABLE_ERRORS live in `app.tasks.common` (shared with the embedding tasks)
# and are imported above, so `from app.tasks.jobs import RETRYABLE_ERRORS` keeps working.


Enqueue = Callable[[uuid.UUID, float], None]  # (job id, delay in seconds)

# Outcomes of `extract_job` after which there is nothing to embed: no job, no usable result, or
# another run holds the job (that run queues the embedding itself).
NO_EMBEDDING_AFTER = frozenset(
    {
        ExtractionOutcome.NOT_FOUND.value,
        ExtractionOutcome.FAILED.value,
        ExtractionOutcome.NOT_A_JOB.value,
        ExtractionOutcome.LOST_RACE.value,
        "skipped_locked",
    }
)


class IngestOutcome(StrEnum):
    INGESTED = "ingested"
    SKIPPED_LOCKED = "skipped_locked"  # another run of this source is in progress
    SKIPPED_MISSING = "skipped_missing"  # the source was deleted after the task was queued


@asynccontextmanager
async def _session(settings: Settings) -> AsyncIterator[AsyncSession]:
    """A database session on a brand-new engine that is thrown away afterwards.

    A Celery task is a normal function, so each run starts its own event loop with
    `asyncio.run`, and a loop is gone when `run` returns. The API's engine (`get_engine`) keeps
    open connections that belong to *its* loop, so using it here would break. NullPool means
    no connection outlives the task, and `dispose()` closes the engine before the loop ends.
    """
    async with _session_factory(settings) as factory, factory() as session:
        yield session


# --- ingest one source -----------------------------------------------------------------------


def run_ingest_source(
    source_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    fetchers: Mapping[SourceType, SourceFetcher] | None = None,
    locks: SourceLocks | None = None,
    enqueue_extraction: Enqueue | None = None,
) -> IngestOutcome:
    """Sync one source, unless another run of it is in progress. Safe to run again any time.

    Every post the sync stored is then handed to the extraction task. The keyword arguments exist
    so tests can pass fakes; a task call uses the real ones.
    """
    settings = settings or get_settings()
    locks = locks or redis_locks_for(settings.redis_url)
    if not locks.acquire(source_id):
        logger.info("Skipping source %s: another run holds its lock", source_id)
        return IngestOutcome.SKIPPED_LOCKED
    try:
        outcome, new_job_ids = asyncio.run(_ingest(source_id, settings, fetchers))
    finally:
        locks.release(source_id)  # also when the sync raised (or hit its time limit)
    enqueue_many(new_job_ids, enqueue_extraction or _enqueue_extraction)
    return outcome


async def _ingest(
    source_id: uuid.UUID,
    settings: Settings,
    fetchers: Mapping[SourceType, SourceFetcher] | None,
) -> tuple[IngestOutcome, tuple[uuid.UUID, ...]]:
    async with _session(settings) as session:
        # Load the source here, not in the API: it may have been changed or deleted since the
        # task was queued.
        source = await session.get(Source, source_id)
        if source is None:
            logger.info("Skipping source %s: it no longer exists", source_id)
            return IngestOutcome.SKIPPED_MISSING, ()
        summary = await ingestion_service.ingest_source(session, source, fetchers)
        logger.info(
            "Synced source %s: fetched=%d inserted=%d skipped=%d error=%s",
            source_id,
            summary.fetched,
            summary.inserted,
            summary.skipped,
            summary.error,  # a short sanitized message, safe to log
        )
        return IngestOutcome.INGESTED, summary.new_job_ids


@celery_app.task(
    name="ingest_source",
    soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    time_limit=HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def ingest_source(source_id: str) -> str:
    """Celery task: sync one source. A feed that fails is recorded on the source, not raised,
    so a bad feed never crashes or retries the worker."""
    return run_ingest_source(uuid.UUID(source_id)).value


# --- queueing extraction tasks -----------------------------------------------------------------


def _enqueue_extraction(job_id: uuid.UUID, delay: float) -> None:
    extract_job.apply_async(args=[str(job_id)], countdown=delay)


def enqueue_many(job_ids: Sequence[uuid.UUID], enqueue: Enqueue) -> int:
    """Queue an extraction task per job, spread over a few seconds. Returns how many were queued.

    The first failed publish ends it with a single warning and no retry here (the broker is
    down; trying 100 more times would only wait 100 more times). Jobs that were not queued are
    found again by the next sweep, so nothing is lost.
    """
    queued = 0
    for job_id, delay in zip(job_ids, spread_delays(len(job_ids)), strict=True):
        try:
            enqueue(job_id, delay)
        except ENQUEUE_ERRORS:
            logger.warning(
                "The queue is unavailable: %d of %d extraction tasks were not queued; "
                "the sweep will pick them up",
                len(job_ids) - queued,
                len(job_ids),
            )
            break
        queued += 1
    return queued


# --- extract one job ---------------------------------------------------------------------------


def run_extract_job(
    job_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    cooldown: Cooldown | None = None,
    locks: SourceLocks | None = None,
) -> str:
    """Run the extraction of one job, unless another run of it is in progress. Returns what
    happened as a short string (`extracted`, `deferred`, `skipped_locked`, ...).

    What happens to a provider failure is decided inside the service (the job stays pending
    behind a shared pause, or is given up on), so this function never asks Celery to retry it:
    only the database or Redis being unreachable raises. The keyword arguments exist so tests can
    pass fakes; a task call builds the real client and cooldown here.
    """
    settings = settings or get_settings()
    locks = locks or redis_job_locks_for(settings.redis_url)
    if not locks.acquire(job_id):
        logger.info("Skipping extraction of job %s: another run holds its lock", job_id)
        return "skipped_locked"
    try:
        return asyncio.run(_extract(job_id, settings, llm, cooldown)).outcome.value
    finally:
        locks.release(job_id)


async def _extract(
    job_id: uuid.UUID, settings: Settings, llm: LLMClient | None, cooldown: Cooldown | None
) -> ExtractionResult:
    client = llm if llm is not None else build_client(settings)  # None: provider off
    if client is not None and cooldown is None:
        cooldown = redis_cooldown_for(settings.redis_url, client.name)
    async with _session_factory(settings) as factory:
        result = await extraction_service.extract_job(
            factory, job_id, settings=settings, llm=client, cooldown=cooldown
        )
    if result.outcome is ExtractionOutcome.DEFERRED:
        logger.info(
            "Job %s deferred (%s), a later run can try in about %.0f s",
            job_id,
            result.deferral.value if result.deferral else "-",
            result.retry_after_seconds or 0,
        )
    return result


@celery_app.task(
    name="extract_job",
    soft_time_limit=EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
    time_limit=EXTRACTION_HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def extract_job(job_id: str) -> str:
    """Celery task: extract one job (rules first, then the model if it is needed and allowed).

    Afterwards the job is queued for embedding (a no-op when embeddings are off). The embedding
    run decides for itself whether there is anything to embed, so it is queued after every
    outcome that can leave an extracted job behind; a failed publish only logs, because the
    embedding sweep finds the job later.
    """
    outcome = run_extract_job(uuid.UUID(job_id))
    queue_embedding_after(uuid.UUID(job_id), outcome)
    return outcome


def queue_embedding_after(
    job_id: uuid.UUID, outcome: str, enqueue: Callable[[uuid.UUID], None] | None = None
) -> bool:
    """Queue the embedding of a job after an extraction run. True if it was queued.

    A failed publish only logs: the embedding sweep finds the job later.
    """
    if outcome in NO_EMBEDDING_AFTER:
        return False
    publish = enqueue or enqueue_embed_job
    try:
        publish(job_id)
    except ENQUEUE_ERRORS:
        logger.warning("Job %s was not queued for embedding; the sweep will", job_id)
        return False
    return True


# --- the sweep -----------------------------------------------------------------------------------


def run_sweep_extraction(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    cooldown: Cooldown | None = None,
    enqueue: Enqueue | None = None,
) -> dict[str, int]:
    """One pass of the extraction safety net; see `services/extraction_sweep.py` for the steps.

    The database steps run first. Then tasks are queued for jobs whose rules phase never ran, and
    for jobs waiting for the model if asking could succeed. If the queue is down, one warning is
    logged and nothing more is attempted; the next scheduled sweep tries again.
    """
    settings = settings or get_settings()
    client = llm if llm is not None else build_client(settings)
    if client is not None and cooldown is None:
        cooldown = redis_cooldown_for(settings.redis_url, client.name)
    work = asyncio.run(_collect(settings, cooldown if client is not None else None))
    queue = enqueue or _enqueue_extraction
    queued = enqueue_many([*work.rules_job_ids, *work.enrichment_job_ids], queue)
    return {
        "abandoned": work.abandoned,
        "gave_up": work.gave_up,
        "rules_due": len(work.rules_job_ids),
        "enrichment_due": len(work.enrichment_job_ids),
        "queued": queued,
    }


async def _collect(settings: Settings, cooldown: Cooldown | None) -> extraction_sweep.SweepWork:
    async with _session_factory(settings) as factory:
        return await extraction_sweep.collect_work(factory, settings=settings, cooldown=cooldown)


@celery_app.task(
    name="sweep_extraction",
    soft_time_limit=SWEEP_SOFT_TIME_LIMIT_SECONDS,
    time_limit=SWEEP_HARD_TIME_LIMIT_SECONDS,
)
def sweep_extraction() -> dict[str, int]:
    """Celery task, run by beat every five minutes: the safety net for extraction."""
    return run_sweep_extraction()


# --- fan out ---------------------------------------------------------------------------------


def spread_delays(count: int) -> list[float]:
    """Seconds to delay each of `count` tasks: 0, step, 2*step, ... (step shrinks if many)."""
    if count == 0:
        return []
    step = min(SPREAD_STEP_SECONDS, SPREAD_WINDOW_SECONDS / count)
    return [index * step for index in range(count)]


def run_ingest_all_active_sources(
    enqueue: Callable[[uuid.UUID, float], None], *, settings: Settings | None = None
) -> int:
    """Queue one sync per active source and return how many were queued.

    One task per source (instead of looping over the sources inside one task) means a slow or
    failing feed only occupies its own task, other workers pick up the rest in parallel, and a
    retry redoes one source instead of all of them. The enqueues are spread over a few seconds
    (see spread_delays) so the workers and the feeds' servers do not get everything at once.
    """
    settings = settings or get_settings()
    source_ids = asyncio.run(_active_source_ids(settings))
    for source_id, delay in zip(source_ids, spread_delays(len(source_ids)), strict=True):
        enqueue(source_id, delay)
    return len(source_ids)


async def _active_source_ids(settings: Settings) -> list[uuid.UUID]:
    async with _session(settings) as session:
        rows = await session.scalars(
            select(Source.id)
            .where(Source.is_active)
            # The sources that have waited longest (or never synced) go first.
            .order_by(Source.last_synced_at.asc().nulls_first(), Source.id)
        )
        return list(rows)


@celery_app.task(name="ingest_all_active_sources")
def ingest_all_active_sources() -> int:
    """Celery task, run by beat: queue a sync of every active source."""
    settings = get_settings()
    expires = settings.ingest_interval_minutes * 60  # a queued sync older than a run is stale

    def enqueue(source_id: uuid.UUID, delay: float) -> None:
        ingest_source.apply_async(args=[str(source_id)], countdown=delay, expires=expires)

    return run_ingest_all_active_sources(enqueue, settings=settings)


# --- housekeeping ----------------------------------------------------------------------------


def run_purge_expired_refresh_tokens(*, settings: Settings | None = None) -> int:
    return asyncio.run(_purge(settings or get_settings()))


async def _purge(settings: Settings) -> int:
    async with _session(settings) as session:
        deleted = await auth_service.purge_expired_refresh_tokens(session)
    logger.info("Purged %d expired refresh tokens", deleted)
    return deleted


@celery_app.task(
    name="purge_expired_refresh_tokens",
    soft_time_limit=PURGE_SOFT_TIME_LIMIT_SECONDS,
    time_limit=PURGE_HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    max_retries=3,
)
def purge_expired_refresh_tokens() -> int:
    """Celery task, run daily by beat: delete refresh tokens that expired over a week ago."""
    return run_purge_expired_refresh_tokens()
