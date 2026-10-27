import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from enum import StrEnum

from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.ingestion.base import SourceFetcher
from app.models import Source, SourceType
from app.services import auth_service, ingestion_service
from app.tasks.celery_app import HARD_TIME_LIMIT_SECONDS, SOFT_TIME_LIMIT_SECONDS, celery_app
from app.tasks.locking import SourceLocks, redis_locks_for

logger = logging.getLogger(__name__)

# Fan-out: the n-th source's task is delayed by n * step seconds, so a big batch of syncs does
# not start at the same instant. Spacing 1 s apart, but never spreading wider than the window.
SPREAD_STEP_SECONDS = 1.0
SPREAD_WINDOW_SECONDS = 30.0

PURGE_SOFT_TIME_LIMIT_SECONDS = 240  # a big purge may take a while; it commits in batches
PURGE_HARD_TIME_LIMIT_SECONDS = 300

# Failures worth retrying: Redis or the database being briefly unreachable. Anything else (a bug)
# is not retried, because running it again would just fail the same way.
RETRYABLE_ERRORS = (RedisError, OperationalError, InterfaceError)


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
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        await engine.dispose()


# --- ingest one source -----------------------------------------------------------------------


def run_ingest_source(
    source_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    fetchers: Mapping[SourceType, SourceFetcher] | None = None,
    locks: SourceLocks | None = None,
) -> IngestOutcome:
    """Sync one source, unless another run of it is in progress. Safe to run again any time.

    The keyword arguments exist so tests can pass fakes; a task call uses the real ones.
    """
    settings = settings or get_settings()
    locks = locks or redis_locks_for(settings.redis_url)
    if not locks.acquire(source_id):
        logger.info("Skipping source %s: another run holds its lock", source_id)
        return IngestOutcome.SKIPPED_LOCKED
    try:
        return asyncio.run(_ingest(source_id, settings, fetchers))
    finally:
        locks.release(source_id)  # also when the sync raised (or hit its time limit)


async def _ingest(
    source_id: uuid.UUID,
    settings: Settings,
    fetchers: Mapping[SourceType, SourceFetcher] | None,
) -> IngestOutcome:
    async with _session(settings) as session:
        # Load the source here, not in the API: it may have been changed or deleted since the
        # task was queued.
        source = await session.get(Source, source_id)
        if source is None:
            logger.info("Skipping source %s: it no longer exists", source_id)
            return IngestOutcome.SKIPPED_MISSING
        summary = await ingestion_service.ingest_source(session, source, fetchers)
        logger.info(
            "Synced source %s: fetched=%d inserted=%d skipped=%d error=%s",
            source_id,
            summary.fetched,
            summary.inserted,
            summary.skipped,
            summary.error,  # a short sanitized message, safe to log
        )
        return IngestOutcome.INGESTED


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
