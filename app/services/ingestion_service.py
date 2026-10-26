import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.base import FetchError, RawJob, SourceFetcher
from app.ingestion.rss import RssFetcher
from app.ingestion.text import MAX_TEXT_CHARS
from app.models import ExtractionStatus, Job, Source, SourceType

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500  # the sources.last_error column size


@dataclass(frozen=True)
class IngestionSummary:
    fetched: int  # posts the source returned
    inserted: int  # new jobs stored
    skipped: int  # posts not stored: already known, repeated inside the feed, or empty
    error: str | None  # short sanitized reason if the sync failed, else None


def default_fetchers() -> dict[SourceType, SourceFetcher]:
    """One fetcher per supported source type. Telegram is added here in a later step."""
    return {SourceType.RSS: RssFetcher()}


async def ingest_source(
    session: AsyncSession,
    source: Source,
    fetchers: Mapping[SourceType, SourceFetcher] | None = None,
    now: datetime | None = None,
) -> IngestionSummary:
    """Fetch one source and store its new posts as jobs. Never raises for a failing source.

    Safe to run again at any time (idempotent): a post is identified by `external_id`, and the
    insert skips posts that already exist for this source (ON CONFLICT DO NOTHING), so re-runs,
    retries and overlapping runs cannot create duplicates.

    A failure (timeout, oversize, bad XML, unsafe URL, HTTP error, ...) is recorded in
    `source.last_error` as a short sanitized message and returned in the summary; it does not
    raise, so one broken source cannot affect the others. A success clears `last_error`.
    Database errors are not caught: they are infrastructure problems, not a source's fault.

    Two syncs of the same source at once are safe: both may fetch, but the unique constraint
    plus ON CONFLICT DO NOTHING means each post is stored exactly once (by whichever insert
    gets there first), and rows are inserted in a fixed order so the two cannot deadlock.
    """
    now = now or datetime.now(UTC)
    # End the transaction the caller's lookups started *before* waiting on the network, so no
    # database connection sits "idle in transaction" for up to ~20 s while a slow feed answers.
    # (The session keeps `source` loaded: expire_on_commit is off.) Writing starts a new one.
    await session.commit()
    fetcher = (fetchers if fetchers is not None else default_fetchers()).get(source.type)
    try:
        if fetcher is None:
            raise FetchError("Unsupported source type")
        result = await fetcher.fetch(source)
    except FetchError as error:
        return await _record_failure(session, source, str(error))
    except Exception:  # last resort for a fetcher bug; the real cause goes to the log only
        logger.exception("Unexpected error while fetching source %s", source.id)
        return await _record_failure(session, source, "Unexpected error while fetching")

    inserted = await _insert_jobs(session, source.id, result.jobs)
    source.last_synced_at = now
    source.last_error = None
    if not result.not_modified:  # on a 304 we keep the validators we already have
        source.etag = result.etag
        source.last_modified = result.last_modified
    await session.commit()
    return IngestionSummary(
        fetched=result.fetched,
        inserted=inserted,
        skipped=max(result.fetched - inserted, 0),
        error=None,
    )


async def _record_failure(session: AsyncSession, source: Source, message: str) -> IngestionSummary:
    source.last_error = message[:MAX_ERROR_LENGTH]
    await session.commit()
    return IngestionSummary(fetched=0, inserted=0, skipped=0, error=source.last_error)


async def _insert_jobs(session: AsyncSession, source_id: uuid.UUID, jobs: list[RawJob]) -> int:
    """Insert the posts that are new for this source and return how many that was."""
    if not jobs:
        return 0
    rows = [
        {
            "id": uuid.uuid4(),
            "source_id": source_id,
            "external_id": job.external_id,
            # Capped here too, so no fetcher (RSS today, Telegram later) can store a huge text.
            "raw_text": job.raw_text[:MAX_TEXT_CHARS],
            "posted_at": job.posted_at,
            "extraction_status": ExtractionStatus.PENDING,
        }
        # A fixed order means two overlapping syncs take their row locks in the same order,
        # which rules out a deadlock between them.
        for job in sorted(jobs, key=lambda j: j.external_id)
    ]
    statement = (
        insert(Job)
        .values(rows)
        # (source_id, external_id) is unique: an existing post is silently skipped, not an error.
        .on_conflict_do_nothing(index_elements=["source_id", "external_id"])
        .returning(Job.id)
    )
    return len((await session.execute(statement)).scalars().all())  # RETURNING lists new rows only
