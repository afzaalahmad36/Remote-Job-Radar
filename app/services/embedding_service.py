"""Computing and storing embeddings for jobs and for the user's CV.

Both functions follow the same plan, which is also what makes them safe to run twice:

1. Read what is to be embedded in one short transaction.
2. Hash the exact text (with the model's identity). Same hash as the stored one: nothing to do.
3. For a job: another job with the same text and model already has a vector: copy it.
4. Otherwise run the model with NO transaction open (it takes about 0.3 s on one thread).
5. Write the vector with a compare-and-set: if the job (or CV) changed while the model ran, the
   write matches zero rows and the stale vector is dropped; whoever changed it queued its own run.

The CV text goes through this module, so it imports no LLM client and no HTTP library (a test
checks it), and logs only counts, never text. The vectors are numbers computed on this machine.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.matching.embeddings import (
    CV_MAX_CHUNKS,
    Embedder,
    EmbeddingError,
    check_vector,
    cv_chunks,
    job_embedding_text,
    normalised_mean,
    text_hash,
)
from app.models import Cv, ExtractionStatus, Job

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


class CvEmbeddingQueue(Protocol):
    """What the API needs to ask for a CV to be embedded (satisfied by the Celery queue)."""

    async def enqueue_cv_embedding(self, user_id: uuid.UUID) -> None: ...


class EmbedOutcome(StrEnum):
    NOT_FOUND = "not_found"  # the job or CV no longer exists
    NOT_READY = "not_ready"  # a job that is not extracted has nothing to embed yet
    UNCHANGED = "unchanged"  # the stored vector already is for this exact text and model
    COPIED = "copied"  # copied from a job with identical text (job only)
    EMBEDDED = "embedded"  # the model ran and the vector was stored
    LOST_RACE = "lost_race"  # the row changed while the model ran; the vector was dropped


@asynccontextmanager
async def _transaction(factory: SessionFactory) -> AsyncIterator[AsyncSession]:
    """One short transaction on its own session; committed when the block ends."""
    async with factory() as session, session.begin():
        yield session


async def _embed_one(embedder: Embedder, text: str) -> list[float]:
    """One text, one vector. The model blocks the CPU, so it runs in a thread."""
    vectors = await asyncio.to_thread(embedder.embed, [text])
    if len(vectors) != 1:
        raise EmbeddingError("The embedder returned the wrong number of vectors")
    return check_vector(vectors[0])


# --- jobs --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _JobSnapshot:
    status: ExtractionStatus
    attempts: int
    extracted_at: object  # compared for equality only (the version of the extraction)
    text: str
    stored_hash: str | None
    stored_model: str | None
    has_vector: bool


async def _load_job(session: AsyncSession, job_id: uuid.UUID) -> _JobSnapshot | None:
    row = (
        await session.execute(
            select(
                Job.extraction_status,
                Job.extraction_attempts,
                Job.extracted_at,
                Job.title,
                Job.company,
                Job.seniority,
                Job.remote_policy,
                Job.stack,
                Job.raw_text,
                Job.embedding_text_hash,
                Job.embedding_model,
                Job.embedding.is_not(None),
            ).where(Job.id == job_id)
        )
    ).first()
    if row is None:
        return None
    text = job_embedding_text(
        title=row.title,
        company=row.company,
        seniority=row.seniority.value if row.seniority else None,
        remote_policy=row.remote_policy.value if row.remote_policy else None,
        stack=row.stack,
        raw_text=row.raw_text,
    )
    return _JobSnapshot(
        status=row.extraction_status,
        attempts=row.extraction_attempts,
        extracted_at=row.extracted_at,
        text=text,
        stored_hash=row.embedding_text_hash,
        stored_model=row.embedding_model,
        has_vector=bool(row[-1]),
    )


async def embed_job(factory: SessionFactory, job_id: uuid.UUID, embedder: Embedder) -> EmbedOutcome:
    """Make sure an extracted job has a vector for its current text. See the module docstring."""
    async with _transaction(factory) as session:
        seen = await _load_job(session, job_id)
    if seen is None:
        return EmbedOutcome.NOT_FOUND
    if seen.status is not ExtractionStatus.EXTRACTED:
        return EmbedOutcome.NOT_READY
    digest = text_hash(embedder.model_id, seen.text)

    if seen.has_vector and seen.stored_hash == digest and seen.stored_model == embedder.model_id:
        await _touch_job(factory, job_id, digest)
        return EmbedOutcome.UNCHANGED

    async with _transaction(factory) as session:
        donor = await _find_donor_vector(session, job_id, digest, embedder.model_id)
    outcome = EmbedOutcome.COPIED
    vector = donor
    if vector is None:
        vector = await _embed_one(embedder, seen.text)
        outcome = EmbedOutcome.EMBEDDED

    async with _transaction(factory) as session:
        stored = await session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.extraction_status == ExtractionStatus.EXTRACTED,
                Job.extraction_attempts == seen.attempts,
                Job.extracted_at.is_not_distinct_from(seen.extracted_at),
            )
            .values(
                embedding=vector,
                embedding_model=embedder.model_id,
                embedding_text_hash=digest,
                embedded_at=func.now(),
            )
            .returning(Job.id)
        )
        if stored.first() is None:
            logger.info("Embedding of job %s dropped: the job changed meanwhile", job_id)
            return EmbedOutcome.LOST_RACE
    logger.info("Job %s embedded (%s)", job_id, outcome.value)
    return outcome


async def _find_donor_vector(
    session: AsyncSession, job_id: uuid.UUID, digest: str, model_id: str
) -> list[float] | None:
    """The vector of another job that was embedded from the identical text by the same model."""
    found = await session.scalar(
        select(Job.embedding)
        .where(
            Job.embedding_text_hash == digest,
            Job.embedding_model == model_id,
            Job.embedding.is_not(None),
            Job.id != job_id,
        )
        .limit(1)
    )
    return None if found is None else check_vector(found)


async def _touch_job(factory: SessionFactory, job_id: uuid.UUID, digest: str) -> None:
    """Record that the vector was checked against the current extraction, so the sweep (which
    looks for `embedded_at < extracted_at`) does not queue the job again."""
    async with _transaction(factory) as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id, Job.embedding_text_hash == digest)
            .values(embedded_at=func.now())
        )


# --- the CV ------------------------------------------------------------------------------------


async def embed_cv(factory: SessionFactory, user_id: uuid.UUID, embedder: Embedder) -> EmbedOutcome:
    """Make sure the user's CV has a vector for its current text: the normalised mean of the
    vectors of its first `CV_MAX_CHUNKS` chunks (each chunk is about what the model can read)."""
    async with _transaction(factory) as session:
        row = (
            await session.execute(
                select(
                    Cv.content_hash,
                    Cv.text,
                    Cv.embedding_text_hash,
                    Cv.embedding_model,
                    Cv.embedding.is_not(None),
                ).where(Cv.user_id == user_id)
            )
        ).first()
    if row is None:
        return EmbedOutcome.NOT_FOUND
    seen_content_hash, text, stored_hash, stored_model, has_vector = row

    pieces = cv_chunks(text)
    if pieces.truncated:
        # Counts only: never any of the text.
        logger.info(
            "CV truncated for embedding: user=%s chunks=%d kept=%d",
            user_id,
            pieces.total,
            CV_MAX_CHUNKS,
        )
    digest = text_hash(embedder.model_id, "\n\x00\n".join(pieces.chunks))
    if has_vector and stored_hash == digest and stored_model == embedder.model_id:
        async with _transaction(factory) as session:
            await session.execute(
                update(Cv)
                .where(Cv.user_id == user_id, Cv.embedding_text_hash == digest)
                .values(embedded_at=func.now())
            )
        return EmbedOutcome.UNCHANGED

    vectors = [await _embed_one(embedder, chunk) for chunk in pieces.chunks]
    vector = normalised_mean(vectors)

    async with _transaction(factory) as session:
        stored = await session.execute(
            update(Cv)
            .where(Cv.user_id == user_id, Cv.content_hash == seen_content_hash)
            .values(
                embedding=vector,
                embedding_model=embedder.model_id,
                embedding_text_hash=digest,
                embedded_at=func.now(),
            )
            .returning(Cv.id)
        )
        if stored.first() is None:
            logger.info("Embedding of the CV of user %s dropped: the CV changed meanwhile", user_id)
            return EmbedOutcome.LOST_RACE
    logger.info("CV embedded: user=%s chunks=%d", user_id, len(pieces.chunks))
    return EmbedOutcome.EMBEDDED


# --- the sweep ---------------------------------------------------------------------------------

SWEEP_BATCH = 100  # most jobs queued by one sweep


@dataclass(frozen=True)
class SweepWork:
    job_ids: list[uuid.UUID]
    cv_user_ids: list[uuid.UUID]


async def collect_work(
    factory: SessionFactory, *, model_id: str, limit: int = SWEEP_BATCH
) -> SweepWork:
    """What the sweep should queue: extracted jobs without a vector, then (with the places left)
    jobs whose vector is from another model or older than their latest extraction; and CVs that
    are in either state. Oldest first, at most `limit` jobs."""
    async with _transaction(factory) as session:
        missing = list(
            await session.scalars(
                select(Job.id)
                .where(Job.embedding.is_(None), Job.extraction_status == ExtractionStatus.EXTRACTED)
                .order_by(Job.created_at)
                .limit(limit)
            )
        )
        stale: list[uuid.UUID] = []
        if len(missing) < limit:
            stale = list(
                await session.scalars(
                    select(Job.id)
                    .where(
                        Job.embedding.is_not(None),
                        Job.extraction_status == ExtractionStatus.EXTRACTED,
                        or_(
                            Job.embedding_model != model_id,
                            Job.embedded_at < Job.extracted_at,
                        ),
                    )
                    .order_by(Job.created_at)
                    .limit(limit - len(missing))
                )
            )
        cvs = list(
            await session.scalars(
                select(Cv.user_id)
                .where(
                    or_(
                        Cv.embedding.is_(None),
                        Cv.embedding_model != model_id,
                        Cv.embedded_at < Cv.updated_at,
                    )
                )
                .order_by(Cv.created_at)
                .limit(limit)
            )
        )
    return SweepWork(job_ids=[*missing, *stale], cv_user_ids=cvs)
