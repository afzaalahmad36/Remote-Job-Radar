"""Types and helpers shared by the extraction and enrichment services."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.llm_budget import DeferralReason

SessionFactory = async_sessionmaker[AsyncSession]


class ExtractionOutcome(StrEnum):
    NOT_FOUND = "not_found"  # the job no longer exists
    ALREADY_DONE = "already_done"  # nothing to do: this stage already ran
    LOST_RACE = "lost_race"  # another worker changed the job first; its result stands
    EXTRACTED = "extracted"  # rules phase done, no model needed (or none possible)
    AWAITING_MODEL = "awaiting_model"  # waits for a model that could not be asked in this run
    FAILED = "failed"  # no title, and nothing can add one
    REUSED = "reused"  # copied from a finished job with identical text
    ENRICHED = "enriched"  # a valid model answer was merged
    NOT_A_JOB = "not_a_job"  # the model said the text is not a job opening
    SKIPPED = "skipped"  # the model will not be asked for this job (reason on the job)
    DEFERRED = "deferred"  # not asked this time; the job is unchanged, try again later


@dataclass(frozen=True)
class ExtractionResult:
    outcome: ExtractionOutcome
    # Only for DEFERRED: why, and a hint for when a later run can succeed. Every hint is at least
    # a second, so a caller that honours it cannot retry in a tight loop.
    deferral: DeferralReason | None = None
    retry_after_seconds: float | None = None


@asynccontextmanager
async def transaction(factory: SessionFactory) -> AsyncIterator[AsyncSession]:
    """One short transaction on its own session; committed when the block ends."""
    async with factory() as session, session.begin():
        yield session
