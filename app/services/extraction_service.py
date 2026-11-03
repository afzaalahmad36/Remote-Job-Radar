"""Extraction of one job: the rules phase, hash reuse and the state changes around them.

A run reads the job, releases the database, decides, and then writes with a compare-and-set, each
in its own short transaction. That is why the functions here take a session *factory*: one session
would have to stay open across the steps, and (from the model path on) across a provider request.
The model call itself is not built yet; a job that needs one is left waiting for it.

Python note: `async_sessionmaker` is a callable that makes a new `AsyncSession` each time;
`async with factory() as session, session.begin():` opens a session, starts a transaction, commits
it when the block ends cleanly (or rolls back on an error) and closes the session.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.extraction.plan import RulesPlan, failed_plan, plan_rules_phase
from app.extraction.text_hash import raw_text_hash
from app.llm.base import LLMClient
from app.llm.prompts import EXTRACT_JOB_V1
from app.models import ExtractionStatus, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.repositories.extraction_repository import JobSnapshot

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


class ExtractionOutcome(StrEnum):
    NOT_FOUND = "not_found"  # the job no longer exists
    ALREADY_DONE = "already_done"  # nothing to do: the rules phase already ran
    LOST_RACE = "lost_race"  # another worker changed the job first; its result stands
    EXTRACTED = "extracted"  # rules phase done, no model needed (or none possible)
    AWAITING_MODEL = "awaiting_model"  # usable or not, it now waits for the model path
    FAILED = "failed"  # no title, and nothing can add one
    REUSED = "reused"  # copied from a finished job with identical text


@asynccontextmanager
async def _transaction(factory: SessionFactory) -> AsyncIterator[AsyncSession]:
    """One short transaction on its own session; committed when the block ends."""
    async with factory() as session, session.begin():
        yield session


async def extract_job(
    factory: SessionFactory,
    job_id: uuid.UUID,
    *,
    settings: Settings,
    llm: LLMClient | None = None,
    now: datetime | None = None,
) -> ExtractionOutcome:
    """Run the rules phase for one job, once. Safe to run again, and safe to run twice at once.

    `llm` says whether a model could be asked (None: provider off). In this step it is only
    checked for presence; the model path comes later. Returns what happened, never raises for a
    job that is missing or already handled.
    """
    async with _transaction(factory) as session:
        snapshot = await repository.load_snapshot(session, job_id)
    if snapshot is None:
        logger.info("Skipping job %s: it no longer exists", job_id)
        return ExtractionOutcome.NOT_FOUND
    if snapshot.enrichment is LlmEnrichment.PENDING:
        return ExtractionOutcome.AWAITING_MODEL  # the rules phase ran; the model path is next
    if (snapshot.status, snapshot.enrichment) != (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    ):
        return ExtractionOutcome.ALREADY_DONE
    return await _rules_phase(
        factory, snapshot, settings, can_enrich=llm is not None, now=now or datetime.now(UTC)
    )


async def _rules_phase(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    settings: Settings,
    *,
    can_enrich: bool,
    now: datetime,
) -> ExtractionOutcome:
    text_hash = raw_text_hash(snapshot.raw_text)
    async with _transaction(factory) as session:
        donor = await repository.find_reusable(
            session,
            text_hash=text_hash,
            exclude_id=snapshot.id,
            prompt_version=EXTRACT_JOB_V1.version,
        )
    if donor is not None:
        async with _transaction(factory) as session:
            won = await repository.write_reused_result(session, snapshot, donor, text_hash, now)
        return _finish(snapshot, won, ExtractionOutcome.REUSED)

    plan = _plan(snapshot, settings, can_enrich)
    async with _transaction(factory) as session:
        won = await repository.write_rules_result(session, snapshot, plan, now)
    return _finish(snapshot, won, _outcome_of(plan))


def _plan(snapshot: JobSnapshot, settings: Settings, can_enrich: bool) -> RulesPlan:
    try:
        return plan_rules_phase(
            snapshot.raw_text, can_enrich=can_enrich, min_chars=settings.llm_min_post_chars
        )
    except Exception as error:  # the rules should never raise; if one does, fail the job
        # Not re-raised: a job that raised on every run would be picked up again by every sweep
        # forever. It is stored as failed (a retry can revive it). Only the job id and the
        # exception *type* are logged: the message and the traceback can quote the post text.
        logger.error(
            "The rules phase raised for job %s (error_type=%s)", snapshot.id, type(error).__name__
        )
        return failed_plan(snapshot.raw_text)


def _outcome_of(plan: RulesPlan) -> ExtractionOutcome:
    if plan.enrichment is LlmEnrichment.PENDING:
        return ExtractionOutcome.AWAITING_MODEL
    if plan.status is ExtractionStatus.EXTRACTION_FAILED:
        return ExtractionOutcome.FAILED
    return ExtractionOutcome.EXTRACTED


def _finish(snapshot: JobSnapshot, won: bool, outcome: ExtractionOutcome) -> ExtractionOutcome:
    if won:
        logger.info("Job %s: rules phase done (%s)", snapshot.id, outcome.value)
        return outcome
    # Zero rows matched: another worker changed the job after our snapshot. Its result stands,
    # and writing again would risk overwriting something this run never saw.
    logger.info("Job %s: another worker wrote first, leaving its result", snapshot.id)
    return ExtractionOutcome.LOST_RACE
