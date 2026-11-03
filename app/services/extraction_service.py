"""Extraction of one job: the rules phase, hash reuse, and then (if needed) the model path.

A run reads the job, releases the database, decides, and then writes with a compare-and-set, each
in its own short transaction. That is why the functions here take a session *factory*: one session
would have to stay open across the steps, and across a provider request. The model path itself
lives in `enrichment_service`.

Python note: `async_sessionmaker` is a callable that makes a new `AsyncSession` each time;
`async with factory() as session, session.begin():` opens a session, starts a transaction, commits
it when the block ends cleanly (or rolls back on an error) and closes the session.
"""

import logging
import uuid
from datetime import UTC, datetime

from app.core.config import Settings
from app.extraction.plan import RulesPlan, failed_plan, plan_rules_phase
from app.extraction.text_hash import raw_text_hash
from app.llm.base import LLMClient
from app.llm.prompts import EXTRACT_JOB_V1
from app.models import ExtractionStatus, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.repositories.extraction_repository import JobSnapshot
from app.services import enrichment_service
from app.services.cooldown import Cooldown
from app.services.extraction_types import (
    ExtractionOutcome,
    ExtractionResult,
    SessionFactory,
    transaction,
)

logger = logging.getLogger(__name__)

__all__ = ["ExtractionOutcome", "ExtractionResult", "SessionFactory", "extract_job"]


async def extract_job(
    factory: SessionFactory,
    job_id: uuid.UUID,
    *,
    settings: Settings,
    llm: LLMClient | None = None,
    cooldown: Cooldown | None = None,
    now: datetime | None = None,
) -> ExtractionResult:
    """Run the rules phase for one job and, if it still needs a model, ask it once.

    Safe to run again, and safe to run twice at once. `llm` and `cooldown` come together: the
    model is only asked when both exist, because without the shared pause there is no safe way to
    ask (the same rule as "if Redis is down, do not ask"). Without them the job simply waits.
    """
    can_enrich = llm is not None and cooldown is not None
    async with transaction(factory) as session:
        snapshot = await repository.load_snapshot(session, job_id)
    if snapshot is None:
        logger.info("Skipping job %s: it no longer exists", job_id)
        return ExtractionResult(ExtractionOutcome.NOT_FOUND)
    if snapshot.enrichment is LlmEnrichment.PENDING:
        result = ExtractionResult(ExtractionOutcome.AWAITING_MODEL)  # resume at the model path
    elif (snapshot.status, snapshot.enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    ):
        result = await _rules_phase(
            factory, snapshot, settings, can_enrich=can_enrich, now=now or datetime.now(UTC)
        )
    else:
        return ExtractionResult(ExtractionOutcome.ALREADY_DONE)
    if result.outcome is not ExtractionOutcome.AWAITING_MODEL or llm is None or cooldown is None:
        return result
    return await enrichment_service.enrich_job(
        factory, job_id, settings=settings, llm=llm, cooldown=cooldown, now=now
    )


async def _rules_phase(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    settings: Settings,
    *,
    can_enrich: bool,
    now: datetime,
) -> ExtractionResult:
    text_hash = raw_text_hash(snapshot.raw_text)
    async with transaction(factory) as session:
        donor = await repository.find_reusable(
            session,
            text_hash=text_hash,
            exclude_id=snapshot.id,
            prompt_version=EXTRACT_JOB_V1.version,
        )
    if donor is not None:
        async with transaction(factory) as session:
            won = await repository.write_reused_result(session, snapshot, donor, text_hash, now)
        return _finish(snapshot, won, ExtractionOutcome.REUSED)

    plan = _plan(snapshot, settings, can_enrich)
    async with transaction(factory) as session:
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


def _finish(snapshot: JobSnapshot, won: bool, outcome: ExtractionOutcome) -> ExtractionResult:
    if won:
        logger.info("Job %s: rules phase done (%s)", snapshot.id, outcome.value)
        return ExtractionResult(outcome)
    # Zero rows matched: another worker changed the job after our snapshot. Its result stands,
    # and writing again would risk overwriting something this run never saw.
    logger.info("Job %s: another worker wrote first, leaving its result", snapshot.id)
    return ExtractionResult(ExtractionOutcome.LOST_RACE)
