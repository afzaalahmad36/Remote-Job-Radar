"""The SQL behind extraction: reading a job's state and changing it safely.

Every change is a compare-and-set. The UPDATE names the state the caller saw (status, enrichment
and attempts) in its WHERE clause, so when two workers hold the same snapshot only the first
UPDATE matches a row; the second matches none and learns it lost. Nothing is locked while the
caller works, and no worker can overwrite a result it never saw.

`extraction_attempts` is the version counter: the rules write always adds one, so even a write
that leaves status and enrichment unchanged still makes every other holder of the snapshot lose.
`extraction_method` is set to NULL by exactly one piece of code, the retry endpoint (planned);
nothing in this module does that.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction.merge import MergedExtraction
from app.extraction.plan import ExtractionError, RulesPlan
from app.extraction.rules import RULES_VERSION
from app.models import (
    ExtractionMethod,
    ExtractionStatus,
    Job,
    LlmEnrichment,
    LlmUsageEntry,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
    Source,
)


@dataclass(frozen=True)
class JobSnapshot:
    """A job's state as one worker saw it. Plain values, so no session has to stay open."""

    id: uuid.UUID
    user_id: uuid.UUID  # the owner, through the job's source: whose daily budget a request uses
    raw_text: str
    status: ExtractionStatus
    enrichment: LlmEnrichment
    attempts: int


@dataclass(frozen=True)
class ReusableExtraction:
    """The extraction fields of another job with the same post text. Never ids or ownership."""

    title: str | None
    company: str | None
    stack: tuple[str, ...]
    seniority: Seniority | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    remote_policy: RemotePolicy | None
    apply_email: str | None
    apply_url: str | None
    status: ExtractionStatus
    enrichment: LlmEnrichment
    method: ExtractionMethod
    rules_version: str | None
    prompt_version: str | None
    model: str | None


async def load_snapshot(session: AsyncSession, job_id: uuid.UUID) -> JobSnapshot | None:
    row = (
        await session.execute(
            select(
                Job.id,
                Source.user_id,
                Job.raw_text,
                Job.extraction_status,
                Job.llm_enrichment,
                Job.extraction_attempts,
            )
            .join(Source, Source.id == Job.source_id)
            .where(Job.id == job_id)
        )
    ).first()
    return JobSnapshot(*row) if row else None


async def find_reusable(
    session: AsyncSession,
    *,
    text_hash: str,
    exclude_id: uuid.UUID,
    prompt_version: str,
) -> ReusableExtraction | None:
    """Another job with the same post text whose extraction is finished and still current.

    "Finished" means the rules phase ran, no model answer is outstanding, and the job holds a
    real result (extracted, or judged not a job). A result made by an older version of the rules
    or of the prompt is not reused. A model-enriched result is preferred over a rules-only one.
    """
    donor = await session.scalar(
        select(Job)
        .where(
            Job.raw_text_hash == text_hash,
            Job.id != exclude_id,
            Job.extraction_method.is_not(None),
            Job.llm_enrichment.in_([LlmEnrichment.DONE, LlmEnrichment.NOT_NEEDED]),
            Job.extraction_status.in_([ExtractionStatus.EXTRACTED, ExtractionStatus.NOT_A_JOB]),
            Job.extraction_rules_version == RULES_VERSION,
            or_(
                Job.extraction_method == ExtractionMethod.RULES,
                Job.extraction_prompt_version == prompt_version,
            ),
        )
        .order_by(
            (Job.extraction_method == ExtractionMethod.RULES_LLM).desc(), Job.extracted_at.desc()
        )
        .limit(1)
    )
    if donor is None or donor.extraction_method is None:
        return None
    return ReusableExtraction(
        title=donor.title,
        company=donor.company,
        stack=tuple(donor.stack),
        seniority=donor.seniority,
        salary_min=donor.salary_min,
        salary_max=donor.salary_max,
        salary_currency=donor.salary_currency,
        salary_period=donor.salary_period,
        remote_policy=donor.remote_policy,
        apply_email=donor.apply_email,
        apply_url=donor.apply_url,
        status=donor.extraction_status,
        enrichment=donor.llm_enrichment,
        method=donor.extraction_method,
        rules_version=donor.extraction_rules_version,
        prompt_version=donor.extraction_prompt_version,
        model=donor.extraction_model,
    )


async def write_rules_result(
    session: AsyncSession, snapshot: JobSnapshot, plan: RulesPlan, now: datetime
) -> bool:
    """Store the rules' result. True if this worker won, False if the job changed under it."""
    return await _compare_and_set(
        session,
        snapshot,
        {
            **_value_columns(plan.values),
            "raw_text_hash": plan.text_hash,
            "extraction_method": ExtractionMethod.RULES,
            "extraction_rules_version": RULES_VERSION,
            "extraction_prompt_version": None,
            "extraction_model": None,
            "extracted_at": now,
            "extraction_error": plan.error.value if plan.error else None,
            "extraction_status": plan.status,
            "llm_enrichment": plan.enrichment,
        },
    )


async def write_reused_result(
    session: AsyncSession,
    snapshot: JobSnapshot,
    donor: ReusableExtraction,
    text_hash: str,
    now: datetime,
) -> bool:
    """Copy a finished extraction of identical text onto this job (extraction fields only)."""
    return await _compare_and_set(
        session,
        snapshot,
        {
            "title": donor.title,
            "company": donor.company,
            "stack": list(donor.stack),
            "seniority": donor.seniority,
            "salary_min": donor.salary_min,
            "salary_max": donor.salary_max,
            "salary_currency": donor.salary_currency,
            "salary_period": donor.salary_period,
            "remote_policy": donor.remote_policy,
            "apply_email": donor.apply_email,
            "apply_url": donor.apply_url,
            "raw_text_hash": text_hash,
            "extraction_method": donor.method,
            "extraction_rules_version": donor.rules_version,
            "extraction_prompt_version": donor.prompt_version,
            "extraction_model": donor.model,
            "extracted_at": now,
            "extraction_error": None,
            "extraction_status": donor.status,
            "llm_enrichment": donor.enrichment,
        },
    )


async def write_model_result(
    session: AsyncSession,
    snapshot: JobSnapshot,
    values: MergedExtraction,
    *,
    status: ExtractionStatus,
    error: ExtractionError | None,
    model: str,
    prompt_version: str,
    now: datetime,
) -> bool:
    """Store the merge of the rules' result and a valid model answer, and finish the enrichment.

    The job must still be waiting for the model, as in the snapshot. It does not add to
    `extraction_attempts`: leaving `pending` is what makes a second writer lose.
    """
    return await _compare_and_set(
        session,
        snapshot,
        {
            **_value_columns(values),
            "extraction_method": ExtractionMethod.RULES_LLM,
            "extraction_rules_version": RULES_VERSION,
            "extraction_prompt_version": prompt_version,
            "extraction_model": model,
            "extracted_at": now,
            "extraction_error": error.value if error else None,
            "extraction_status": status,
            "llm_enrichment": LlmEnrichment.DONE,
        },
        bump_attempts=False,
    )


async def write_skipped(
    session: AsyncSession, snapshot: JobSnapshot, error: ExtractionError
) -> bool:
    """Give up on the model for this job and keep what the rules found.

    A job that has no title fails here, because nothing is left that could still give it one.
    """
    return await _compare_and_set(
        session,
        snapshot,
        {
            "llm_enrichment": LlmEnrichment.SKIPPED,
            "extraction_error": error.value,
            "extraction_status": case(
                (Job.title.is_(None), ExtractionStatus.EXTRACTION_FAILED),
                else_=Job.extraction_status,
            ),
        },
        bump_attempts=False,
    )


async def reset_for_retry(
    session: AsyncSession, *, user_id: uuid.UUID, job_id: uuid.UUID, max_retries: int
) -> int | None:
    """Put a failed or skipped job back at the start, and use one of its retries.

    This is the only place that sets `extraction_method` back to NULL, which is what makes the
    sweep (and the task the caller queues) treat the job as "rules phase not run". The WHERE
    clause is the whole decision, so two clicks at once use one retry: the job must belong to the
    user, be failed or skipped, and have retries left. Returns the new retry count, or None if
    the job did not qualify (the caller then looks again to say why).
    """
    owned_sources = select(Source.id).where(Source.user_id == user_id)
    statement = (
        update(Job)
        .where(
            Job.id == job_id,
            Job.source_id.in_(owned_sources),
            Job.extraction_retries < max_retries,
            or_(
                Job.extraction_status == ExtractionStatus.EXTRACTION_FAILED,
                Job.llm_enrichment == LlmEnrichment.SKIPPED,
            ),
        )
        .values(
            extraction_status=ExtractionStatus.PENDING,
            llm_enrichment=LlmEnrichment.NOT_NEEDED,
            extraction_error=None,
            extraction_method=None,
            extraction_retries=Job.extraction_retries + 1,
        )
        .returning(Job.extraction_retries)
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def give_up_waiting(session: AsyncSession, *, older_than: datetime) -> int:
    """Stop waiting for the model for jobs that have waited since before `older_than`.

    The wait starts when the rules phase made the job `pending` (`extracted_at`), not when the row
    was created: a job that sat in the database before the pipeline ran must still get its full
    waiting period. `created_at` is only the fallback when `extracted_at` is NULL. Every rules
    write sets `extracted_at`, and a retry reset puts the job back to `not_needed`, so a retried
    job starts a fresh wait when its new rules phase runs.

    One set-based UPDATE. The rules' data stays; a job that still has no title becomes failed,
    because nothing is left that could give it one. A worker finishing the same job at the same
    moment either wins (then this row no longer matches) or loses its compare-and-set.
    """
    statement = (
        update(Job)
        .where(
            Job.llm_enrichment == LlmEnrichment.PENDING,
            func.coalesce(Job.extracted_at, Job.created_at) < older_than,
        )
        .values(
            llm_enrichment=LlmEnrichment.SKIPPED,
            extraction_error=ExtractionError.GAVE_UP.value,
            extraction_status=case(
                (Job.title.is_(None), ExtractionStatus.EXTRACTION_FAILED),
                else_=Job.extraction_status,
            ),
        )
        .returning(Job.id)
    )
    return len((await session.execute(statement)).all())


async def job_ids_without_rules(session: AsyncSession, limit: int) -> list[uuid.UUID]:
    """Jobs whose rules phase has not run and could run now, oldest first.

    That is every new job (the task queued for it may have been lost) and every job a retry
    reset. Jobs in any other state are left out, so the sweep never queues work that would end
    as a no-op.
    """
    rows = await session.scalars(
        select(Job.id)
        .where(
            Job.extraction_method.is_(None),
            Job.extraction_status == ExtractionStatus.PENDING,
            Job.llm_enrichment == LlmEnrichment.NOT_NEEDED,
        )
        .order_by(Job.created_at, Job.id)
        .limit(limit)
    )
    return list(rows)


async def waiting_job_ids(
    session: AsyncSession, *, limit: int, user_cap: int, since: datetime
) -> list[uuid.UUID]:
    """Jobs waiting for the model, oldest first, leaving out owners who used their daily cap.

    Leaving them out matters: otherwise one user with many old waiting jobs and no budget left
    would fill every sweep's batch and starve everyone behind them.
    """
    if user_cap <= 0:
        return []
    capped = (
        select(LlmUsageEntry.user_id)
        .where(LlmUsageEntry.created_at > since)
        .group_by(LlmUsageEntry.user_id)
        .having(func.count() >= user_cap)
    )
    rows = await session.scalars(
        select(Job.id)
        .join(Source, Source.id == Job.source_id)
        .where(Job.llm_enrichment == LlmEnrichment.PENDING, Source.user_id.not_in(capped))
        .order_by(Job.created_at, Job.id)
        .limit(limit)
    )
    return list(rows)


async def _compare_and_set(
    session: AsyncSession,
    snapshot: JobSnapshot,
    changes: dict[str, Any],
    *,
    bump_attempts: bool = True,
) -> bool:
    if bump_attempts:
        changes = {**changes, "extraction_attempts": snapshot.attempts + 1}
    statement = update(Job).where(*_still_as_seen(snapshot)).values(**changes).returning(Job.id)
    return (await session.execute(statement)).first() is not None  # no row = another worker won


def _still_as_seen(snapshot: JobSnapshot) -> list[ColumnElement[bool]]:
    return [
        Job.id == snapshot.id,
        Job.extraction_status == snapshot.status,
        Job.llm_enrichment == snapshot.enrichment,
        Job.extraction_attempts == snapshot.attempts,
    ]


def _value_columns(values: MergedExtraction) -> dict[str, Any]:
    return {
        "title": values.title,
        "company": values.company,
        "stack": list(values.stack),
        "seniority": values.seniority,
        "salary_min": values.salary_min,
        "salary_max": values.salary_max,
        "salary_currency": values.salary_currency,
        "salary_period": values.salary_period,
        "remote_policy": values.remote_policy,
        "apply_email": values.apply_email,
        "apply_url": values.apply_url,
    }
