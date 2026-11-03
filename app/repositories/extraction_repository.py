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

from sqlalchemy import ColumnElement, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction.merge import MergedExtraction
from app.extraction.plan import RulesPlan
from app.extraction.rules import RULES_VERSION
from app.models import (
    ExtractionMethod,
    ExtractionStatus,
    Job,
    LlmEnrichment,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
)


@dataclass(frozen=True)
class JobSnapshot:
    """A job's state as one worker saw it. Plain values, so no session has to stay open."""

    id: uuid.UUID
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
                Job.raw_text,
                Job.extraction_status,
                Job.llm_enrichment,
                Job.extraction_attempts,
            ).where(Job.id == job_id)
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


async def _compare_and_set(
    session: AsyncSession, snapshot: JobSnapshot, changes: dict[str, Any]
) -> bool:
    statement = (
        update(Job)
        .where(*_still_as_seen(snapshot))
        .values(**changes, extraction_attempts=snapshot.attempts + 1)
        .returning(Job.id)
    )
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
