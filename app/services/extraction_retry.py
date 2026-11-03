"""Asking for another extraction attempt of a job that failed or was given up on.

The whole thing happens in one database transaction that is committed only after the task has
been queued: if the queue is unreachable, the reset is rolled back and no retry is used.
"""

import uuid
from enum import StrEnum
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotRetryableError, RetriesExhaustedError
from app.models import ExtractionStatus, Job, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.services import job_service

MAX_EXTRACTION_RETRIES = 5
# The task is queued before the commit (so a broker failure can undo the reset). This small delay
# lets the commit land first, so the worker normally sees the reset job. If it is ever too early,
# the job is picked up by the sweep, which finds jobs whose rules phase has not run.
QUEUE_DELAY_SECONDS = 2.0


class ExtractionQueue(Protocol):
    """What this service needs from the background-job system."""

    async def is_extracting(self, job_id: uuid.UUID) -> bool: ...

    async def enqueue_extraction(self, job_id: uuid.UUID, countdown: float = 0.0) -> None: ...


class RetryStatus(StrEnum):
    QUEUED = "queued"
    ALREADY_RUNNING = "already_running"


def can_be_retried(job: Job) -> bool:
    """Only a job that failed, or whose model step was given up on, can be tried again."""
    return (
        job.extraction_status is ExtractionStatus.EXTRACTION_FAILED
        or job.llm_enrichment is LlmEnrichment.SKIPPED
    )


async def request_extraction_retry(
    session: AsyncSession, queue: ExtractionQueue, user_id: uuid.UUID, job_id: uuid.UUID
) -> RetryStatus:
    """Reset a failed or skipped job and queue its extraction.

    Raises NotFoundError (not yours, or missing), NotRetryableError, RetriesExhaustedError, or
    QueueUnavailableError (in which case nothing was changed). A retry does not skip the length
    gate, the budgets or the kill switch: it only gives the job another turn.
    """
    job = await job_service.get_owned_job(session, user_id, job_id)
    _check_retryable(job)
    if await queue.is_extracting(job.id):
        return RetryStatus.ALREADY_RUNNING  # a run is in progress; it is not an error
    used = await repository.reset_for_retry(
        session, user_id=user_id, job_id=job.id, max_retries=MAX_EXTRACTION_RETRIES
    )
    if used is None:  # lost a race with another request or a worker: say why
        await session.refresh(job)
        _check_retryable(job)
        raise NotRetryableError("This job cannot be retried now")
    try:
        await queue.enqueue_extraction(job.id, QUEUE_DELAY_SECONDS)  # BEFORE the commit
    except Exception:
        await session.rollback()  # the reset and the retry it used are undone
        raise
    await session.commit()
    return RetryStatus.QUEUED


def _check_retryable(job: Job) -> None:
    if not can_be_retried(job):
        raise NotRetryableError("This job cannot be retried now")
    if job.extraction_retries >= MAX_EXTRACTION_RETRIES:
        raise RetriesExhaustedError("This job has been retried as often as allowed")
