import uuid

from fastapi import APIRouter, Depends, status

from app.api_deps import (
    CurrentUserDep,
    ExtractionQueueDep,
    PageDep,
    SessionDep,
    rate_limit_extract,
)
from app.schemas.job import ExtractAcceptedOut, JobRead
from app.schemas.pagination import Page
from app.services import extraction_retry, job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(
    user: CurrentUserDep,
    session: SessionDep,
    page: PageDep,
    source_id: uuid.UUID | None = None,
) -> Page[JobRead]:
    """Jobs collected from your sources, newest first. Optionally only those of one source."""
    jobs, total = await job_service.list_jobs(session, user.id, page, source_id)
    return Page[JobRead](
        items=[JobRead.model_validate(job) for job in jobs],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "/{job_id}/extract",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit_extract)],
)
async def retry_extraction(
    job_id: uuid.UUID, user: CurrentUserDep, session: SessionDep, queue: ExtractionQueueDep
) -> ExtractAcceptedOut:
    """Try again to extract a job that failed or whose model step was given up on.

    Returns 202: the extraction runs in the background and the result appears in `GET /jobs`.
    `status` is `queued`, or `already_running` if a run is in progress (not an error). A job in
    any other state gets 409 `not_retryable`; after 5 retries, 409 `retries_exhausted`. If the
    queue is down the answer is 503 and nothing was changed. A retry does not skip the length
    gate, the request budgets or the kill switch.
    """
    outcome = await extraction_retry.request_extraction_retry(session, queue, user.id, job_id)
    return ExtractAcceptedOut(job_id=job_id, status=outcome)
