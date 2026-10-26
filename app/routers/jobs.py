import uuid

from fastapi import APIRouter

from app.api_deps import CurrentUserDep, PageDep, SessionDep
from app.schemas.job import JobRead
from app.schemas.pagination import Page
from app.services import job_service

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
