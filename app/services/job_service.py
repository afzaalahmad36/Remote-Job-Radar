import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job, Source
from app.schemas.pagination import PageParams


async def list_jobs(
    session: AsyncSession,
    user_id: uuid.UUID,
    page: PageParams,
    source_id: uuid.UUID | None = None,
) -> tuple[list[Job], int]:
    """One page of the jobs from the user's own sources, newest first, plus the total.

    Jobs have no owner column: they belong to a user through their source, so every query joins
    `sources` and filters on `sources.user_id`. That filter is what keeps other users' jobs out
    of both the page and the total. A `source_id` that is not the user's (or does not exist)
    just matches nothing, the same as an empty source.
    """
    owned = select(Job).join(Source, Source.id == Job.source_id).where(Source.user_id == user_id)
    if source_id is not None:
        owned = owned.where(Job.source_id == source_id)
    total = await session.scalar(select(func.count()).select_from(owned.subquery()))
    rows = await session.scalars(
        owned
        # id breaks ties (every job from one sync shares a created_at), so pages are stable
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    return list(rows), total or 0
