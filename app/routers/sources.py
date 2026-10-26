import uuid

from fastapi import APIRouter, Response, status

from app.api_deps import CurrentUserDep, PageDep, ResolverDep, SessionDep
from app.schemas.pagination import Page
from app.schemas.source import SourceCreate, SourceRead, SourceUpdate
from app.services import source_service

# Every endpoint takes CurrentUserDep, so all of them answer 401 without a valid access token.
router = APIRouter(prefix="/sources", tags=["sources"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_source(
    body: SourceCreate, user: CurrentUserDep, session: SessionDep, resolver: ResolverDep
) -> SourceRead:
    source = await source_service.create_source(session, user.id, body, resolver)
    return SourceRead.model_validate(source)


@router.get("")
async def list_sources(
    user: CurrentUserDep, session: SessionDep, page: PageDep
) -> Page[SourceRead]:
    sources, total = await source_service.list_sources(session, user.id, page)
    return Page[SourceRead](
        items=[SourceRead.model_validate(s) for s in sources],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{source_id}")
async def get_source(source_id: uuid.UUID, user: CurrentUserDep, session: SessionDep) -> SourceRead:
    return SourceRead.model_validate(await source_service.get_source(session, user.id, source_id))


@router.patch("/{source_id}")
async def update_source(
    source_id: uuid.UUID,
    body: SourceUpdate,
    user: CurrentUserDep,
    session: SessionDep,
    resolver: ResolverDep,
) -> SourceRead:
    """Change name, config and/or `is_active`. To stop fetching a source, set `is_active` to
    false instead of deleting it."""
    source = await source_service.update_source(session, user.id, source_id, body, resolver)
    return SourceRead.model_validate(source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> Response:
    """Delete a source **and all jobs collected from it**. Prefer `PATCH is_active=false`."""
    await source_service.delete_source(session, user.id, source_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
