import uuid
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateSourceError, InvalidConfigError, NotFoundError
from app.core.url_safety import STANDARD_PORTS, Resolver, system_resolver, validate_public_url
from app.models import Source, SourceType
from app.schemas.pagination import PageParams
from app.schemas.source import CONFIG_MODELS, SourceCreate, SourceUpdate


async def create_source(
    session: AsyncSession,
    user_id: uuid.UUID,
    data: SourceCreate,
    resolver: Resolver = system_resolver,
) -> Source:
    """Create a source owned by `user_id` after validating its config and URL.

    Raises DuplicateSourceError if the user already has an RSS source with the same URL.
    """
    config = await _validated_config(session, user_id, data.type, data.config, resolver)
    source = Source(user_id=user_id, type=data.type, name=data.name, config=config)
    session.add(source)
    await session.commit()
    await session.refresh(source)  # reload the values the database filled in (timestamps)
    return source


async def list_sources(
    session: AsyncSession, user_id: uuid.UUID, page: PageParams
) -> tuple[list[Source], int]:
    """One page of the user's sources, newest first, plus how many they have in total."""
    total = await session.scalar(
        select(func.count()).select_from(Source).where(Source.user_id == user_id)
    )
    rows = await session.scalars(
        select(Source)
        .where(Source.user_id == user_id)
        # id as a tiebreaker keeps the order stable, so no row repeats or vanishes between pages
        .order_by(Source.created_at.desc(), Source.id.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    return list(rows), total or 0


async def get_source(session: AsyncSession, user_id: uuid.UUID, source_id: uuid.UUID) -> Source:
    """The user's source, or NotFoundError. Someone else's source is "not found" too."""
    source = await session.scalar(
        select(Source).where(Source.id == source_id, Source.user_id == user_id)
    )
    if source is None:
        raise NotFoundError("Source not found")
    return source


async def update_source(
    session: AsyncSession,
    user_id: uuid.UUID,
    source_id: uuid.UUID,
    data: SourceUpdate,
    resolver: Resolver = system_resolver,
) -> Source:
    source = await get_source(session, user_id, source_id)
    # Validate before changing anything, so a rejected update leaves the source untouched.
    # A new config gets exactly the checks a created one does: shape, duplicates (ignoring this
    # source itself) and URL safety including DNS, whether or not the URL text changed.
    config = (
        await _validated_config(session, user_id, source.type, data.config, resolver, source.id)
        if data.config is not None
        else None
    )
    if data.name is not None:
        source.name = data.name
    if data.is_active is not None:
        source.is_active = data.is_active
    if config is not None:
        if config != source.config:
            # A different URL is a different feed: its old ETag / Last-Modified would be sent to
            # the wrong server, and the old error no longer describes anything.
            source.etag = source.last_modified = source.last_error = None
        source.config = config
    await session.commit()
    await session.refresh(source)
    return source


async def delete_source(session: AsyncSession, user_id: uuid.UUID, source_id: uuid.UUID) -> None:
    """Delete a source. The database cascades the delete to the source's jobs.

    Prefer deactivating (`is_active=false` via update_source): that stops fetching but keeps the
    jobs already collected.
    """
    source = await get_source(session, user_id, source_id)
    await session.delete(source)
    await session.commit()


def normalize_feed_url(url: str) -> str:
    """The key for "is this the same feed?": two URLs with the same key count as duplicates.

    Lowercases the scheme and host, drops a trailing dot on the host, the scheme's default port,
    any trailing slash(es) on the path, and the #fragment. The path (case-sensitive) and the
    query string (which often selects a different feed) are kept, and http and https stay
    different. The stored URL is left exactly as the user typed it; only the comparison is
    normalized.
    """
    url = url.strip()
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:  # cannot happen for stored URLs (they were validated); stay safe anyway
        return url
    host = (parts.hostname or "").removesuffix(".")
    netloc = host if port in (None, STANDARD_PORTS.get(parts.scheme)) else f"{host}:{port}"
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}{query}"


async def _ensure_not_duplicate(
    session: AsyncSession, user_id: uuid.UUID, url: str, ignore_id: uuid.UUID | None
) -> None:
    """Raise DuplicateSourceError if the user already has an RSS source for this URL.

    Only this user's sources are looked at, so two users can follow the same feed. This is a
    check in code, not a database constraint: two simultaneous requests for the same new URL
    could both pass it. A unique index on (user_id, normalized url) would close that gap.
    """
    wanted = normalize_feed_url(url)
    query = select(Source.config).where(Source.user_id == user_id, Source.type == SourceType.RSS)
    if ignore_id is not None:
        query = query.where(Source.id != ignore_id)
    for existing in await session.scalars(query):
        if normalize_feed_url(str(existing.get("url", ""))) == wanted:
            raise DuplicateSourceError("You already have a source for this URL")


async def _validated_config(
    session: AsyncSession,
    user_id: uuid.UUID,
    source_type: SourceType,
    config: dict[str, Any],
    resolver: Resolver,
    ignore_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    model = CONFIG_MODELS.get(source_type)
    if model is None:
        raise InvalidConfigError(f"Source type '{source_type.value}' is not supported yet")
    try:
        valid = model.model_validate(config).model_dump()
    except ValidationError as error:
        problems = "; ".join(
            f"config.{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise InvalidConfigError(problems) from error
    if source_type is SourceType.RSS:
        # Cheap database check first, so a duplicate never costs a DNS lookup.
        await _ensure_not_duplicate(session, user_id, valid["url"], ignore_id)
        await validate_public_url(valid["url"], resolver)  # raises UnsafeUrlError (422)
    return valid
