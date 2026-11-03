"""Shared setup for the extraction tests that need committed data in a real Postgres.

Two workers need two real transactions, so these tests commit and clean up after themselves:
everything is created under users whose email ends in `@extraction.test`, and deleting those
users (the cascade) removes the sources, jobs and usage rows.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Job, LlmUsageEntry, Source, SourceType, User
from app.services.extraction_types import SessionFactory

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

RICH = (
    "Senior Python Developer at Acme\n\nWe are hiring a Senior Python Developer for our remote "
    "team.\nStack: Python, Django, PostgreSQL, Docker.\nApply: https://acme.com/careers/senior"
)
TITLE_ONLY_SHORT = "Marketing Manager at Acme\n\nWe are hiring a Marketing Manager."
TITLE_ONLY_LONG = TITLE_ONLY_SHORT + "\n\n" + "We build great products for happy customers. " * 20
NOTHING_SHORT = "lorem ipsum dolor sit amet " * 4
NOTHING_LONG = "lorem ipsum dolor sit amet " * 40
# A title the rules find, a technology they do not know, and a long enough post.
NEEDS_STACK = TITLE_ONLY_LONG + "\n\nWe rely on Zigzag Engine and Quux every day."


@pytest.fixture
async def factory(test_database_url: str, migrated_db: None) -> AsyncIterator[SessionFactory]:
    # A real connection pool (not NullPool), so a test can ask how many connections are checked
    # out at a given moment: `factory.kw["bind"].sync_engine.pool.checkedout()`.
    engine = create_async_engine(test_database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    yield sessions
    async with sessions() as session, session.begin():
        await session.execute(delete(User).where(User.email.like("%@extraction.test")))  # cascades
    await engine.dispose()


def checked_out_connections(factory: SessionFactory) -> int:
    """Database connections currently in use: 0 means no session or transaction is open."""
    return int(factory.kw["bind"].sync_engine.pool.checkedout())


async def add_job(
    factory: SessionFactory, text: str, *, user_id: uuid.UUID | None = None, **fields: Any
) -> uuid.UUID:
    """A job in a new source of `user_id` (or of a new user, so owners never coincide by chance)."""
    async with factory() as session, session.begin():
        if user_id is None:
            user = User(email=f"{uuid.uuid4()}@extraction.test", hashed_password="x")
            session.add(user)
            await session.flush()
            user_id = user.id
        source = Source(user_id=user_id, type=SourceType.RSS, name="feed")
        session.add(source)
        await session.flush()
        job = Job(source_id=source.id, external_id="post-1", raw_text=text, **fields)
        session.add(job)
        await session.flush()
        return job.id


async def add_user(factory: SessionFactory) -> uuid.UUID:
    async with factory() as session, session.begin():
        user = User(email=f"{uuid.uuid4()}@extraction.test", hashed_password="x")
        session.add(user)
        await session.flush()
        return user.id


async def owner_of(factory: SessionFactory, job_id: uuid.UUID) -> uuid.UUID:
    async with factory() as session:
        job = await session.get_one(Job, job_id)
        return (await session.get_one(Source, job.source_id)).user_id


async def row(factory: SessionFactory, job_id: uuid.UUID) -> Job:
    async with factory() as session:
        return await session.get_one(Job, job_id)


async def add_usage(
    factory: SessionFactory,
    user_id: uuid.UUID,
    created_at: datetime,
    *,
    outcome: str = "ok",
    job_id: uuid.UUID | None = None,
) -> uuid.UUID:
    async with factory() as session, session.begin():
        entry = LlmUsageEntry(
            user_id=user_id,
            job_id=job_id,
            provider="fake",
            model="fake-model",
            prompt_version="extract_job_v1",
            outcome=outcome,
            created_at=created_at,
        )
        session.add(entry)
        await session.flush()
        return entry.id


async def usage_rows(factory: SessionFactory) -> list[LlmUsageEntry]:
    """Every usage row of the test users, oldest first."""
    from sqlalchemy import select

    async with factory() as session:
        rows = await session.scalars(
            select(LlmUsageEntry)
            .join(User, User.id == LlmUsageEntry.user_id)
            .where(User.email.like("%@extraction.test"))
            .order_by(LlmUsageEntry.created_at, LlmUsageEntry.id)
        )
        return list(rows)
