"""Helpers for testing Celery task code, which opens its own database connections.

Tasks cannot share the rollback-per-test session of the other tests (they create their own engine
and event loop), so these tests commit real rows into `jobradar_test` and delete them afterwards.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.models import Job, LlmUsageEntry, RefreshToken, Source, SourceType, User
from tests.fakes import UNREACHABLE_REDIS_URL


class Seeder:
    """Creates and inspects committed rows. Every call is its own short `asyncio.run`."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._user_ids: list[uuid.UUID] = []

    def _run(self, work: Callable[[AsyncSession], Awaitable[Any]]) -> Any:
        async def go() -> Any:
            engine = create_async_engine(self._url, poolclass=NullPool)
            try:
                async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                    return await work(session)
            finally:
                await engine.dispose()

        return asyncio.run(go())

    def user(self) -> uuid.UUID:
        async def work(session: AsyncSession) -> uuid.UUID:
            user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="x")
            session.add(user)
            await session.commit()
            return user.id

        user_id: uuid.UUID = self._run(work)
        self._user_ids.append(user_id)
        return user_id

    def source(
        self,
        user_id: uuid.UUID,
        url: str = "https://example.com/feed.xml",
        name: str = "feed",
        active: bool = True,
        last_synced_at: datetime | None = None,
    ) -> uuid.UUID:
        async def work(session: AsyncSession) -> uuid.UUID:
            source = Source(
                user_id=user_id,
                type=SourceType.RSS,
                name=name,
                config={"url": url},
                is_active=active,
                last_synced_at=last_synced_at,
            )
            session.add(source)
            await session.commit()
            return source.id

        source_id: uuid.UUID = self._run(work)
        return source_id

    def source_row(self, source_id: uuid.UUID) -> Source:
        async def work(session: AsyncSession) -> Source | None:
            return await session.get(Source, source_id)

        row: Source | None = self._run(work)
        assert row is not None
        return row

    def jobs(self, source_id: uuid.UUID) -> list[Job]:
        async def work(session: AsyncSession) -> list[Job]:
            rows = await session.scalars(select(Job).where(Job.source_id == source_id))
            return list(rows)

        jobs: list[Job] = self._run(work)
        return jobs

    def job(self, source_id: uuid.UUID, text: str, **fields: Any) -> uuid.UUID:
        """A job in `source_id`; `fields` are Job columns (status, created_at, title, ...)."""

        async def work(session: AsyncSession) -> uuid.UUID:
            job = Job(source_id=source_id, external_id=str(uuid.uuid4()), raw_text=text, **fields)
            session.add(job)
            await session.commit()
            return job.id

        job_id: uuid.UUID = self._run(work)
        return job_id

    def job_row(self, job_id: uuid.UUID) -> Job:
        async def work(session: AsyncSession) -> Job | None:
            return await session.get(Job, job_id)

        row: Job | None = self._run(work)
        assert row is not None
        return row

    def usage(self, user_id: uuid.UUID) -> list[LlmUsageEntry]:
        async def work(session: AsyncSession) -> list[LlmUsageEntry]:
            rows = await session.scalars(
                select(LlmUsageEntry).where(LlmUsageEntry.user_id == user_id)
            )
            return list(rows)

        entries: list[LlmUsageEntry] = self._run(work)
        return entries

    def add_usage(self, user_id: uuid.UUID, created_at: datetime, outcome: str = "ok") -> uuid.UUID:
        async def work(session: AsyncSession) -> uuid.UUID:
            entry = LlmUsageEntry(
                user_id=user_id,
                provider="fake",
                model="fake-model",
                prompt_version="extract_job_v1",
                outcome=outcome,
                created_at=created_at,
            )
            session.add(entry)
            await session.commit()
            return entry.id

        entry_id: uuid.UUID = self._run(work)
        return entry_id

    def delete_source(self, source_id: uuid.UUID) -> None:
        async def work(session: AsyncSession) -> None:
            await session.execute(delete(Source).where(Source.id == source_id))
            await session.commit()

        self._run(work)

    def refresh_token(self, user_id: uuid.UUID, expires_at: datetime) -> uuid.UUID:
        async def work(session: AsyncSession) -> uuid.UUID:
            token = RefreshToken(
                user_id=user_id,
                family_id=uuid.uuid4(),
                token_hash=uuid.uuid4().hex * 2,  # 64 characters, like a SHA-256 hex digest
                expires_at=expires_at,
            )
            session.add(token)
            await session.commit()
            return token.id

        token_id: uuid.UUID = self._run(work)
        return token_id

    def token_exists(self, token_id: uuid.UUID) -> bool:
        async def work(session: AsyncSession) -> bool:
            return await session.get(RefreshToken, token_id) is not None

        exists: bool = self._run(work)
        return exists

    def cleanup(self) -> None:
        async def work(session: AsyncSession) -> None:
            await session.execute(delete(User).where(User.id.in_(self._user_ids)))  # cascades
            await session.commit()

        if self._user_ids:
            self._run(work)


@pytest.fixture
def seed(test_database_url: str, migrated_db: None) -> Iterator[Seeder]:
    seeder = Seeder(test_database_url)
    try:
        yield seeder
    finally:
        seeder.cleanup()


@pytest.fixture
def celery_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Celery task objects read the app settings the first time they are used, so tests that
    touch them provide a throwaway environment (no broker is ever contacted)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/x")
    monkeypatch.setenv("REDIS_URL", UNREACHABLE_REDIS_URL)
    monkeypatch.setenv("SECRET_KEY", "celery-test-secret-key-at-least-32-chars")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
