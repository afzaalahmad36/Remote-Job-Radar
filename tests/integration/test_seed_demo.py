"""`scripts/seed_demo.py` against a real Postgres. No provider, no network, no Redis."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.core.security import verify_password
from app.models import ExtractionMethod, Job, LlmEnrichment, Source, User
from scripts import seed_demo


@pytest.fixture
async def demo_factory(
    test_database_url: str, migrated_db: None
) -> AsyncIterator[async_sessionmaker]:  # type: ignore[type-arg]
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await seed_demo.delete_demo_user(factory)
    yield factory
    await seed_demo.delete_demo_user(factory)
    await engine.dispose()


async def count(factory: async_sessionmaker, model: type) -> int:  # type: ignore[type-arg]
    async with factory() as session:
        return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_seeding_creates_the_user_one_inactive_source_and_twenty_extracted_posts(
    demo_factory: async_sessionmaker,  # type: ignore[type-arg]
    settings: Settings,
) -> None:
    jobs = seed_demo.load_fixture()

    result = await seed_demo.seed(demo_factory, jobs, settings=settings, password="a-test-pass-1")

    assert (result.user_created, result.inserted, result.total) == (True, 20, 20)
    assert result.extracted == 20
    async with demo_factory() as session:
        user = await session.scalar(select(User).where(User.email == seed_demo.DEMO_EMAIL))
        assert user is not None
        assert await verify_password("a-test-pass-1", user.hashed_password)
        sources = (await session.scalars(select(Source).where(Source.user_id == user.id))).all()
        assert len(sources) == 1 and sources[0].is_active is False
        assert sources[0].config == {"url": seed_demo.SOURCE_URL}
        stored = (await session.scalars(select(Job).where(Job.source_id == sources[0].id))).all()
    assert {j.extraction_method for j in stored} == {ExtractionMethod.RULES}  # no model ran
    # a post missing a key field waits for a model that is off: pending, never done
    assert LlmEnrichment.DONE not in {j.llm_enrichment for j in stored}
    assert all(j.embedding is None for j in stored)  # embeddings come from the running stack


async def test_seeding_twice_adds_nothing_and_keeps_the_first_password(
    demo_factory: async_sessionmaker,  # type: ignore[type-arg]
    settings: Settings,
) -> None:
    jobs = seed_demo.load_fixture()
    await seed_demo.seed(demo_factory, jobs, settings=settings, password="first-password-1")

    again = await seed_demo.seed(demo_factory, jobs, settings=settings, password="other-password-2")

    assert (again.user_created, again.inserted, again.total) == (False, 0, 20)
    assert again.password is None  # nothing to print: the password was not changed
    async with demo_factory() as session:
        user = await session.scalar(select(User).where(User.email == seed_demo.DEMO_EMAIL))
        assert user is not None
        assert await verify_password("first-password-1", user.hashed_password)
        assert not await verify_password("other-password-2", user.hashed_password)
    assert await count(demo_factory, Source) >= 1


async def test_reset_deletes_only_the_demo_user_and_its_posts(
    demo_factory: async_sessionmaker,  # type: ignore[type-arg]
    settings: Settings,
) -> None:
    await seed_demo.seed(
        demo_factory, seed_demo.load_fixture(), settings=settings, password="x" * 12
    )
    before_other_users = await count(demo_factory, User) - 1

    deleted = await seed_demo.delete_demo_user(demo_factory)

    assert deleted == 1
    assert await count(demo_factory, User) == before_other_users
    async with demo_factory() as session:
        assert (
            await session.scalar(select(User.id).where(User.email == seed_demo.DEMO_EMAIL)) is None
        )
