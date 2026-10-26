import asyncio
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from tests.db_utils import alembic_config

# These tests are sync on purpose: Alembic's env.py calls asyncio.run(), which cannot run inside
# an already-running event loop (the one an `async def` test would be in).


@pytest.fixture
def config(test_database_url: str) -> Iterator[Config]:
    """Alembic config for the test DB. Whatever a test does, the DB is back at head afterwards,
    so these tests cannot affect the order-sensitive DB tests that run after them."""
    cfg = alembic_config(test_database_url)
    command.upgrade(cfg, "head")
    yield cfg
    command.upgrade(cfg, "head")


def query(database_url: str, sql: str) -> list[str]:
    async def run() -> list[str]:
        engine = create_async_engine(database_url, poolclass=NullPool)
        async with engine.connect() as connection:
            rows = (await connection.execute(text(sql))).scalars().all()
        await engine.dispose()
        return [str(row) for row in rows]

    return asyncio.run(run())


def test_upgrade_head_creates_the_schema(config: Config, test_database_url: str) -> None:
    command.downgrade(config, "base")

    command.upgrade(config, "head")

    tables = query(test_database_url, "SELECT tablename FROM pg_tables WHERE schemaname='public'")
    assert {"users", "sources", "jobs"} <= set(tables)


def test_migrations_match_the_models(config: Config) -> None:
    command.check(config)  # raises if the models differ from what the migrations build


def test_downgrade_removes_tables_and_enum_types(config: Config, test_database_url: str) -> None:
    command.downgrade(config, "base")

    tables = query(test_database_url, "SELECT tablename FROM pg_tables WHERE schemaname='public'")
    enum_types = query(test_database_url, "SELECT typname FROM pg_type WHERE typtype = 'e'")
    assert tables == ["alembic_version"]
    assert enum_types == []


def test_upgrade_after_downgrade_lands_on_the_latest_revision(
    config: Config, test_database_url: str
) -> None:
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    version = query(test_database_url, "SELECT version_num FROM alembic_version")
    assert version == [ScriptDirectory.from_config(config).get_current_head()]
