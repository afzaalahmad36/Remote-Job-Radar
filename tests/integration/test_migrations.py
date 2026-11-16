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
    assert {"users", "sources", "jobs", "cvs", "job_preferences"} <= set(tables)


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


# --- the extraction-state migration -------------------------------------------------------------

BEFORE_STATE = "7bb837af91cb"  # the revision just before extraction state was added
INSERT_JOB = (
    "WITH u AS (INSERT INTO users (id, email, hashed_password) "
    "  VALUES (gen_random_uuid(), 'm@example.com', 'x') RETURNING id), "
    "s AS (INSERT INTO sources (id, user_id, type, name) "
    "  SELECT gen_random_uuid(), id, 'rss', 'feed' FROM u RETURNING id) "
    "INSERT INTO jobs (id, source_id, external_id, raw_text{columns}) "
    "SELECT gen_random_uuid(), id, 'post-1', 'Hiring a Python developer'{values} FROM s"
)


def execute(database_url: str, sql: str) -> None:
    async def run() -> None:
        engine = create_async_engine(database_url, poolclass=NullPool)
        async with engine.begin() as connection:
            await connection.execute(text(sql))
        await engine.dispose()

    asyncio.run(run())


def clear_data(database_url: str) -> None:
    execute(database_url, "DELETE FROM users")  # cascades to sources and jobs


def test_existing_jobs_keep_their_data_and_wait_for_the_rules(
    config: Config, test_database_url: str
) -> None:
    command.downgrade(config, BEFORE_STATE)
    try:
        execute(test_database_url, INSERT_JOB.format(columns="", values=""))

        command.upgrade(config, "head")

        row = query(
            test_database_url,
            "SELECT concat_ws('|', extraction_status, llm_enrichment, extraction_attempts, "
            "extraction_retries, coalesce(extraction_method::text, 'NULL'), raw_text) FROM jobs",
        )
        assert row == ["pending|not_needed|0|0|NULL|Hiring a Python developer"]
        # ...and the partial index predicate matches it: this is how the sweep finds it.
        waiting = query(
            test_database_url, "SELECT count(*) FROM jobs WHERE extraction_method IS NULL"
        )
        assert waiting == ["1"]
    finally:
        clear_data(test_database_url)


def test_the_upgrade_stops_if_a_job_already_has_a_seniority(
    config: Config, test_database_url: str
) -> None:
    command.downgrade(config, BEFORE_STATE)
    try:
        execute(
            test_database_url,
            INSERT_JOB.format(columns=", seniority", values=", 'senior'"),
        )

        with pytest.raises(Exception, match="jobs.seniority has values"):
            command.upgrade(config, "head")

        # Nothing was half-applied: still on the old revision, and seniority is still text.
        assert query(test_database_url, "SELECT version_num FROM alembic_version") == [BEFORE_STATE]
        assert query(test_database_url, "SELECT seniority FROM jobs") == ["senior"]
    finally:
        clear_data(test_database_url)


def test_a_one_step_downgrade_undoes_the_schema_but_keeps_the_not_a_job_label(
    config: Config, test_database_url: str
) -> None:
    execute(
        test_database_url,
        INSERT_JOB.format(columns=", extraction_status", values=", 'not_a_job'"),
    )
    try:
        command.downgrade(config, BEFORE_STATE)

        assert "llm_usage" not in query(
            test_database_url, "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        columns = query(
            test_database_url,
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'jobs'",
        )
        assert not {"salary_period", "llm_enrichment", "extraction_retries"} & set(columns)
        # Older code never sees an unknown status: the row moved to extraction_failed.
        assert query(test_database_url, "SELECT extraction_status FROM jobs") == [
            "extraction_failed"
        ]
        # The enum value cannot be removed from the type (documented in the migration).
        labels = query(
            test_database_url,
            "SELECT enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
            "WHERE t.typname = 'extraction_status'",
        )
        assert "not_a_job" in labels
        # The new enum types are gone, and the seniority column is text again.
        assert (
            query(
                test_database_url,
                "SELECT typname FROM pg_type WHERE typname IN "
                "('salary_period', 'extraction_method', 'llm_enrichment', 'seniority')",
            )
            == []
        )
        assert query(
            test_database_url,
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'jobs' AND column_name = 'seniority'",
        ) == ["character varying"]

        command.upgrade(config, "head")  # ADD VALUE IF NOT EXISTS: the leftover label is fine
    finally:
        clear_data(test_database_url)


# --- the skills and preferences migration -------------------------------------------------------

BEFORE_SKILLS = "f29a32b4831a"  # the revision just before cv skills and job preferences
INSERT_CV = (
    "WITH u AS (INSERT INTO users (id, email, hashed_password) "
    "  VALUES (gen_random_uuid(), 'cv@example.com', 'x') RETURNING id) "
    "INSERT INTO cvs (id, user_id, text, source_format, char_count, content_hash) "
    "SELECT gen_random_uuid(), id, 'Python developer', 'text', 16, repeat('0', 64) FROM u"
)


def test_a_cv_stored_before_the_migration_gets_no_skills_and_is_not_marked_edited(
    config: Config, test_database_url: str
) -> None:
    command.downgrade(config, BEFORE_SKILLS)
    try:
        execute(test_database_url, INSERT_CV)

        command.upgrade(config, "head")

        row = query(
            test_database_url, "SELECT concat_ws('|', skills, skills_edited, text) FROM cvs"
        )
        assert row == ["{}|f|Python developer"]
    finally:
        clear_data(test_database_url)


def test_downgrading_the_skills_migration_drops_its_columns_and_table_but_not_the_job_enums(
    config: Config, test_database_url: str
) -> None:
    command.downgrade(config, BEFORE_SKILLS)

    tables = query(test_database_url, "SELECT tablename FROM pg_tables WHERE schemaname='public'")
    columns = query(
        test_database_url,
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'cvs'",
    )
    enums = query(test_database_url, "SELECT typname FROM pg_type WHERE typtype = 'e'")
    assert "job_preferences" not in tables and "cvs" in tables
    assert not {"skills", "skills_edited"} & set(columns)
    assert {"seniority", "remote_policy", "salary_period"} <= set(enums)  # still used by jobs
