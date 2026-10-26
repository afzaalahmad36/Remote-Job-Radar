from datetime import timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExtractionStatus, Job, RemotePolicy, Source, SourceType, User


async def make_source(session: AsyncSession, name: str = "feed") -> Source:
    source = Source(type=SourceType.RSS, name=name)
    session.add(source)
    await session.flush()
    return source


def make_job(source: Source, external_id: str = "post-1") -> Job:
    return Job(source_id=source.id, external_id=external_id, raw_text="Hiring a Python dev")


async def test_job_external_id_is_unique_per_source(db_session: AsyncSession) -> None:
    source = await make_source(db_session)
    db_session.add_all([make_job(source), make_job(source)])

    with pytest.raises(IntegrityError, match="uq_jobs_source_id_external_id"):
        await db_session.flush()


async def test_same_external_id_is_allowed_in_different_sources(db_session: AsyncSession) -> None:
    first = await make_source(db_session, "first")
    second = await make_source(db_session, "second")
    db_session.add_all([make_job(first), make_job(second)])

    await db_session.flush()  # no error


async def test_user_email_is_unique(db_session: AsyncSession) -> None:
    db_session.add_all(
        [
            User(email="a@example.com", hashed_password="x"),
            User(email="a@example.com", hashed_password="y"),
        ]
    )

    with pytest.raises(IntegrityError, match="uq_users_email"):
        await db_session.flush()


async def test_new_job_gets_defaults(db_session: AsyncSession) -> None:
    source = await make_source(db_session)
    job = make_job(source)
    db_session.add(job)
    await db_session.flush()
    await db_session.refresh(job)  # reload the values the database filled in

    assert job.extraction_status == ExtractionStatus.PENDING
    assert job.stack == []
    assert job.title is None
    assert job.created_at.utcoffset() == timedelta(0)  # timezone-aware, UTC
    assert job.updated_at.tzinfo is not None


async def test_source_defaults(db_session: AsyncSession) -> None:
    source = await make_source(db_session)
    await db_session.refresh(source)

    assert source.is_active is True
    assert source.config == {}
    assert source.last_synced_at is None


async def test_deleting_a_source_deletes_its_jobs(db_session: AsyncSession) -> None:
    source = await make_source(db_session)
    db_session.add(make_job(source))
    await db_session.flush()

    await db_session.delete(source)
    await db_session.flush()

    assert (await db_session.scalars(select(Job))).all() == []


async def test_postgres_enum_labels_match_python_enums(db_session: AsyncSession) -> None:
    rows = await db_session.execute(
        text(
            "SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) "
            "FROM pg_type t JOIN pg_enum e ON e.enumtypid = t.oid GROUP BY t.typname"
        )
    )

    assert dict(rows.all()) == {
        "source_type": [m.value for m in SourceType],
        "remote_policy": [m.value for m in RemotePolicy],
        "extraction_status": [m.value for m in ExtractionStatus],
    }


async def test_every_foreign_key_is_covered_by_an_index(db_session: AsyncSession) -> None:
    # An index "covers" a foreign key when the FK column is the index's first column.
    rows = await db_session.execute(
        text(
            "SELECT c.conrelid::regclass::text, a.attname FROM pg_constraint c "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] "
            "WHERE c.contype = 'f' AND NOT EXISTS ("
            "  SELECT 1 FROM pg_index i"
            "  WHERE i.indrelid = c.conrelid AND i.indkey[0] = c.conkey[1])"
        )
    )

    assert rows.all() == []  # any row here is a foreign key without an index


def test_enum_values() -> None:
    assert [m.value for m in SourceType] == ["telegram", "rss"]
    assert [m.value for m in RemotePolicy] == ["remote", "hybrid", "onsite"]
    assert [m.value for m in ExtractionStatus] == ["pending", "extracted", "extraction_failed"]


async def test_database_rejects_unknown_enum_value(db_session: AsyncSession) -> None:
    with pytest.raises(DBAPIError):
        await db_session.execute(
            text("INSERT INTO sources (id, type, name) VALUES (gen_random_uuid(), 'bogus', 'x')")
        )
