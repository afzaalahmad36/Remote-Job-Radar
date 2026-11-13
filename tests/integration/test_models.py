import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CvFormat,
    ExtractionMethod,
    ExtractionStatus,
    Job,
    LlmEnrichment,
    LlmUsageEntry,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
    Source,
    SourceType,
    User,
)


async def make_user(session: AsyncSession) -> User:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="x")
    session.add(user)
    await session.flush()
    return user


async def make_source(
    session: AsyncSession, name: str = "feed", owner: User | None = None
) -> Source:
    owner = owner or await make_user(session)
    source = Source(user_id=owner.id, type=SourceType.RSS, name=name)
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
    assert job.llm_enrichment == LlmEnrichment.NOT_NEEDED
    assert job.extraction_method is None  # "the rules phase has not run"
    assert (job.extraction_attempts, job.extraction_retries) == (0, 0)
    assert job.raw_text_hash is None and job.salary_period is None and job.seniority is None
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
        "extraction_method": [m.value for m in ExtractionMethod],
        "llm_enrichment": [m.value for m in LlmEnrichment],
        "salary_period": [m.value for m in SalaryPeriod],
        "seniority": [m.value for m in Seniority],
        "cv_format": [m.value for m in CvFormat],
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
    assert [m.value for m in ExtractionStatus] == [
        "pending",
        "extracted",
        "extraction_failed",
        "not_a_job",
    ]
    assert [m.value for m in ExtractionMethod] == ["rules", "rules_llm"]
    assert [m.value for m in LlmEnrichment] == ["not_needed", "pending", "done", "skipped"]
    assert [m.value for m in SalaryPeriod] == ["year", "month", "week", "day", "hour"]


async def test_database_rejects_unknown_enum_value(db_session: AsyncSession) -> None:
    owner = await make_user(db_session)  # so the enum is the only thing wrong with the row

    with pytest.raises(DBAPIError, match="invalid input value for enum source_type"):
        await db_session.execute(
            text(
                "INSERT INTO sources (id, user_id, type, name) "
                "VALUES (gen_random_uuid(), :user_id, 'bogus', 'x')"
            ),
            {"user_id": owner.id},
        )


async def test_a_source_must_have_an_owner(db_session: AsyncSession) -> None:
    db_session.add(Source(type=SourceType.RSS, name="orphan"))

    with pytest.raises(IntegrityError, match="user_id"):
        await db_session.flush()


async def test_deleting_a_user_deletes_their_sources_and_jobs(db_session: AsyncSession) -> None:
    owner = await make_user(db_session)
    source = await make_source(db_session, owner=owner)
    db_session.add(make_job(source))
    await db_session.flush()

    await db_session.delete(owner)
    await db_session.flush()

    assert (await db_session.scalars(select(Source))).all() == []
    assert (await db_session.scalars(select(Job))).all() == []


# --- extraction state ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [ExtractionStatus.PENDING, ExtractionStatus.EXTRACTION_FAILED, ExtractionStatus.NOT_A_JOB],
)
async def test_only_an_extracted_job_must_have_a_title(
    db_session: AsyncSession, status: ExtractionStatus
) -> None:
    job = make_job(await make_source(db_session))
    job.extraction_status = status
    db_session.add(job)

    await db_session.flush()  # no title, and no error


async def test_an_extracted_job_with_a_title_is_accepted(db_session: AsyncSession) -> None:
    job = make_job(await make_source(db_session))
    job.extraction_status, job.title = ExtractionStatus.EXTRACTED, "Python developer"
    db_session.add(job)

    await db_session.flush()


async def test_the_database_rejects_an_extracted_job_without_a_title(
    db_session: AsyncSession,
) -> None:
    job = make_job(await make_source(db_session))
    job.extraction_status = ExtractionStatus.EXTRACTED
    db_session.add(job)

    with pytest.raises(IntegrityError, match="ck_jobs_extracted_has_title"):
        await db_session.flush()


async def test_a_job_cannot_lose_its_title_while_extracted(db_session: AsyncSession) -> None:
    job = make_job(await make_source(db_session))
    job.extraction_status, job.title = ExtractionStatus.EXTRACTED, "Python developer"
    db_session.add(job)
    await db_session.flush()

    job.title = None

    with pytest.raises(IntegrityError, match="ck_jobs_extracted_has_title"):
        await db_session.flush()


async def test_seniority_salary_period_and_state_round_trip(db_session: AsyncSession) -> None:
    job = make_job(await make_source(db_session))
    job.seniority, job.salary_period = Seniority.SENIOR, SalaryPeriod.MONTH
    job.extraction_method, job.llm_enrichment = ExtractionMethod.RULES_LLM, LlmEnrichment.DONE
    db_session.add(job)
    await db_session.flush()
    await db_session.refresh(job)  # read the values back from the database

    assert (job.seniority, job.salary_period) == (Seniority.SENIOR, SalaryPeriod.MONTH)
    assert (job.extraction_method, job.llm_enrichment) == (
        ExtractionMethod.RULES_LLM,
        LlmEnrichment.DONE,
    )


async def test_database_rejects_a_seniority_that_is_not_in_the_enum(
    db_session: AsyncSession,
) -> None:
    source = await make_source(db_session)

    with pytest.raises(DBAPIError, match="invalid input value for enum seniority"):
        await db_session.execute(
            text(
                "INSERT INTO jobs (id, source_id, external_id, raw_text, seniority) "
                "VALUES (gen_random_uuid(), :source_id, 'x', 'x', 'wizard')"
            ),
            {"source_id": source.id},
        )


async def test_the_sweep_indexes_are_partial(db_session: AsyncSession) -> None:
    rows = await db_session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'jobs'")
    )
    definitions = dict(rows.all())

    assert "WHERE (extraction_method IS NULL)" in definitions["ix_jobs_rules_not_run"]
    assert "WHERE (llm_enrichment = 'pending'" in definitions["ix_jobs_enrichment_pending"]
    assert "raw_text_hash" in definitions["ix_jobs_raw_text_hash"]


async def test_a_conditional_update_changes_a_row_only_from_the_expected_state(
    db_session: AsyncSession,
) -> None:
    """The mechanism the service will rely on: the second of two racing writers matches no row."""
    job = make_job(await make_source(db_session))
    db_session.add(job)
    await db_session.flush()
    claim = text(
        "UPDATE jobs SET llm_enrichment = 'done' "
        "WHERE id = :id AND llm_enrichment = 'not_needed' RETURNING id"
    )

    first = await db_session.execute(claim, {"id": job.id})
    second = await db_session.execute(claim, {"id": job.id})

    assert first.all() != [] and second.all() == []


# --- llm_usage ----------------------------------------------------------------------------------


def make_usage(user: User, job: Job | None = None) -> LlmUsageEntry:
    return LlmUsageEntry(
        user_id=user.id,
        job_id=job.id if job else None,
        provider="gemini",
        model="some-model",
        prompt_version="extract_job_v1",
        outcome="ok",
        input_tokens=1500,
    )


async def test_a_usage_row_has_a_timestamp_and_nullable_token_counts(
    db_session: AsyncSession,
) -> None:
    user = await make_user(db_session)
    entry = make_usage(user)
    db_session.add(entry)
    await db_session.flush()
    await db_session.refresh(entry)

    assert entry.created_at.utcoffset() == timedelta(0)
    assert (entry.input_tokens, entry.output_tokens, entry.thought_tokens) == (1500, None, None)


async def test_deleting_a_user_deletes_their_usage_rows(db_session: AsyncSession) -> None:
    user = await make_user(db_session)
    db_session.add(make_usage(user))
    await db_session.flush()

    await db_session.delete(user)
    await db_session.flush()

    assert (await db_session.scalars(select(LlmUsageEntry))).all() == []


async def test_deleting_a_job_keeps_the_usage_row_without_the_job(
    db_session: AsyncSession,
) -> None:
    source = await make_source(db_session)
    job = make_job(source)
    user = await db_session.get_one(User, source.user_id)
    db_session.add(job)
    await db_session.flush()
    entry = make_usage(user, job)
    db_session.add(entry)
    await db_session.flush()

    await db_session.delete(job)
    await db_session.flush()
    await db_session.refresh(entry)

    assert entry.job_id is None  # the request was still made, so it still counts


async def test_a_usage_row_must_have_a_user(db_session: AsyncSession) -> None:
    db_session.add(LlmUsageEntry(provider="gemini", model="m", prompt_version="v1", outcome="ok"))

    with pytest.raises(IntegrityError, match="user_id"):
        await db_session.flush()


async def test_usage_has_the_indexes_the_budget_queries_need(db_session: AsyncSession) -> None:
    rows = await db_session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'llm_usage'")
    )
    definitions = dict(rows.all())

    assert "(user_id, created_at)" in definitions["ix_llm_usage_user_id_created_at"]
    assert "(created_at)" in definitions["ix_llm_usage_created_at"]
    assert "(job_id)" in definitions["ix_llm_usage_job_id"]
