"""The rules phase against a real Postgres. These tests commit their data (two workers need two
real transactions), so each one deletes what it created."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.extraction.plan import ExtractionError, plan_rules_phase
from app.extraction.rules import RULES_VERSION
from app.extraction.text_hash import raw_text_hash
from app.llm.prompts import EXTRACT_JOB_V1
from app.llm.providers.fake import FakeLLMClient
from app.models import (
    ExtractionMethod,
    ExtractionStatus,
    Job,
    LlmEnrichment,
    Source,
    SourceType,
    User,
)
from app.repositories import extraction_repository as repository
from app.services import extraction_service
from app.services.extraction_service import ExtractionOutcome, SessionFactory, extract_job
from tests.settings_helpers import settings_with

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
SETTINGS = settings_with(llm_min_post_chars=500)

RICH = (
    "Senior Python Developer at Acme\n\nWe are hiring a Senior Python Developer for our remote "
    "team.\nStack: Python, Django, PostgreSQL, Docker.\nApply: https://acme.com/careers/senior"
)
TITLE_ONLY_SHORT = "Marketing Manager at Acme\n\nWe are hiring a Marketing Manager."
TITLE_ONLY_LONG = TITLE_ONLY_SHORT + "\n\n" + "We build great products for happy customers. " * 20
NOTHING_SHORT = "lorem ipsum dolor sit amet " * 4
NOTHING_LONG = "lorem ipsum dolor sit amet " * 40


@pytest.fixture
async def factory(test_database_url: str, migrated_db: None) -> AsyncIterator[SessionFactory]:
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    yield sessions
    async with sessions() as session, session.begin():
        await session.execute(delete(User).where(User.email.like("%@extraction.test")))  # cascades
    await engine.dispose()


async def add_job(factory: SessionFactory, text: str, **fields: Any) -> uuid.UUID:
    """A job in its own new user and source (so jobs never share an owner by accident)."""
    async with factory() as session, session.begin():
        user = User(email=f"{uuid.uuid4()}@extraction.test", hashed_password="x")
        session.add(user)
        await session.flush()
        source = Source(user_id=user.id, type=SourceType.RSS, name="feed")
        session.add(source)
        await session.flush()
        job = Job(source_id=source.id, external_id="post-1", raw_text=text, **fields)
        session.add(job)
        await session.flush()
        return job.id


async def row(factory: SessionFactory, job_id: uuid.UUID) -> Job:
    async with factory() as session:
        return await session.get_one(Job, job_id)


async def run(factory: SessionFactory, job_id: uuid.UUID, llm: Any = None) -> ExtractionOutcome:
    return await extract_job(factory, job_id, settings=SETTINGS, llm=llm, now=NOW)


# --- the rules phase ----------------------------------------------------------------------------


async def test_a_complete_post_is_extracted_by_the_rules_alone(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH)

    assert await run(factory, job_id) is ExtractionOutcome.EXTRACTED

    job = await row(factory, job_id)
    assert job.extraction_status is ExtractionStatus.EXTRACTED
    assert job.llm_enrichment is LlmEnrichment.NOT_NEEDED
    assert job.extraction_method is ExtractionMethod.RULES
    assert (job.title, job.stack) == (
        "Senior Python Developer",
        ["Python", "Django", "PostgreSQL", "Docker"],
    )
    assert job.raw_text_hash == raw_text_hash(RICH)
    assert job.extraction_rules_version == RULES_VERSION
    assert (job.extraction_prompt_version, job.extraction_model) == (None, None)
    assert (job.extracted_at, job.extraction_attempts, job.extraction_error) == (NOW, 1, None)


async def test_a_job_with_a_title_is_usable_at_once_and_waits_for_a_model(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, TITLE_ONLY_LONG)

    assert await run(factory, job_id) is ExtractionOutcome.AWAITING_MODEL

    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
    )
    assert job.title == "Marketing Manager"


async def test_no_model_is_called_in_this_step_even_when_one_is_available(
    factory: SessionFactory,
) -> None:
    llm = FakeLLMClient()
    waiting_job = await add_job(factory, NOTHING_LONG)

    assert await run(factory, waiting_job, llm) is ExtractionOutcome.AWAITING_MODEL

    job = await row(factory, waiting_job)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.PENDING,
    )
    assert job.title is None and job.extraction_error is None
    assert llm.requests == []


async def test_a_short_post_with_a_title_is_extracted_and_the_model_is_skipped(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, TITLE_ONLY_SHORT)

    assert await run(factory, job_id) is ExtractionOutcome.EXTRACTED

    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.SKIPPED,
    )
    assert job.extraction_error == ExtractionError.TOO_SHORT.value


async def test_no_title_and_no_model_fails_at_once_with_a_reason(factory: SessionFactory) -> None:
    job_id = await add_job(factory, NOTHING_LONG)

    assert await run(factory, job_id, llm=None) is ExtractionOutcome.FAILED

    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTION_FAILED,
        LlmEnrichment.SKIPPED,
    )
    assert job.extraction_error == ExtractionError.MODEL_NOT_ENABLED.value
    assert job.title is None and job.extraction_method is ExtractionMethod.RULES


async def test_no_title_in_a_short_post_fails_even_when_a_model_exists(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, NOTHING_SHORT)

    assert await run(factory, job_id, FakeLLMClient()) is ExtractionOutcome.FAILED

    assert (await row(factory, job_id)).extraction_error == ExtractionError.TOO_SHORT.value


async def test_a_missing_job_is_not_an_error(factory: SessionFactory) -> None:
    assert await run(factory, uuid.uuid4()) is ExtractionOutcome.NOT_FOUND


# --- running again -----------------------------------------------------------------------------


async def test_running_a_finished_job_again_changes_nothing(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH)
    await run(factory, job_id)
    before = await row(factory, job_id)

    assert await run(factory, job_id) is ExtractionOutcome.ALREADY_DONE

    after = await row(factory, job_id)
    assert (after.extraction_attempts, after.extracted_at) == (1, NOW)
    assert after.updated_at == before.updated_at  # not even touched


async def test_a_job_waiting_for_the_model_is_left_alone_by_this_step(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, TITLE_ONLY_LONG)
    await run(factory, job_id)

    assert await run(factory, job_id) is ExtractionOutcome.AWAITING_MODEL

    assert (await row(factory, job_id)).extraction_attempts == 1


@pytest.mark.parametrize(
    ("status", "enrichment"),
    [
        (ExtractionStatus.EXTRACTED, LlmEnrichment.DONE),
        (ExtractionStatus.EXTRACTION_FAILED, LlmEnrichment.SKIPPED),
        (ExtractionStatus.NOT_A_JOB, LlmEnrichment.DONE),
    ],
)
async def test_every_finished_state_is_left_alone(
    factory: SessionFactory, status: ExtractionStatus, enrichment: LlmEnrichment
) -> None:
    title = "Kept title" if status is ExtractionStatus.EXTRACTED else None
    job_id = await add_job(
        factory, RICH, extraction_status=status, llm_enrichment=enrichment, title=title
    )

    assert await run(factory, job_id) is ExtractionOutcome.ALREADY_DONE

    assert (await row(factory, job_id)).title == title


async def test_a_revived_job_is_rewritten_from_scratch_by_the_rules(
    factory: SessionFactory,
) -> None:
    """The state a retry leaves behind: pending / not_needed, with an old model result."""
    job_id = await add_job(
        factory,
        RICH,
        title="Old model title",
        extraction_method=ExtractionMethod.RULES_LLM,
        extraction_prompt_version="extract_job_v1",
        extraction_model="some-model",
        extraction_error="model reply was not valid",
        extraction_attempts=3,
    )

    assert await run(factory, job_id) is ExtractionOutcome.EXTRACTED

    job = await row(factory, job_id)
    assert job.title == "Senior Python Developer"
    assert job.extraction_method is ExtractionMethod.RULES
    assert (job.extraction_prompt_version, job.extraction_model) == (None, None)
    assert (job.extraction_error, job.extraction_attempts) == (None, 4)


async def test_a_crash_in_the_rules_is_stored_as_a_failure_without_leaking_the_cause(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(text: str, **_: Any) -> Any:
        raise RuntimeError("boom: secret post text")

    monkeypatch.setattr(extraction_service, "plan_rules_phase", broken)
    job_id = await add_job(factory, RICH)

    with caplog.at_level(logging.ERROR, logger="app.services.extraction_service"):
        assert await run(factory, job_id) is ExtractionOutcome.FAILED

    job = await row(factory, job_id)
    assert job.extraction_status is ExtractionStatus.EXTRACTION_FAILED
    assert job.llm_enrichment is LlmEnrichment.SKIPPED
    assert job.extraction_error == ExtractionError.UNEXPECTED.value
    assert "boom" not in (job.extraction_error or "")
    [record] = [r for r in caplog.records if r.name == "app.services.extraction_service"]
    assert record.levelno == logging.ERROR
    assert record.getMessage() == (
        f"The rules phase raised for job {job_id} (error_type=RuntimeError)"
    )
    assert record.exc_info is None  # no traceback, which could quote the post
    for leaked in ("boom", "secret post text", "Senior Python Developer", "acme.com"):
        assert leaked not in caplog.text


# --- reusing an extraction of identical text ---------------------------------------------------


def finished_donor_fields(**overrides: Any) -> dict[str, Any]:
    """A job a model enriched: title 'Model Title' (the rules alone would not produce it)."""
    fields: dict[str, Any] = {
        "raw_text_hash": raw_text_hash(TITLE_ONLY_LONG),
        "title": "Model Title",
        "company": "Model Co",
        "stack": ["Go"],
        "extraction_status": ExtractionStatus.EXTRACTED,
        "llm_enrichment": LlmEnrichment.DONE,
        "extraction_method": ExtractionMethod.RULES_LLM,
        "extraction_rules_version": RULES_VERSION,
        "extraction_prompt_version": EXTRACT_JOB_V1.version,
        "extraction_model": "some-model",
        "extracted_at": NOW,
    }
    return fields | overrides


async def test_identical_text_reuses_a_model_enriched_extraction(factory: SessionFactory) -> None:
    await add_job(factory, TITLE_ONLY_LONG, **finished_donor_fields())
    spaced = "  " + TITLE_ONLY_LONG.replace("\n\n", "\n \n") + "\n"  # same text, other whitespace
    job_id = await add_job(factory, spaced)

    assert await run(factory, job_id) is ExtractionOutcome.REUSED

    job = await row(factory, job_id)
    assert (job.title, job.company, job.stack) == ("Model Title", "Model Co", ["Go"])
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.DONE,
    )
    assert job.extraction_method is ExtractionMethod.RULES_LLM
    assert (job.extraction_prompt_version, job.extraction_model) == (
        EXTRACT_JOB_V1.version,
        "some-model",
    )
    assert job.raw_text_hash == raw_text_hash(TITLE_ONLY_LONG)
    assert (job.extracted_at, job.extraction_attempts, job.extraction_error) == (NOW, 1, None)


async def test_reuse_copies_extraction_fields_only_never_identity(
    factory: SessionFactory,
) -> None:
    donor_id = await add_job(factory, TITLE_ONLY_LONG, **finished_donor_fields())
    job_id = await add_job(factory, TITLE_ONLY_LONG)
    before, donor = await row(factory, job_id), await row(factory, donor_id)

    await run(factory, job_id)

    after = await row(factory, job_id)
    assert after.id != donor.id and after.source_id != donor.source_id
    assert (after.id, after.source_id, after.external_id) == (
        before.id,
        before.source_id,
        before.external_id,
    )
    assert (after.raw_text, after.posted_at, after.created_at) == (
        before.raw_text,
        before.posted_at,
        before.created_at,
    )
    assert after.extraction_retries == 0


async def test_reuse_prefers_the_model_enriched_extraction_over_a_rules_only_one(
    factory: SessionFactory,
) -> None:
    await add_job(
        factory,
        TITLE_ONLY_LONG,
        **finished_donor_fields(
            title="Rules Title",
            llm_enrichment=LlmEnrichment.NOT_NEEDED,
            extraction_method=ExtractionMethod.RULES,
            extraction_prompt_version=None,
            extraction_model=None,
            extracted_at=datetime(2026, 11, 3, 13, 0, tzinfo=UTC),  # newer than the other donor
        ),
    )
    await add_job(factory, TITLE_ONLY_LONG, **finished_donor_fields())
    job_id = await add_job(factory, TITLE_ONLY_LONG)

    await run(factory, job_id)

    assert (await row(factory, job_id)).title == "Model Title"


@pytest.mark.parametrize(
    "unusable",
    [
        {"extraction_method": None},  # the rules phase never ran
        {"llm_enrichment": LlmEnrichment.PENDING},  # an answer is still outstanding
        {"llm_enrichment": LlmEnrichment.SKIPPED},
        {"extraction_status": ExtractionStatus.EXTRACTION_FAILED, "title": None},
        {"extraction_status": ExtractionStatus.PENDING, "title": None},
        {"extraction_rules_version": "rules_v0"},  # made by older rules
        {"extraction_prompt_version": "extract_job_v0"},  # made by an older prompt
        {"raw_text_hash": raw_text_hash("a different post")},
    ],
)
async def test_an_unfinished_stale_or_different_job_is_not_reused(
    factory: SessionFactory, unusable: dict[str, Any]
) -> None:
    await add_job(factory, TITLE_ONLY_LONG, **finished_donor_fields(**unusable))
    job_id = await add_job(factory, TITLE_ONLY_LONG)

    outcome = await run(factory, job_id)

    assert outcome is ExtractionOutcome.AWAITING_MODEL  # worked out by the rules, not copied
    job = await row(factory, job_id)
    assert job.title == "Marketing Manager" and job.extraction_method is ExtractionMethod.RULES


async def test_a_job_never_reuses_itself(factory: SessionFactory) -> None:
    job_id = await add_job(
        factory,
        TITLE_ONLY_LONG,
        **finished_donor_fields(
            extraction_status=ExtractionStatus.PENDING,
            llm_enrichment=LlmEnrichment.NOT_NEEDED,
            title=None,
        ),
    )

    assert await run(factory, job_id) is ExtractionOutcome.AWAITING_MODEL


async def test_a_not_a_job_verdict_is_reused_too(factory: SessionFactory) -> None:
    await add_job(
        factory,
        NOTHING_LONG,
        **finished_donor_fields(
            raw_text_hash=raw_text_hash(NOTHING_LONG),
            extraction_status=ExtractionStatus.NOT_A_JOB,
            title=None,
            company=None,
            stack=[],
        ),
    )
    job_id = await add_job(factory, NOTHING_LONG)

    assert await run(factory, job_id) is ExtractionOutcome.REUSED

    assert (await row(factory, job_id)).extraction_status is ExtractionStatus.NOT_A_JOB


# --- two workers racing ------------------------------------------------------------------------


async def snapshot_of(factory: SessionFactory, job_id: uuid.UUID) -> repository.JobSnapshot:
    async with factory() as session, session.begin():
        snapshot = await repository.load_snapshot(session, job_id)
    assert snapshot is not None
    return snapshot


async def test_of_two_workers_holding_the_same_snapshot_exactly_one_writes(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, RICH)
    snapshot = await snapshot_of(factory, job_id)
    plan = plan_rules_phase(RICH, can_enrich=False, min_chars=500)

    async def attempt() -> bool:
        async with factory() as session, session.begin():
            return await repository.write_rules_result(session, snapshot, plan, NOW)

    results = await asyncio.gather(attempt(), attempt())

    assert sorted(results) == [False, True]
    assert (await row(factory, job_id)).extraction_attempts == 1


async def test_a_stale_snapshot_cannot_overwrite_a_newer_result(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH)
    stale = await snapshot_of(factory, job_id)
    await run(factory, job_id)  # another worker finishes first
    donor = repository.ReusableExtraction(
        title="Stale title",
        company=None,
        stack=(),
        seniority=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
        remote_policy=None,
        apply_email=None,
        apply_url=None,
        status=ExtractionStatus.EXTRACTED,
        enrichment=LlmEnrichment.DONE,
        method=ExtractionMethod.RULES_LLM,
        rules_version=RULES_VERSION,
        prompt_version=EXTRACT_JOB_V1.version,
        model="m",
    )

    async with factory() as session, session.begin():
        won = await repository.write_reused_result(session, stale, donor, "x" * 64, NOW)

    assert won is False
    job = await row(factory, job_id)
    assert job.title == "Senior Python Developer" and job.extraction_attempts == 1


async def test_two_full_runs_at_once_leave_one_consistent_result(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH)

    outcomes = await asyncio.gather(run(factory, job_id), run(factory, job_id))

    assert outcomes.count(ExtractionOutcome.EXTRACTED) == 1
    assert set(outcomes) <= {
        ExtractionOutcome.EXTRACTED,
        ExtractionOutcome.LOST_RACE,
        ExtractionOutcome.ALREADY_DONE,
    }
    job = await row(factory, job_id)
    assert (job.extraction_status, job.extraction_attempts) == (ExtractionStatus.EXTRACTED, 1)


async def test_ten_runs_at_once_still_write_exactly_once(factory: SessionFactory) -> None:
    job_id = await add_job(factory, TITLE_ONLY_LONG)

    outcomes = await asyncio.gather(*(run(factory, job_id) for _ in range(10)))

    assert outcomes.count(ExtractionOutcome.AWAITING_MODEL) >= 1
    assert (await row(factory, job_id)).extraction_attempts == 1


async def test_every_state_the_plan_can_produce_is_accepted_by_the_database(
    factory: SessionFactory,
) -> None:
    """The title check would raise if any case wrote 'extracted' without a title."""
    texts = [RICH, TITLE_ONLY_SHORT, TITLE_ONLY_LONG, NOTHING_SHORT, NOTHING_LONG, ""]
    job_ids = [await add_job(factory, text) for text in texts]

    for job_id in job_ids:
        await run(factory, job_id)

    statuses = {(await row(factory, job_id)).extraction_status for job_id in job_ids}
    assert statuses == {ExtractionStatus.EXTRACTED, ExtractionStatus.EXTRACTION_FAILED}
