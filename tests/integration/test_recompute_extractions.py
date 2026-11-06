"""`scripts/recompute_extractions.py` against a real Postgres. No provider, no network.

These tests commit (two sessions must see each other), and the `factory` fixture deletes
everything they create. Each test builds its own jobs and looks only at them.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from app.models import ExtractionMethod, ExtractionStatus, Job, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.services.extraction_types import SessionFactory
from scripts.recompute_extractions import candidate_conditions, recompute
from tests.extraction_helpers import RICH, add_job, owner_of, row

CURRENT = "rules_v2"
OLD_RULES = {
    "extraction_method": ExtractionMethod.RULES,
    "extraction_rules_version": "rules_v1",
    "extraction_status": ExtractionStatus.EXTRACTED,
    "title": "Backend Engineer",
    "llm_enrichment": LlmEnrichment.NOT_NEEDED,
    "extracted_at": datetime(2026, 11, 1, tzinfo=UTC),
}


def old_rules(**changes: Any) -> dict[str, Any]:
    return {**OLD_RULES, **changes}


async def reset_state(factory: SessionFactory, job_id: uuid.UUID) -> tuple[object, ...]:
    job = await row(factory, job_id)
    return (job.extraction_status, job.llm_enrichment, job.extraction_error, job.extraction_method)


async def test_an_old_rules_only_job_is_reset_without_touching_its_counters(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, RICH, **old_rules(extraction_retries=3, extraction_attempts=7))

    result = await recompute(factory, current_version=CURRENT)

    job = await row(factory, job_id)
    assert result.reset >= 1
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    )
    assert job.extraction_method is None and job.extraction_error is None
    assert (job.extraction_retries, job.extraction_attempts) == (3, 7)  # not touched
    assert job.title == "Backend Engineer"  # the old values stay until the rules run again
    assert job.extraction_rules_version == "rules_v1"


async def test_a_failed_or_skipped_old_rules_job_is_reset_too(factory: SessionFactory) -> None:
    failed = await add_job(
        factory,
        RICH,
        **old_rules(
            extraction_status=ExtractionStatus.EXTRACTION_FAILED,
            title=None,
            llm_enrichment=LlmEnrichment.SKIPPED,
            extraction_error="no title found; the model is not enabled",
        ),
    )

    await recompute(factory, current_version=CURRENT)

    assert await reset_state(factory, failed) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
        None,
        None,
    )


async def test_rows_that_are_not_old_rules_only_rows_are_never_touched(
    factory: SessionFactory,
) -> None:
    untouchable = {
        "current version": old_rules(extraction_rules_version="rules_v2"),
        "a newer version": old_rules(extraction_rules_version="rules_v3"),
        "enriched by the model": old_rules(
            extraction_method=ExtractionMethod.RULES_LLM,
            llm_enrichment=LlmEnrichment.DONE,
            extraction_model="some-model",
            extraction_prompt_version="extract_job_v1",
        ),
        "enrichment done": old_rules(llm_enrichment=LlmEnrichment.DONE),
        "a model name on a rules row": old_rules(extraction_model="some-model"),
        "a prompt version on a rules row": old_rules(extraction_prompt_version="extract_job_v1"),
        "rules phase not run": old_rules(
            extraction_method=None,
            extraction_status=ExtractionStatus.PENDING,
            extraction_rules_version=None,
            title=None,
        ),
        "an odd version": old_rules(extraction_rules_version="rules_vX"),
        "a huge version": old_rules(extraction_rules_version="rules_v" + "9" * 30),
        "no version": old_rules(extraction_rules_version=None),
    }
    ids = {name: await add_job(factory, RICH, **fields) for name, fields in untouchable.items()}
    before = {name: await reset_state(factory, job_id) for name, job_id in ids.items()}

    await recompute(factory, current_version=CURRENT)

    for name, job_id in ids.items():
        assert await reset_state(factory, job_id) == before[name], name


async def test_a_dry_run_counts_and_changes_nothing(factory: SessionFactory) -> None:
    baseline = (await recompute(factory, current_version=CURRENT, dry_run=True)).candidates.total
    job_ids = [await add_job(factory, RICH, **old_rules()) for _ in range(3)]

    result = await recompute(factory, current_version=CURRENT, dry_run=True)

    assert result.candidates.total == baseline + 3
    assert result.batches == [] and result.reset == 0
    assert ("rules_v1", "extracted", "not_needed") in {
        (version, status, enrichment)
        for version, status, enrichment, _ in result.candidates.breakdown
    }
    for job_id in job_ids:
        assert (await row(factory, job_id)).extraction_method is ExtractionMethod.RULES


async def test_rows_are_reset_in_batches_and_each_batch_is_logged(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = (await recompute(factory, current_version=CURRENT, dry_run=True)).candidates.total
    for _ in range(5):
        await add_job(factory, RICH, **old_rules())

    with caplog.at_level(logging.INFO, logger="recompute_extractions"):
        result = await recompute(factory, current_version=CURRENT, batch_size=2)

    assert sum(result.batches) == baseline + 5
    assert all(0 < size <= 2 for size in result.batches)
    assert len(result.batches) >= 3
    logged = [r.getMessage() for r in caplog.records if r.name == "recompute_extractions"]
    assert logged == [f"batch {i}: reset {n} row(s)" for i, n in enumerate(result.batches, 1)]


async def test_running_it_twice_resets_nothing_the_second_time(factory: SessionFactory) -> None:
    await add_job(factory, RICH, **old_rules())
    await recompute(factory, current_version=CURRENT)

    second = await recompute(factory, current_version=CURRENT)

    assert second.candidates.total == 0 and second.batches == []


async def test_the_reset_matches_what_reset_for_retry_does_to_the_same_columns(
    factory: SessionFactory,
) -> None:
    fields = old_rules(
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        title=None,
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_error="model request was rejected",
    )
    retried = await add_job(factory, RICH, **fields)
    recomputed = await add_job(factory, RICH, **fields)
    owner = await owner_of(factory, retried)
    async with factory() as session, session.begin():
        await repository.reset_for_retry(session, user_id=owner, job_id=retried, max_retries=5)

    await recompute(factory, current_version=CURRENT)

    assert await reset_state(factory, recomputed) == await reset_state(factory, retried)
    assert (await row(factory, retried)).extraction_retries == 1  # only the endpoint counts
    assert (await row(factory, recomputed)).extraction_retries == 0


async def test_a_row_locked_by_a_worker_is_left_for_a_later_run(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH, **old_rules())

    async with factory() as worker, worker.begin():
        await worker.execute(select(Job.id).where(Job.id == job_id).with_for_update())
        await recompute(factory, current_version=CURRENT)
        assert (await row(factory, job_id)).extraction_method is ExtractionMethod.RULES

    await recompute(factory, current_version=CURRENT)

    assert (await row(factory, job_id)).extraction_method is None


def test_the_conditions_cover_exactly_the_documented_columns() -> None:
    text = " ".join(str(condition) for condition in candidate_conditions(2))

    for column in (
        "extraction_method",
        "extraction_rules_version",
        "llm_enrichment",
        "extraction_model",
        "extraction_prompt_version",
    ):
        assert column in text
    assert "extraction_retries" not in text
