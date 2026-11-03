"""The extraction sweep: every step and its order, against a real Postgres, with fakes only."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.extraction.plan import ExtractionError
from app.llm.providers.fake import FakeLLMClient
from app.models import ExtractionMethod, ExtractionStatus, Job, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.services import extraction_sweep
from app.services.extraction_service import extract_job
from app.services.extraction_sweep import SWEEP_BATCH, collect_work
from app.services.extraction_types import ExtractionOutcome, SessionFactory, transaction
from app.services.llm_budget import DAY, DeferralReason
from app.tasks.jobs import run_sweep_extraction
from tests.extraction_helpers import (
    NEEDS_STACK,
    NOTHING_LONG,
    NOTHING_SHORT,
    NOW,
    RICH,
    add_job,
    add_usage,
    add_user,
    owner_of,
    row,
    usage_rows,
)
from tests.fakes import FakeCooldown
from tests.settings_helpers import settings_with
from tests.task_helpers import Seeder

SETTINGS = settings_with()  # give-up age 7 days; caps 100 / 150 / 5 per minute
WAITING = {
    "extraction_status": ExtractionStatus.EXTRACTED,
    "title": "Marketing Manager",
    "llm_enrichment": LlmEnrichment.PENDING,
    "extraction_method": ExtractionMethod.RULES,
}
TOO_OLD = NOW - timedelta(days=SETTINGS.llm_enrichment_give_up_days, seconds=1)


async def work(
    factory: SessionFactory,
    settings: Settings = SETTINGS,
    cooldown: FakeCooldown | None = None,
) -> extraction_sweep.SweepWork:
    return await collect_work(factory, settings=settings, cooldown=cooldown, now=NOW)


# --- step 1: reservations nobody finished ------------------------------------------------------


async def test_step_one_relabels_reservations_that_nobody_finished(factory: SessionFactory) -> None:
    user_id = await add_user(factory)
    await add_usage(factory, user_id, NOW - timedelta(minutes=30), outcome="reserved")
    await add_usage(factory, user_id, NOW - timedelta(minutes=1), outcome="reserved")

    result = await work(factory)

    assert result.abandoned == 1
    assert sorted(entry.outcome for entry in await usage_rows(factory)) == ["abandoned", "reserved"]


# --- step 2: giving up -------------------------------------------------------------------------


async def test_step_two_gives_up_on_jobs_that_waited_too_long_and_keeps_their_data(
    factory: SessionFactory,
) -> None:
    titled = await add_job(factory, NEEDS_STACK, created_at=TOO_OLD, **WAITING)
    untitled = await add_job(
        factory,
        NOTHING_LONG,
        created_at=TOO_OLD,
        extraction_status=ExtractionStatus.PENDING,
        llm_enrichment=LlmEnrichment.PENDING,
        extraction_method=ExtractionMethod.RULES,
    )

    result = await work(factory, cooldown=FakeCooldown())

    assert result.gave_up == 2
    kept, failed = await row(factory, titled), await row(factory, untitled)
    assert (kept.extraction_status, kept.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.SKIPPED,
    )
    assert kept.title == "Marketing Manager" and kept.stack == []  # the rules' data stays
    # No title and nothing left that could add one: the CASE turns it into a failure.
    assert (failed.extraction_status, failed.llm_enrichment) == (
        ExtractionStatus.EXTRACTION_FAILED,
        LlmEnrichment.SKIPPED,
    )
    assert kept.extraction_error == failed.extraction_error == ExtractionError.GAVE_UP.value
    assert ExtractionError.GAVE_UP.value == "gave up waiting for the model"


async def test_step_two_leaves_younger_jobs_and_other_states_alone(factory: SessionFactory) -> None:
    cutoff = NOW - timedelta(days=SETTINGS.llm_enrichment_give_up_days)
    exactly_at_the_limit = await add_job(factory, NEEDS_STACK, created_at=cutoff, **WAITING)
    young = await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(days=1), **WAITING)
    old_but_done = await add_job(
        factory,
        RICH,
        created_at=TOO_OLD,
        extraction_status=ExtractionStatus.EXTRACTED,
        title="Done",
        llm_enrichment=LlmEnrichment.DONE,
        extraction_method=ExtractionMethod.RULES_LLM,
    )

    result = await work(factory)

    assert result.gave_up == 0
    for job_id in (exactly_at_the_limit, young):
        assert (await row(factory, job_id)).llm_enrichment is LlmEnrichment.PENDING
    assert (await row(factory, old_but_done)).llm_enrichment is LlmEnrichment.DONE


async def test_the_give_up_age_follows_the_setting(factory: SessionFactory) -> None:
    job_id = await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(days=2), **WAITING)

    assert (await work(factory, settings_with(llm_enrichment_give_up_days=3))).gave_up == 0
    assert (await work(factory, settings_with(llm_enrichment_give_up_days=1))).gave_up == 1
    assert (await row(factory, job_id)).llm_enrichment is LlmEnrichment.SKIPPED


async def test_a_given_up_job_can_be_retried_by_its_owner(factory: SessionFactory) -> None:
    """The sweep's skipped state is exactly what the retry endpoint accepts."""
    job_id = await add_job(factory, NEEDS_STACK, created_at=TOO_OLD, **WAITING)
    await work(factory)
    user_id = await owner_of(factory, job_id)

    async with transaction(factory) as session:
        used = await repository.reset_for_retry(
            session, user_id=user_id, job_id=job_id, max_retries=5
        )

    assert used == 1


# --- the title check can never be violated by the sweep -----------------------------------------

PENDING_S, EXTRACTED_S = ExtractionStatus.PENDING, ExtractionStatus.EXTRACTED
FAILED_S, NOT_A_JOB_S = ExtractionStatus.EXTRACTION_FAILED, ExtractionStatus.NOT_A_JOB
# Every combination the database allows for a job waiting for the model. ("extracted" with no
# title is not in the list: the database refuses to create it, see the next test.)
STATES_THAT_CAN_EXIST = [
    (PENDING_S, None),
    (PENDING_S, "Titled"),
    (EXTRACTED_S, "Titled"),
    (FAILED_S, None),
    (FAILED_S, "Titled"),
    (NOT_A_JOB_S, None),
    (NOT_A_JOB_S, "Titled"),
]


async def untitled_extracted_jobs(factory: SessionFactory) -> int:
    async with factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.extraction_status == EXTRACTED_S, Job.title.is_(None))
        )
    return count or 0


async def test_an_extracted_job_without_a_title_cannot_be_created(factory: SessionFactory) -> None:
    with pytest.raises(IntegrityError, match="ck_jobs_extracted_has_title"):
        await add_job(factory, RICH, extraction_status=EXTRACTED_S, title=None)

    assert await untitled_extracted_jobs(factory) == 0


async def test_nor_can_a_title_be_taken_from_an_extracted_job(factory: SessionFactory) -> None:
    job_id = await add_job(factory, RICH, extraction_status=EXTRACTED_S, title="Titled")

    with pytest.raises(IntegrityError, match="ck_jobs_extracted_has_title"):
        async with transaction(factory) as session:
            await session.execute(update(Job).where(Job.id == job_id).values(title=None))

    assert (await row(factory, job_id)).title == "Titled"


@pytest.mark.parametrize(("status", "title"), STATES_THAT_CAN_EXIST)
async def test_the_give_up_update_cannot_violate_the_title_check_for_any_state(
    factory: SessionFactory, status: ExtractionStatus, title: str | None
) -> None:
    """The CASE reads the same row the UPDATE changes: no title means failed, a title keeps the
    status. So the one forbidden pair (extracted, no title) cannot come out of it."""
    job_id = await add_job(
        factory,
        NEEDS_STACK,
        created_at=TOO_OLD,
        extraction_status=status,
        title=title,
        llm_enrichment=LlmEnrichment.PENDING,
        extraction_method=ExtractionMethod.RULES,
    )

    await work(factory)  # no exception: the database accepted the UPDATE

    job = await row(factory, job_id)
    expected = FAILED_S if title is None else status
    assert (job.extraction_status, job.llm_enrichment) == (expected, LlmEnrichment.SKIPPED)
    assert await untitled_extracted_jobs(factory) == 0


async def test_the_give_up_update_over_every_state_at_once_leaves_the_invariant_true(
    factory: SessionFactory,
) -> None:
    for status, title in STATES_THAT_CAN_EXIST:
        await add_job(
            factory,
            NEEDS_STACK,
            created_at=TOO_OLD,
            extraction_status=status,
            title=title,
            llm_enrichment=LlmEnrichment.PENDING,
        )

    result = await work(factory)

    assert result.gave_up == len(STATES_THAT_CAN_EXIST)
    assert await untitled_extracted_jobs(factory) == 0


@pytest.mark.parametrize(("status", "title"), STATES_THAT_CAN_EXIST)
async def test_writing_skipped_after_a_failed_model_attempt_follows_the_same_rule(
    factory: SessionFactory, status: ExtractionStatus, title: str | None
) -> None:
    job_id = await add_job(
        factory,
        NEEDS_STACK,
        extraction_status=status,
        title=title,
        llm_enrichment=LlmEnrichment.PENDING,
        extraction_method=ExtractionMethod.RULES,
    )
    async with transaction(factory) as session:
        snapshot = await repository.load_snapshot(session, job_id)
    assert snapshot is not None

    async with transaction(factory) as session:
        won = await repository.write_skipped(session, snapshot, ExtractionError.INVALID_REPLY)

    assert won is True
    assert (await row(factory, job_id)).extraction_status is (FAILED_S if title is None else status)
    assert await untitled_extracted_jobs(factory) == 0


async def test_a_retry_reset_cannot_violate_the_check_either(factory: SessionFactory) -> None:
    """The reset writes status 'pending', which the check allows with or without a title."""
    job_id = await add_job(
        factory,
        NEEDS_STACK,
        extraction_status=EXTRACTED_S,
        title="Titled",
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )
    async with transaction(factory) as session:
        await repository.reset_for_retry(
            session, user_id=await owner_of(factory, job_id), job_id=job_id, max_retries=5
        )

    assert (await row(factory, job_id)).extraction_status is PENDING_S
    assert await untitled_extracted_jobs(factory) == 0


# --- step 3: rules not run ---------------------------------------------------------------------


async def test_step_three_lists_jobs_whose_rules_phase_never_ran_oldest_first(
    factory: SessionFactory,
) -> None:
    newest = await add_job(factory, RICH, created_at=NOW - timedelta(hours=1))
    oldest = await add_job(factory, RICH, created_at=NOW - timedelta(hours=3))
    middle = await add_job(factory, RICH, created_at=NOW - timedelta(hours=2))

    result = await work(factory)

    assert result.rules_job_ids == [oldest, middle, newest]


async def test_step_three_skips_jobs_the_rules_already_handled_or_that_could_not_progress(
    factory: SessionFactory,
) -> None:
    due = await add_job(factory, RICH)
    await add_job(factory, RICH, extraction_method=ExtractionMethod.RULES)  # already ran
    await add_job(  # any other state: a task would end as a no-op
        factory,
        RICH,
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        llm_enrichment=LlmEnrichment.SKIPPED,
    )

    assert (await work(factory)).rules_job_ids == [due]


async def test_step_three_takes_at_most_one_batch(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extraction_sweep, "SWEEP_BATCH", 2)
    ids = [await add_job(factory, RICH, created_at=NOW - timedelta(hours=h)) for h in (3, 2, 1)]

    assert (await work(factory)).rules_job_ids == ids[:2]
    assert SWEEP_BATCH == 100


async def test_step_three_finds_the_job_a_retry_reset(factory: SessionFactory) -> None:
    job_id = await add_job(
        factory,
        NOTHING_SHORT,
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )
    user_id = await owner_of(factory, job_id)
    assert (await work(factory)).rules_job_ids == []
    async with transaction(factory) as session:
        await repository.reset_for_retry(session, user_id=user_id, job_id=job_id, max_retries=5)

    assert (await work(factory)).rules_job_ids == [job_id]  # method NULL again: due


# --- step 4: jobs waiting for the model --------------------------------------------------------


async def waiting_jobs(factory: SessionFactory, count: int = 1) -> list[uuid.UUID]:
    return [
        await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(hours=count - i), **WAITING)
        for i in range(count)
    ]


async def test_step_four_lists_waiting_jobs_oldest_first_when_asking_could_succeed(
    factory: SessionFactory,
) -> None:
    ids = await waiting_jobs(factory, 3)

    assert (await work(factory, cooldown=FakeCooldown())).enrichment_job_ids == ids


async def test_step_four_queues_nothing_without_a_client(factory: SessionFactory) -> None:
    await waiting_jobs(factory)

    assert (await work(factory, cooldown=None)).enrichment_job_ids == []


async def test_step_four_queues_nothing_while_a_pause_is_in_effect(factory: SessionFactory) -> None:
    await waiting_jobs(factory)

    result = await work(factory, cooldown=FakeCooldown(seconds_left=30.0))

    assert result.enrichment_job_ids == []


async def test_step_four_with_redis_down_logs_once_and_does_nothing(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    await waiting_jobs(factory, 3)
    rules_due = await add_job(factory, RICH)

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_sweep"):
        result = await work(factory, cooldown=FakeCooldown(unavailable=True))

    assert result.enrichment_job_ids == []
    assert result.rules_job_ids == [rules_due]  # the rules need no Redis to be worth queueing
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize(
    "settings",
    [
        settings_with(llm_daily_request_cap_global=1),
        settings_with(llm_requests_per_minute=1),
    ],
    ids=["global cap", "per-minute throttle"],
)
async def test_step_four_queues_nothing_when_the_global_budget_has_no_room(
    factory: SessionFactory, settings: Settings
) -> None:
    await waiting_jobs(factory)
    stranger = await add_user(factory)
    await add_usage(factory, stranger, NOW - timedelta(seconds=10))

    assert (await work(factory, settings, FakeCooldown())).enrichment_job_ids == []


async def test_step_four_queues_no_more_jobs_than_there_are_places_left(
    factory: SessionFactory,
) -> None:
    ids = await waiting_jobs(factory, 4)
    stranger = await add_user(factory)
    await add_usage(factory, stranger, NOW - timedelta(hours=2))  # 1 of 3 used: 2 places left

    result = await work(factory, settings_with(llm_daily_request_cap_global=3), FakeCooldown())

    assert result.enrichment_job_ids == ids[:2]


async def test_an_owner_who_used_their_daily_cap_does_not_starve_the_others(
    factory: SessionFactory,
) -> None:
    capped_job = await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(hours=5), **WAITING)
    capped_user = await owner_of(factory, capped_job)
    await add_usage(factory, capped_user, NOW - timedelta(hours=1))
    other = await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(hours=1), **WAITING)

    result = await work(
        factory, settings_with(llm_daily_request_cap_per_user=1), cooldown=FakeCooldown()
    )

    assert result.enrichment_job_ids == [other]  # the older job of the capped user is left out


async def test_with_a_per_user_cap_of_zero_nothing_waiting_is_queued(
    factory: SessionFactory,
) -> None:
    await waiting_jobs(factory)

    result = await work(
        factory, settings_with(llm_daily_request_cap_per_user=0), cooldown=FakeCooldown()
    )

    assert result.enrichment_job_ids == []


async def test_the_daily_window_for_the_user_cap_is_24_hours(factory: SessionFactory) -> None:
    (job_id,) = await waiting_jobs(factory)
    user_id = await owner_of(factory, job_id)
    await add_usage(factory, user_id, NOW - DAY)  # exactly 24 h old: no longer counts

    result = await work(
        factory, settings_with(llm_daily_request_cap_per_user=1), cooldown=FakeCooldown()
    )

    assert result.enrichment_job_ids == [job_id]


# --- the order of the steps --------------------------------------------------------------------


async def test_a_job_that_waited_too_long_is_given_up_before_it_could_be_listed(
    factory: SessionFactory,
) -> None:
    old = await add_job(factory, NEEDS_STACK, created_at=TOO_OLD, **WAITING)
    fresh = await add_job(factory, NEEDS_STACK, created_at=NOW - timedelta(hours=1), **WAITING)

    result = await work(factory, cooldown=FakeCooldown())

    assert result.gave_up == 1 and result.enrichment_job_ids == [fresh]
    assert old not in result.enrichment_job_ids


async def test_abandoned_reservations_still_count_when_the_budget_is_read(
    factory: SessionFactory,
) -> None:
    (job_id,) = await waiting_jobs(factory)
    user_id = await owner_of(factory, job_id)
    await add_usage(factory, user_id, NOW - timedelta(minutes=30), outcome="reserved")

    result = await work(
        factory, settings_with(llm_daily_request_cap_per_user=1), cooldown=FakeCooldown()
    )

    assert result.abandoned == 1
    assert result.enrichment_job_ids == []  # relabelled in step 1, and still counted in step 4


# --- a retry meets the same gates --------------------------------------------------------------


async def test_a_retried_job_still_meets_the_length_gate_the_budget_and_the_kill_switch(
    factory: SessionFactory,
) -> None:
    short = await add_job(
        factory,
        NOTHING_SHORT,
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )
    long_ = await add_job(
        factory,
        NEEDS_STACK,
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )
    for job_id in (short, long_):
        async with transaction(factory) as session:
            await repository.reset_for_retry(
                session, user_id=await owner_of(factory, job_id), job_id=job_id, max_retries=5
            )
    llm = FakeLLMClient()

    gate = await extract_job(factory, short, settings=SETTINGS, llm=llm, cooldown=FakeCooldown())
    switch = await extract_job(factory, long_, settings=SETTINGS, llm=None, cooldown=None)
    budget = await extract_job(
        factory,
        long_,
        settings=settings_with(llm_daily_request_cap_per_user=0),
        llm=llm,
        cooldown=FakeCooldown(),
    )

    assert gate.outcome is ExtractionOutcome.FAILED  # still too short for the model
    assert (await row(factory, short)).extraction_error == ExtractionError.TOO_SHORT.value
    assert switch.outcome is ExtractionOutcome.AWAITING_MODEL  # no client: waits, not asked
    assert (budget.outcome, budget.deferral) == (
        ExtractionOutcome.DEFERRED,
        DeferralReason.USER_CAP,
    )
    assert llm.requests == []


# --- the reset itself --------------------------------------------------------------------------


async def test_reset_for_retry_only_serves_the_owner_and_stops_at_the_limit(
    factory: SessionFactory,
) -> None:
    failed = {
        "extraction_status": ExtractionStatus.EXTRACTION_FAILED,
        "llm_enrichment": LlmEnrichment.SKIPPED,
        "extraction_method": ExtractionMethod.RULES,
        "extraction_error": "post too short for the model",
    }
    job_id = await add_job(factory, NOTHING_SHORT, extraction_retries=4, **failed)
    owner, stranger = await owner_of(factory, job_id), await add_user(factory)

    async with transaction(factory) as session:
        nobody = await repository.reset_for_retry(
            session, user_id=stranger, job_id=job_id, max_retries=5
        )
        fifth = await repository.reset_for_retry(
            session, user_id=owner, job_id=job_id, max_retries=5
        )
    job = await row(factory, job_id)
    assert (nobody, fifth) == (None, 5)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    )
    assert (job.extraction_error, job.extraction_method) == (None, None)

    async with transaction(factory) as session:  # fail it again; the limit now says no
        await session.merge(job)
        job.extraction_status = ExtractionStatus.EXTRACTION_FAILED
    async with transaction(factory) as session:
        sixth = await repository.reset_for_retry(
            session, user_id=owner, job_id=job_id, max_retries=5
        )
    assert sixth is None and (await row(factory, job_id)).extraction_retries == 5


async def test_two_resets_at_once_use_one_retry(factory: SessionFactory) -> None:
    job_id = await add_job(
        factory,
        NOTHING_SHORT,
        extraction_status=ExtractionStatus.EXTRACTION_FAILED,
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )
    owner = await owner_of(factory, job_id)

    async def attempt() -> int | None:
        async with transaction(factory) as session:
            return await repository.reset_for_retry(
                session, user_id=owner, job_id=job_id, max_retries=5
            )

    results = await asyncio.gather(attempt(), attempt())

    assert sorted(r for r in results if r is not None) == [1]
    assert (await row(factory, job_id)).extraction_retries == 1


# --- the Celery wrapper, synchronous -----------------------------------------------------------


class RecordingEnqueue:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[uuid.UUID] = []
        self.fail_with = fail_with

    def __call__(self, job_id: uuid.UUID, delay: float) -> None:
        self.calls.append(job_id)
        if self.fail_with:
            raise self.fail_with


def task_user(seed: Seeder) -> tuple[uuid.UUID, uuid.UUID]:
    user_id = seed.user()
    return user_id, seed.source(user_id)


def test_the_sweep_task_runs_every_step_in_order_and_queues_rules_work_before_model_work(
    seed: Seeder, settings: Settings
) -> None:
    user_id, source_id = task_user(seed)
    abandoned = seed.add_usage(user_id, datetime_ago(minutes=30), outcome="reserved")
    gave_up = seed.job(source_id, NEEDS_STACK, created_at=datetime_ago(days=30), **WAITING)
    rules_due = seed.job(source_id, RICH)
    waiting = seed.job(source_id, NEEDS_STACK, created_at=datetime_ago(hours=1), **WAITING)
    enqueue = RecordingEnqueue()

    summary = run_sweep_extraction(
        settings=settings, llm=FakeLLMClient(), cooldown=FakeCooldown(), enqueue=enqueue
    )

    assert summary == {
        "abandoned": 1,
        "gave_up": 1,
        "rules_due": 1,
        "enrichment_due": 1,
        "queued": 2,
    }
    assert enqueue.calls == [rules_due, waiting]  # step 3 first, then step 4
    assert seed.job_row(gave_up).llm_enrichment is LlmEnrichment.SKIPPED
    assert {e.id: e.outcome for e in seed.usage(user_id)}[abandoned] == "abandoned"


def test_with_the_provider_off_only_rules_work_is_queued(seed: Seeder, settings: Settings) -> None:
    _, source_id = task_user(seed)
    rules_due = seed.job(source_id, RICH)
    seed.job(source_id, NEEDS_STACK, created_at=datetime_ago(hours=1), **WAITING)
    enqueue = RecordingEnqueue()

    summary = run_sweep_extraction(settings=settings, enqueue=enqueue)  # default: provider none

    assert enqueue.calls == [rules_due]
    assert summary["enrichment_due"] == 0


def test_the_sweep_with_the_queue_down_warns_once_and_never_loops(
    seed: Seeder, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    _, source_id = task_user(seed)
    for _ in range(4):
        seed.job(source_id, RICH)
    enqueue = RecordingEnqueue(fail_with=OSError("redis is down"))

    with caplog.at_level(logging.WARNING, logger="app.tasks.jobs"):
        summary = run_sweep_extraction(settings=settings, enqueue=enqueue)

    assert len(enqueue.calls) == 1 and summary["queued"] == 0
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_the_sweep_with_the_cooldown_store_down_still_queues_rules_work(
    seed: Seeder, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    _, source_id = task_user(seed)
    rules_due = seed.job(source_id, RICH)
    seed.job(source_id, NEEDS_STACK, created_at=datetime_ago(hours=1), **WAITING)
    enqueue = RecordingEnqueue()

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_sweep"):
        summary = run_sweep_extraction(
            settings=settings,
            llm=FakeLLMClient(),
            cooldown=FakeCooldown(unavailable=True),
            enqueue=enqueue,
        )

    assert enqueue.calls == [rules_due] and summary["enrichment_due"] == 0
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_the_sweep_task_wrapper_and_limits(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.tasks import jobs

    monkeypatch.setattr("app.tasks.jobs.run_sweep_extraction", lambda: {"queued": 0})

    assert jobs.sweep_extraction.run() == {"queued": 0}
    assert (jobs.sweep_extraction.soft_time_limit, jobs.sweep_extraction.time_limit) == (120, 150)
    assert not getattr(jobs.sweep_extraction, "autoretry_for", ())  # the next sweep is the retry


def datetime_ago(**delta: Any) -> datetime:
    """A moment in the past by the real clock (the task code reads the real clock)."""
    return datetime.now(UTC) - timedelta(**delta)
