"""Request caps, the per-minute throttle and the reservation row, against a real Postgres."""

import asyncio
import logging
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.llm.base import LLMUsage
from app.models import LlmUsageEntry
from app.services import llm_budget
from app.services.extraction_types import SessionFactory, transaction
from app.services.llm_budget import (
    ABANDONED_WARNING_THRESHOLD,
    STALE_AFTER,
    DeferralReason,
    Outcome,
    Refusal,
    Reservation,
)
from tests.extraction_helpers import NOW, RICH, add_job, add_usage, add_user, owner_of, usage_rows
from tests.settings_helpers import settings_with

DAY = timedelta(hours=24)
MINUTE = timedelta(minutes=1)
LOTS = 10_000  # a limit nothing in a test reaches


async def reserve(
    factory: SessionFactory,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    per_user: int = LOTS,
    overall: int = LOTS,
    per_minute: int = LOTS,
) -> Reservation | Refusal:
    settings = settings_with(
        llm_daily_request_cap_per_user=per_user,
        llm_daily_request_cap_global=overall,
        llm_requests_per_minute=per_minute,
    )
    async with transaction(factory) as session:
        return await llm_budget.reserve(
            session,
            user_id=user_id,
            job_id=job_id,
            provider="fake",
            model="fake-model",
            prompt_version="extract_job_v1",
            settings=settings,
            now=NOW,
        )


@pytest.fixture
async def user_and_job(factory: SessionFactory) -> tuple[uuid.UUID, uuid.UUID]:
    job_id = await add_job(factory, RICH)
    return await owner_of(factory, job_id), job_id


# --- a successful reservation --------------------------------------------------------------------


async def test_a_reservation_inserts_a_reserved_row_before_any_call(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job

    result = await reserve(factory, user_id, job_id)

    assert isinstance(result, Reservation)
    [entry] = await usage_rows(factory)
    assert entry.id == result.id
    assert (entry.user_id, entry.job_id) == (user_id, job_id)
    assert (entry.provider, entry.model, entry.prompt_version) == (
        "fake",
        "fake-model",
        "extract_job_v1",
    )
    assert entry.outcome == Outcome.RESERVED.value and entry.created_at == NOW
    assert (entry.input_tokens, entry.output_tokens, entry.thought_tokens) == (None, None, None)


# --- every limit and its boundary ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "counts"),
    [
        (DAY - timedelta(seconds=1), True),  # inside the window
        (DAY, False),  # exactly 24 h old: just outside
        (DAY + timedelta(hours=1), False),
    ],
)
async def test_the_per_user_cap_is_a_rolling_24_hours(
    factory: SessionFactory,
    user_and_job: tuple[uuid.UUID, uuid.UUID],
    age: timedelta,
    counts: bool,
) -> None:
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - age)

    result = await reserve(factory, user_id, job_id, per_user=1)

    assert isinstance(result, Refusal) is counts
    if isinstance(result, Refusal):
        assert result.reason is DeferralReason.USER_CAP


async def test_one_below_the_per_user_cap_still_reserves_and_the_cap_then_refuses(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    for _ in range(2):
        await add_usage(factory, user_id, NOW - timedelta(hours=1))

    assert isinstance(await reserve(factory, user_id, job_id, per_user=3), Reservation)
    refusal = await reserve(factory, user_id, job_id, per_user=3)

    assert isinstance(refusal, Refusal) and refusal.reason is DeferralReason.USER_CAP


async def test_other_users_requests_count_for_the_global_cap_but_not_the_user_cap(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    stranger = await add_user(factory)
    await add_usage(factory, stranger, NOW - timedelta(hours=1))

    assert isinstance(await reserve(factory, user_id, job_id, per_user=1), Reservation)
    refusal = await reserve(factory, user_id, job_id, overall=2)

    assert isinstance(refusal, Refusal) and refusal.reason is DeferralReason.GLOBAL_CAP


async def test_the_global_cap_is_also_a_rolling_24_hours(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    stranger = await add_user(factory)
    await add_usage(factory, stranger, NOW - DAY)  # exactly 24 h: outside

    assert isinstance(await reserve(factory, user_id, job_id, overall=1), Reservation)


@pytest.mark.parametrize(
    ("age", "counts"),
    [
        (MINUTE - timedelta(seconds=1), True),
        (MINUTE, False),  # exactly 60 s old: just outside
        (MINUTE + timedelta(seconds=1), False),
    ],
)
async def test_the_per_minute_throttle_looks_at_the_last_60_seconds(
    factory: SessionFactory,
    user_and_job: tuple[uuid.UUID, uuid.UUID],
    age: timedelta,
    counts: bool,
) -> None:
    user_id, job_id = user_and_job
    stranger = await add_user(factory)  # the throttle protects the provider, so it is global
    await add_usage(factory, stranger, NOW - age)

    result = await reserve(factory, user_id, job_id, per_minute=1)

    assert isinstance(result, Refusal) is counts
    if isinstance(result, Refusal):
        assert result.reason is DeferralReason.THROTTLE


async def test_a_cap_of_zero_refuses_everything(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job

    refusal = await reserve(factory, user_id, job_id, per_user=0)

    assert isinstance(refusal, Refusal)
    assert refusal.retry_after_seconds == DAY.total_seconds()  # nothing will ever free up


async def test_when_several_limits_are_full_the_users_own_cap_is_reported_first(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - timedelta(seconds=5))

    refusal = await reserve(factory, user_id, job_id, per_user=1, overall=1, per_minute=1)

    assert isinstance(refusal, Refusal) and refusal.reason is DeferralReason.USER_CAP


@pytest.mark.parametrize("outcome", [member.value for member in Outcome])
async def test_every_outcome_counts_toward_the_caps(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID], outcome: str
) -> None:
    """A failed, reserved or abandoned request may still have been counted by the provider."""
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - timedelta(minutes=5), outcome=outcome)

    refusal = await reserve(factory, user_id, job_id, per_user=1)

    assert isinstance(refusal, Refusal)


async def test_the_retry_hint_is_when_enough_old_requests_age_out(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    for hours in (3, 2, 1):  # the oldest leaves the window 21 hours from now
        await add_usage(factory, user_id, NOW - timedelta(hours=hours))

    refusal = await reserve(factory, user_id, job_id, per_user=3)

    assert isinstance(refusal, Refusal)
    assert refusal.retry_after_seconds == timedelta(hours=21).total_seconds()


async def test_over_the_limit_the_hint_waits_for_enough_rows_not_just_one(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    for hours in (4, 3, 2, 1):  # 4 used, limit 2: two rows must leave, so wait for the second
        await add_usage(factory, user_id, NOW - timedelta(hours=hours))

    refusal = await reserve(factory, user_id, job_id, per_user=2)

    assert isinstance(refusal, Refusal)
    assert refusal.retry_after_seconds == timedelta(hours=22).total_seconds()


async def test_the_retry_hint_is_never_below_one_second(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - DAY + timedelta(milliseconds=1))

    refusal = await reserve(factory, user_id, job_id, per_user=1)

    assert isinstance(refusal, Refusal) and refusal.retry_after_seconds == 1.0


# --- two workers at one place below the cap ------------------------------------------------------


async def test_at_one_below_the_cap_exactly_one_of_two_concurrent_reservations_wins(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - timedelta(hours=1))  # 1 used, cap 2: one place left

    results = await asyncio.gather(
        reserve(factory, user_id, job_id, per_user=2), reserve(factory, user_id, job_id, per_user=2)
    )

    assert sorted(type(result).__name__ for result in results) == ["Refusal", "Reservation"]
    assert len(await usage_rows(factory)) == 2  # the old row and exactly one new one


@pytest.mark.parametrize("limit", ["per_user", "overall", "per_minute"])
async def test_the_race_is_decided_the_same_way_for_every_limit(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID], limit: str
) -> None:
    user_id, job_id = user_and_job
    await add_usage(factory, user_id, NOW - timedelta(seconds=10))

    results = await asyncio.gather(
        *(reserve(factory, user_id, job_id, **{limit: 2}) for _ in range(5))
    )  # type: ignore[arg-type]

    assert sum(isinstance(result, Reservation) for result in results) == 1


# --- finishing a reservation ---------------------------------------------------------------------


async def test_finalize_records_the_outcome_and_the_token_counts(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    reservation = await reserve(factory, user_id, job_id)
    assert isinstance(reservation, Reservation)

    async with transaction(factory) as session:
        done = await llm_budget.finalize(session, reservation, Outcome.OK, LLMUsage(10, 5, 3))

    assert done is True
    [entry] = await usage_rows(factory)
    assert entry.outcome == "ok"
    assert (entry.input_tokens, entry.output_tokens, entry.thought_tokens) == (10, 5, 3)


async def test_a_finished_or_abandoned_row_cannot_be_finalized_again(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, _ = user_and_job
    abandoned = Reservation(await add_usage(factory, user_id, NOW, outcome="abandoned"))
    async with transaction(factory) as session:
        assert await llm_budget.finalize(session, abandoned, Outcome.OK) is False

    [entry] = await usage_rows(factory)
    assert entry.outcome == "abandoned"


async def test_discard_removes_a_reserved_row_and_only_that(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    reservation = await reserve(factory, user_id, job_id)
    assert isinstance(reservation, Reservation)
    kept = Reservation(await add_usage(factory, user_id, NOW, outcome="ok"))

    async with transaction(factory) as session:
        await llm_budget.discard(session, reservation)
        await llm_budget.discard(session, kept)  # not reserved: left alone

    assert [entry.outcome for entry in await usage_rows(factory)] == ["ok"]


# --- reservations nobody finished ----------------------------------------------------------------


async def test_a_stale_reservation_becomes_abandoned_and_keeps_counting(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, job_id = user_and_job
    stale = await add_usage(
        factory, user_id, NOW - STALE_AFTER - timedelta(seconds=1), outcome="reserved"
    )
    fresh = await add_usage(factory, user_id, NOW - timedelta(minutes=1), outcome="reserved")

    async with transaction(factory) as session:
        relabelled = await llm_budget.abandon_stale_reservations(session, NOW)

    assert relabelled == 1
    outcomes = {entry.id: entry.outcome for entry in await usage_rows(factory)}
    assert outcomes == {stale: "abandoned", fresh: "reserved"}
    # Still inside the 24 h window, so it still counts against the cap.
    assert isinstance(await reserve(factory, user_id, job_id, per_user=2), Refusal)


async def test_a_reservation_exactly_at_the_stale_age_is_not_abandoned_yet(
    factory: SessionFactory, user_and_job: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, _ = user_and_job
    await add_usage(factory, user_id, NOW - STALE_AFTER, outcome="reserved")

    async with transaction(factory) as session:
        assert await llm_budget.abandon_stale_reservations(session, NOW) == 0


async def test_more_than_three_abandoned_rows_in_one_window_logs_a_warning(
    factory: SessionFactory,
    user_and_job: tuple[uuid.UUID, uuid.UUID],
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_id, _ = user_and_job
    for minutes in range(ABANDONED_WARNING_THRESHOLD + 1):  # 4 rows
        await add_usage(
            factory, user_id, NOW - timedelta(hours=1, minutes=minutes), outcome="reserved"
        )

    with caplog.at_level(logging.WARNING, logger="app.services.llm_budget"):
        async with transaction(factory) as session:
            await llm_budget.abandon_stale_reservations(session, NOW)

    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "4 abandoned" in record.getMessage()


async def test_exactly_three_abandoned_rows_do_not_warn(
    factory: SessionFactory,
    user_and_job: tuple[uuid.UUID, uuid.UUID],
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_id, _ = user_and_job
    for minutes in range(ABANDONED_WARNING_THRESHOLD):  # 3 rows
        await add_usage(
            factory, user_id, NOW - timedelta(hours=1, minutes=minutes), outcome="reserved"
        )

    with caplog.at_level(logging.WARNING, logger="app.services.llm_budget"):
        async with transaction(factory) as session:
            await llm_budget.abandon_stale_reservations(session, NOW)

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_abandoned_rows_older_than_the_window_do_not_count_toward_the_warning(
    factory: SessionFactory,
    user_and_job: tuple[uuid.UUID, uuid.UUID],
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_id, _ = user_and_job
    for days in range(1, 6):  # five old abandoned rows, all outside the last 24 hours
        await add_usage(factory, user_id, NOW - timedelta(days=days), outcome="abandoned")
    await add_usage(factory, user_id, NOW - timedelta(hours=2), outcome="reserved")

    with caplog.at_level(logging.WARNING, logger="app.services.llm_budget"):
        async with transaction(factory) as session:
            assert await llm_budget.abandon_stale_reservations(session, NOW) == 1

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    async with factory() as session:
        count = (await session.scalars(select(LlmUsageEntry.id))).all()
    assert len(count) >= 6
