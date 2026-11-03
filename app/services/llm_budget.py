"""Counting provider requests: caps, a per-minute throttle, and a reservation made before a call.

A request is counted by a row in `llm_usage` that is inserted *before* the call (outcome
`reserved`) and filled in afterwards. Inserting first means a worker that dies mid-request still
leaves a row, so the request keeps counting (the provider may have counted it too).

The three limits are checked and the row inserted in one short transaction that holds a Postgres
advisory lock, so two workers reserving at once are counted one after the other: at one request
below a cap, exactly one of them gets the last place. The lock is released when that transaction
commits (milliseconds), long before the provider call starts, so no transaction is ever open
while a request is in flight.

Python note: `pg_advisory_xact_lock(key)` is a lock on a number we choose, not on a table or row.
It is held until the end of the current transaction and queues other callers with the same key.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.llm.base import LLMUsage
from app.models import LlmUsageEntry

logger = logging.getLogger(__name__)

# An arbitrary constant that names "reserve an llm_usage row". Any fixed number works.
RESERVATION_LOCK_KEY = 7_203_001

DAY = timedelta(hours=24)
MINUTE = timedelta(seconds=60)
# A reservation older than this belongs to a worker that is gone: no run lasts this long
# (the hard time limit is 90 s).
STALE_AFTER = timedelta(minutes=10)
# More abandoned reservations than this inside one 24 h window look like a crash loop.
ABANDONED_WARNING_THRESHOLD = 3


class Outcome(StrEnum):
    """The `llm_usage.outcome` vocabulary. Every value counts toward the caps."""

    RESERVED = "reserved"  # about to be sent, or the worker died before filling it in
    ABANDONED = "abandoned"  # a reservation nobody finished (set by the sweep)
    OK = "ok"
    RATE_LIMITED = "rate_limited"  # HTTP 429
    UNAVAILABLE = "unavailable"  # 5xx or the connection failed
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"  # empty, cut off or failed validation
    REJECTED = "rejected"  # the provider refused the request itself


class DeferralReason(StrEnum):
    """Why a job was not sent to the model this time. It stays as it is and is tried later."""

    COOLDOWN = "cooldown"  # a 429 or an outage set a shared pause
    COOLDOWN_UNAVAILABLE = "cooldown_unavailable"  # Redis is down, so we cannot tell: wait
    USER_CAP = "user_cap"
    GLOBAL_CAP = "global_cap"
    THROTTLE = "throttle"  # too many requests in the last minute
    RATE_LIMITED = "rate_limited"  # the provider answered 429 to this very request
    UNAVAILABLE = "unavailable"  # the request failed with a 5xx or a connection error
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"  # bad key, no access or unknown model: nothing will work yet


@dataclass(frozen=True)
class Reservation:
    id: uuid.UUID


@dataclass(frozen=True)
class Refusal:
    reason: DeferralReason
    retry_after_seconds: float  # when the limit that said no frees a place (at least 1 s)


async def reserve(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    provider: str,
    model: str,
    prompt_version: str,
    settings: Settings,
    now: datetime | None = None,
) -> Reservation | Refusal:
    """Take one request from the budget, or say which limit is full.

    Runs inside the caller's transaction, which must commit promptly: the advisory lock is held
    until then. `now` exists so tests can place rows in time; production leaves it out.
    """
    await session.execute(select(func.pg_advisory_xact_lock(RESERVATION_LOCK_KEY)))
    now = now or datetime.now(UTC)  # read after the lock is ours, not before waiting for it
    limits = (
        (DeferralReason.USER_CAP, DAY, settings.llm_daily_request_cap_per_user, user_id),
        (DeferralReason.GLOBAL_CAP, DAY, settings.llm_daily_request_cap_global, None),
        (DeferralReason.THROTTLE, MINUTE, settings.llm_requests_per_minute, None),
    )
    for reason, window, limit, only_user in limits:
        wait = await _wait_if_full(session, now, window, limit, only_user)
        if wait is not None:
            return Refusal(reason, wait)
    entry = LlmUsageEntry(
        user_id=user_id,
        job_id=job_id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        outcome=Outcome.RESERVED.value,
        created_at=now,
    )
    session.add(entry)
    await session.flush()
    return Reservation(entry.id)


async def places_left(
    session: AsyncSession, settings: Settings, now: datetime | None = None
) -> int:
    """How many more requests the global limits (24 h cap and per-minute throttle) allow right
    now, never below 0. A read only: it reserves nothing, so it is a hint for the sweep, not a
    promise; `reserve` is what actually decides."""
    now = now or datetime.now(UTC)
    places: list[int] = []
    for window, limit in (
        (DAY, settings.llm_daily_request_cap_global),
        (MINUTE, settings.llm_requests_per_minute),
    ):
        used = await session.scalar(
            select(func.count())
            .select_from(LlmUsageEntry)
            .where(LlmUsageEntry.created_at > now - window)
        )
        places.append(max(limit - (used or 0), 0))
    return min(places)


async def finalize(
    session: AsyncSession,
    reservation: Reservation,
    outcome: Outcome,
    usage: LLMUsage | None = None,
) -> bool:
    """Fill in how a reserved request ended. False if the row is no longer `reserved` (the sweep
    already relabelled it), in which case nothing is changed."""
    usage = usage or LLMUsage()
    statement = (
        update(LlmUsageEntry)
        .where(LlmUsageEntry.id == reservation.id, LlmUsageEntry.outcome == Outcome.RESERVED.value)
        .values(
            outcome=outcome.value,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thought_tokens=usage.thought_tokens,
        )
        .returning(LlmUsageEntry.id)
    )
    return (await session.execute(statement)).first() is not None


async def discard(session: AsyncSession, reservation: Reservation) -> None:
    """Remove a reservation whose request never left this process (our own guard refused it)."""
    await session.execute(
        delete(LlmUsageEntry).where(
            LlmUsageEntry.id == reservation.id, LlmUsageEntry.outcome == Outcome.RESERVED.value
        )
    )


async def abandon_stale_reservations(session: AsyncSession, now: datetime | None = None) -> int:
    """Relabel reservations nobody finished as `abandoned` (they keep counting) and return how
    many. Logs a warning when more than a few abandoned rows fall in one 24 h window, because a
    worker that keeps dying mid-request is a crash loop that should be seen."""
    now = now or datetime.now(UTC)
    relabelled = (
        await session.execute(
            update(LlmUsageEntry)
            .where(
                LlmUsageEntry.outcome == Outcome.RESERVED.value,
                LlmUsageEntry.created_at < now - STALE_AFTER,
            )
            .values(outcome=Outcome.ABANDONED.value)
            .returning(LlmUsageEntry.id)
        )
    ).all()
    abandoned = await session.scalar(
        select(func.count())
        .select_from(LlmUsageEntry)
        .where(
            LlmUsageEntry.outcome == Outcome.ABANDONED.value,
            LlmUsageEntry.created_at > now - DAY,
        )
    )
    if (abandoned or 0) > ABANDONED_WARNING_THRESHOLD:
        logger.warning(
            "%d abandoned llm requests in the last 24 hours: a worker may be dying mid-request",
            abandoned,
        )
    return len(relabelled)


async def _wait_if_full(
    session: AsyncSession,
    now: datetime,
    window: timedelta,
    limit: int,
    only_user: uuid.UUID | None,
) -> float | None:
    """None if the window has room; otherwise the seconds until enough rows age out of it."""
    conditions: list[ColumnElement[Any]] = [LlmUsageEntry.created_at > now - window]
    if only_user is not None:
        conditions.append(LlmUsageEntry.user_id == only_user)
    used = await session.scalar(select(func.count()).select_from(LlmUsageEntry).where(*conditions))
    used = used or 0
    if used < limit:
        return None
    # The row that must age out for a place to open is the (used - limit + 1)-th oldest.
    blocking = await session.scalar(
        select(LlmUsageEntry.created_at)
        .where(*conditions)
        .order_by(LlmUsageEntry.created_at)
        .offset(used - limit)
        .limit(1)
    )
    if blocking is None:  # a limit of 0 with nothing in the window: nothing will ever free up
        return window.total_seconds()
    return max((blocking + window - now).total_seconds(), 1.0)
