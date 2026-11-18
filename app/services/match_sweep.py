"""The safety net for scoring: find users whose matches are out of date and have no task coming.

One pass per scheduled run (no loop, no retry inside). It only reads and updates `match_status`
and the users' CV ids; the task module publishes what it returns. Three steps, each capped:

1. Users with a CV but no status row (a CV from before scoring existed) get a request.
2. Users whose last run used another scorer version or other settings (`fingerprint`) get a
   request, a few at a time and spread over a window, so a settings change does not queue every
   user at once.
3. Users who have been waiting for a run for a while are queued, at most once per
   `REQUEUE_INTERVAL` each, and never after `MAX_FAILURES` failed runs in a row.
4. Users who are up to date but own a job embedded after their last run finished are
   requested and queued (same caps). This recovers a request that was never recorded, for
   example when the database was down for every retry of an `embed_job` task after its
   vector was committed.
"""

import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.matching.inputs_hash import scoring_fingerprint
from app.models import Cv, Job, MatchStatus, Source
from app.services.match_service import scoring_config
from app.services.match_status_service import MAX_FAILURES, RUNNING_STALE

SessionFactory = async_sessionmaker[AsyncSession]

DUE_BATCH = 100  # users queued by one pass because they are waiting
REFRESH_BATCH = 50  # users requested by one pass because the fingerprint changed
RECOVER_BATCH = 100  # users requested by one pass because a job was embedded after their run
MIN_PENDING_AGE = timedelta(minutes=2)  # a fresh request is not "lost" yet: its task is coming
REQUEUE_INTERVAL = timedelta(minutes=5)  # the sweep queues the same user at most this often
SPREAD_WINDOW_SECONDS = 60.0  # fingerprint-triggered runs are spread over this window


@dataclass(frozen=True)
class SweepWork:
    """Whom to queue: the waiting ones now, the refreshed ones spread over a window."""

    waiting: list[uuid.UUID]
    refreshed: list[uuid.UUID]
    recovered: list[uuid.UUID]  # a job was embedded after their last run and nobody asked


def spread_delays(count: int, window: float = SPREAD_WINDOW_SECONDS) -> list[float]:
    """Delays of 0, window/count, 2*window/count, ... so `count` tasks fill the window evenly."""
    return [index * window / count for index in range(count)] if count else []


async def collect_work(factory: SessionFactory, settings: Settings) -> SweepWork:
    """Run the database steps in one transaction and say whom to queue."""
    fingerprint = scoring_fingerprint(scoring_config(settings))
    async with factory() as session, session.begin():
        new_users = await _request_users_without_status(session)
        changed = await _request_users_with_other_fingerprint(session, fingerprint)
        waiting = await _claim_waiting_users(session)
        recovered = await _request_users_with_newer_jobs(session)
    return SweepWork(waiting=waiting, refreshed=[*new_users, *changed], recovered=recovered)


async def _request_users_without_status(session: AsyncSession) -> list[uuid.UUID]:
    without_status = (
        select(Cv.user_id, func.clock_timestamp())
        .where(~exists().where(MatchStatus.user_id == Cv.user_id))
        .order_by(Cv.user_id)
        .limit(REFRESH_BATCH)
    )
    statement = (
        insert(MatchStatus)
        .from_select(["user_id", "queued_at"], without_status)
        .on_conflict_do_nothing()
        .returning(MatchStatus.user_id)
    )
    return list((await session.scalars(statement)).all())


async def _request_users_with_other_fingerprint(
    session: AsyncSession, fingerprint: str
) -> list[uuid.UUID]:
    """Request a re-score for users who are up to date but under another fingerprint.

    A user who is already waiting is left alone (their run will record the new fingerprint), so the
    same users are not picked again on the next pass.
    """
    chosen = (
        select(MatchStatus.user_id)
        .join(Cv, Cv.user_id == MatchStatus.user_id)
        .where(
            MatchStatus.fingerprint.is_distinct_from(fingerprint),
            MatchStatus.requested_seq <= MatchStatus.finished_seq,
            MatchStatus.failure_count < MAX_FAILURES,
        )
        .order_by(MatchStatus.user_id)
        .limit(REFRESH_BATCH)
    )
    statement = (
        update(MatchStatus)
        .where(MatchStatus.user_id.in_(chosen))
        .values(
            requested_seq=MatchStatus.requested_seq + 1,
            requested_at=func.clock_timestamp(),
            queued_at=func.clock_timestamp(),
        )
        .returning(MatchStatus.user_id)
    )
    return list((await session.scalars(statement)).all())


async def _request_users_with_newer_jobs(session: AsyncSession) -> list[uuid.UUID]:
    """Request a re-score for users who are up to date but own a job embedded after their
    last run finished. Same caps as the waiting users: not while a run is in progress, not
    after MAX_FAILURES failures, at most once per REQUEUE_INTERVAL per user, at most
    RECOVER_BATCH users per pass.

    A job embedded during a run raised its own request, so this only finds requests that
    were lost. Cost: one correlated look at the user's jobs per up-to-date user, once per
    pass (no index on `jobs.embedded_at` yet; fine at this size).
    """
    now = func.clock_timestamp()
    newer_job = (
        select(Job.id)
        .join(Source, Source.id == Job.source_id)
        .where(Source.user_id == MatchStatus.user_id, Job.embedded_at > MatchStatus.finished_at)
        .exists()
    )
    chosen = (
        select(MatchStatus.user_id)
        .join(Cv, Cv.user_id == MatchStatus.user_id)
        .where(
            MatchStatus.finished_at.is_not(None),
            MatchStatus.requested_seq <= MatchStatus.finished_seq,
            MatchStatus.failure_count < MAX_FAILURES,
            or_(MatchStatus.queued_at.is_(None), MatchStatus.queued_at < now - REQUEUE_INTERVAL),
            or_(
                MatchStatus.running_since.is_(None), MatchStatus.running_since < now - RUNNING_STALE
            ),
            newer_job,
        )
        .order_by(MatchStatus.finished_at)
        .limit(RECOVER_BATCH)
    )
    statement = (
        update(MatchStatus)
        .where(MatchStatus.user_id.in_(chosen))
        .values(requested_seq=MatchStatus.requested_seq + 1, requested_at=now, queued_at=now)
        .returning(MatchStatus.user_id)
    )
    return list((await session.scalars(statement)).all())


async def _claim_waiting_users(session: AsyncSession) -> list[uuid.UUID]:
    """Users with an old open request and nothing running, marked as queued now (the cap)."""
    now = func.clock_timestamp()
    chosen = (
        select(MatchStatus.user_id)
        .where(
            MatchStatus.requested_seq > MatchStatus.finished_seq,
            MatchStatus.failure_count < MAX_FAILURES,
            MatchStatus.requested_at < now - MIN_PENDING_AGE,
            or_(MatchStatus.queued_at.is_(None), MatchStatus.queued_at < now - REQUEUE_INTERVAL),
            or_(
                MatchStatus.running_since.is_(None), MatchStatus.running_since < now - RUNNING_STALE
            ),
        )
        .order_by(MatchStatus.requested_at)
        .limit(DUE_BATCH)
        .with_for_update(skip_locked=True)
    )
    statement = (
        update(MatchStatus)
        .where(MatchStatus.user_id.in_(chosen))
        .values(queued_at=now)
        .returning(MatchStatus.user_id)
    )
    return list((await session.scalars(statement)).all())
