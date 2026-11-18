"""The `match_status` table: asking for a re-score, running one, and the state the API reports.

The key property is the timeline: a request made while a run is in progress must still be open
when that run ends. Counters (not clocks) decide that, so these tests need no sleeping.
"""

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JobMatch, MatchStatus
from app.services import match_status_service as status
from app.services.extraction_types import SessionFactory
from app.services.match_status_service import ScoringState
from tests.integration.test_match_service import (
    add_cv,
    add_job,
    add_user,
    two_jobs,
)


async def row(session: AsyncSession, user_id: uuid.UUID) -> MatchStatus:
    return (
        await session.execute(
            select(MatchStatus)
            .where(MatchStatus.user_id == user_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def user_with_cv(session: AsyncSession) -> uuid.UUID:
    user = await add_user(session)
    await add_cv(session, user, skills=["Python"])
    return user


# --- asking ------------------------------------------------------------------------------------


async def test_the_first_request_creates_a_pending_row(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)

    await status.request_scoring(db_session, user)

    stored = await row(db_session, user)
    assert (stored.requested_seq, stored.finished_seq, stored.failure_count) == (1, 0, 0)
    assert (await status.read_view(db_session, user)).state is ScoringState.PENDING


async def test_every_further_request_raises_the_request_number(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)

    for _ in range(3):
        await status.request_scoring(db_session, user)

    assert (await row(db_session, user)).requested_seq == 3


async def test_a_user_with_a_cv_and_no_row_is_reported_as_pending(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)

    view = await status.read_view(db_session, user)

    assert view.state is ScoringState.PENDING
    assert (view.requested_at, view.finished_at, view.failure_count) == (None, None, 0)


async def test_a_user_without_a_cv_is_reported_as_no_cv(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await status.request_scoring(db_session, user)

    assert (await status.read_view(db_session, user)).state is ScoringState.NO_CV


async def test_a_job_request_goes_to_the_jobs_owner_only(db_session: AsyncSession) -> None:
    alice, bob = await user_with_cv(db_session), await user_with_cv(db_session)
    near, _ = await two_jobs(db_session, alice)

    owner = await status.request_scoring_for_job(db_session, near.id)

    assert owner == alice
    assert (await row(db_session, alice)).requested_seq == 1
    assert (await db_session.get(MatchStatus, bob)) is None
    assert await status.request_scoring_for_job(db_session, uuid.uuid4()) is None


# --- the timeline ------------------------------------------------------------------------------


async def test_a_change_during_a_run_leaves_the_user_pending_when_the_run_ends(
    db_session: AsyncSession,
) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)  # request 1
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None and ticket.seq == 1

    await status.request_scoring(db_session, user)  # request 2 arrives while the run is going
    await status.finish_run(db_session, ticket, fingerprint="f", candidates=3, written=3)

    stored = await row(db_session, user)
    assert (stored.requested_seq, stored.finished_seq) == (2, 1)
    assert stored.running_since is None  # the run is over...
    assert (await status.read_view(db_session, user)).state is ScoringState.PENDING  # ...not done
    assert await status.is_pending(db_session, user)


async def test_a_run_that_started_after_the_last_change_ends_up_to_date(
    db_session: AsyncSession,
) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)
    await status.request_scoring(db_session, user)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None

    await status.finish_run(db_session, ticket, fingerprint="f", candidates=2, written=2)

    view = await status.read_view(db_session, user)
    assert view.state is ScoringState.UP_TO_DATE
    assert view.finished_at is not None
    assert not await status.is_pending(db_session, user)


async def test_a_second_run_is_refused_while_one_is_in_progress(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)

    first = await status.begin_run(db_session, user)
    second = await status.begin_run(db_session, user)

    assert first is not None and second is None
    assert (await status.read_view(db_session, user)).state is ScoringState.RUNNING


async def test_a_run_that_died_does_not_block_for_ever(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)
    assert await status.begin_run(db_session, user) is not None
    long_ago = datetime.now(UTC) - status.RUNNING_STALE - timedelta(seconds=1)
    await db_session.execute(
        update(MatchStatus).where(MatchStatus.user_id == user).values(running_since=long_ago)
    )

    assert (await status.read_view(db_session, user)).state is ScoringState.PENDING
    assert await status.begin_run(db_session, user) is not None  # a new run may start


async def test_nothing_to_run_without_a_request(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)

    assert await status.begin_run(db_session, user) is None  # no row at all


async def test_an_aborted_run_clears_itself_and_stays_pending(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None

    await status.abort_run(db_session, ticket)

    stored = await row(db_session, user)
    assert stored.running_since is None and stored.finished_seq == 0
    assert (await status.read_view(db_session, user)).state is ScoringState.PENDING


# --- failures ----------------------------------------------------------------------------------


async def test_three_failures_in_a_row_stop_the_user_and_show_it(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)

    for _ in range(status.MAX_FAILURES):
        await status.record_failure(db_session, user, status.ERROR_UNEXPECTED)

    view = await status.read_view(db_session, user)
    assert view.state is ScoringState.FAILED
    assert (view.failure_count, view.max_attempts) == (3, 3)
    assert view.error == "unexpected_error"
    assert not await status.is_pending(db_session, user)  # the sweep leaves this user alone


async def test_a_background_change_does_not_clear_failures_but_a_user_change_does(
    db_session: AsyncSession,
) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)
    for _ in range(status.MAX_FAILURES):
        await status.record_failure(db_session, user, status.ERROR_UNEXPECTED)

    await status.request_scoring(db_session, user)  # a job was embedded
    assert (await status.read_view(db_session, user)).state is ScoringState.FAILED

    await status.request_scoring(db_session, user, reset_failures=True)  # the user saved something
    view = await status.read_view(db_session, user)
    assert view.state is ScoringState.PENDING and view.error is None


async def test_a_success_clears_the_failure_count(db_session: AsyncSession) -> None:
    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)
    await status.record_failure(db_session, user, status.ERROR_UNEXPECTED)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None

    await status.finish_run(db_session, ticket, fingerprint="f", candidates=0, written=0)

    stored = await row(db_session, user)
    assert (stored.failure_count, stored.last_error) == (0, None)


# --- a deleted CV ------------------------------------------------------------------------------


async def test_clearing_for_a_deleted_cv_removes_only_that_users_matches(
    db_session: AsyncSession,
) -> None:
    alice, bob = await user_with_cv(db_session), await user_with_cv(db_session)
    alice_job = await add_job(db_session, alice, body="x " * 40, stack=["Python"])
    bob_job = await add_job(db_session, bob, body="y " * 40, stack=["Python"])
    for user, job in ((alice, alice_job), (bob, bob_job)):
        db_session.add(
            JobMatch(
                user_id=user,
                job_id=job.id,
                score=50,
                reasons={},
                scorer_version="score_v1",
                inputs_hash="h" * 64,
            )
        )
    await db_session.flush()

    await status.clear_matches_and_request(db_session, alice)

    left = (await db_session.scalars(select(JobMatch.user_id))).all()
    assert left == [bob]
    assert (await row(db_session, alice)).requested_seq == 1


# --- what is stored ----------------------------------------------------------------------------


def test_the_status_row_and_the_view_hold_no_text_column() -> None:
    columns = {column.name for column in MatchStatus.__table__.columns}
    assert columns == {
        "user_id",
        "requested_seq",
        "finished_seq",
        "requested_at",
        "finished_at",
        "running_since",
        "queued_at",
        "fingerprint",
        "failure_count",
        "last_error",
        "last_candidates",
        "last_written",
        "created_at",
        "updated_at",
    }
    assert {f.name for f in dataclasses.fields(status.ScoringView)} == {
        "state",
        "requested_at",
        "finished_at",
        "failure_count",
        "max_attempts",
        "error",
    }


# --- the database refuses nonsense -------------------------------------------------------------


async def test_the_database_refuses_a_negative_failure_count(db_session: AsyncSession) -> None:
    from sqlalchemy.exc import IntegrityError

    user = await user_with_cv(db_session)
    await status.request_scoring(db_session, user)

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                update(MatchStatus).where(MatchStatus.user_id == user).values(failure_count=-1)
            )


# --- the lock the write step takes -------------------------------------------------------------


async def test_the_share_lock_of_the_write_step_holds_back_an_edit_until_it_ends(
    factory: SessionFactory,
) -> None:
    """Two real transactions: while the write step holds the CV row (FOR SHARE), an edit of that
    row has to wait, so the token it checked cannot change under it."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    from app.models import Cv
    from app.repositories import match_repository as repo

    async with factory() as session, session.begin():
        user = await add_user(session)
        await add_cv(session, user, skills=["Python"])
    async with factory() as writer, factory() as editor:
        await repo.lock_inputs(writer, user)  # the write step's first statements
        await editor.execute(text("SET LOCAL lock_timeout = '300ms'"))
        with pytest.raises(DBAPIError):  # the edit cannot proceed while the lock is held
            await editor.execute(update(Cv).where(Cv.user_id == user).values(skills=["Rust"]))
        await editor.rollback()

        await writer.commit()  # the write step is done: the lock is released

        await editor.execute(update(Cv).where(Cv.user_id == user).values(skills=["Rust"]))
        await editor.commit()
