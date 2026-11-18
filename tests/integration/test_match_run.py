"""One scoring run end to end: start, score, write under a short lock, finish.

Most tests use committed data and real, separate transactions (the `factory` fixture), because
what they check is exactly what happens between transactions: a change made while a run is
computing, a run that must not write, a failure that must be counted without leaking its message.
"""

import logging
import uuid

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.matching.inputs_hash import scoring_fingerprint
from app.models import Cv, Job, JobMatch, MatchStatus
from app.repositories import match_repository as repo
from app.services import match_service
from app.services import match_status_service as status
from app.services.extraction_types import SessionFactory
from app.services.match_runner import RunOutcome, run_user_scoring
from app.services.match_service import score_job_for_user, score_jobs_for_user, scoring_config
from app.services.match_status_service import ScoringState
from tests.integration.test_match_service import (
    SETTINGS,
    add_cv,
    add_user,
    make_settings,
    two_jobs,
)


async def seeded_user(
    factory: SessionFactory, *, cv: bool = True
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A user with (optionally) a CV and two jobs, and a scoring request open."""
    async with factory() as session, session.begin():
        user = await add_user(session)
        jobs: list[uuid.UUID] = []
        if cv:
            await add_cv(session, user, skills=["Python", "Django", "PostgreSQL"])
            jobs = [job.id for job in await two_jobs(session, user)]
        await status.request_scoring(session, user)
    return user, jobs


async def stored_matches(factory: SessionFactory, user: uuid.UUID) -> list[JobMatch]:
    async with factory() as session:
        return list(await session.scalars(select(JobMatch).where(JobMatch.user_id == user)))


async def view(factory: SessionFactory, user: uuid.UUID) -> status.ScoringView:
    async with factory() as session:
        return await status.read_view(session, user)


# --- the normal run ----------------------------------------------------------------------------


async def test_a_requested_run_scores_the_jobs_and_marks_the_user_up_to_date(
    factory: SessionFactory,
) -> None:
    user, _ = await seeded_user(factory)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert (result.outcome, result.pending_after) == (RunOutcome.COMPLETED, False)
    assert len(await stored_matches(factory, user)) == 2
    assert (await view(factory, user)).state is ScoringState.UP_TO_DATE
    async with factory() as session:
        stored = await session.get_one(MatchStatus, user)
    assert stored.fingerprint == scoring_fingerprint(scoring_config(SETTINGS))
    assert (stored.last_candidates, stored.last_written) == (2, 2)
    assert stored.running_since is None


async def test_without_a_request_nothing_runs(factory: SessionFactory) -> None:
    async with factory() as session, session.begin():
        user = await add_user(session)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert result.outcome is RunOutcome.SKIPPED


async def test_a_user_whose_cv_is_gone_is_settled_without_scoring(factory: SessionFactory) -> None:
    user, _ = await seeded_user(factory, cv=False)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert (result.outcome, result.pending_after) == (RunOutcome.NO_CV, False)
    async with factory() as session:
        stored = await session.get_one(MatchStatus, user)
    assert stored.finished_seq == stored.requested_seq and stored.running_since is None
    assert (await view(factory, user)).state is ScoringState.NO_CV


# --- a change during the run -------------------------------------------------------------------


async def test_a_request_made_while_the_run_computes_leaves_the_user_pending(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timeline test, through the real runner: the second request arrives after the run began
    and before it finished, so the run's result must not close it."""
    user, _ = await seeded_user(factory)
    original = repo.existing_hashes

    async def request_meanwhile(
        session: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        async with factory() as other, other.begin():
            await status.request_scoring(other, user)  # e.g. a job was embedded
        return await original(session, user_id, job_ids)

    monkeypatch.setattr(repo, "existing_hashes", request_meanwhile)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert result.outcome is RunOutcome.COMPLETED  # the scores were written...
    assert result.pending_after is True  # ...but a request is still open
    assert (await view(factory, user)).state is ScoringState.PENDING
    async with factory() as session:
        stored = await session.get_one(MatchStatus, user)
    assert (stored.requested_seq, stored.finished_seq) == (2, 1)


async def test_a_cv_edited_while_the_run_computes_writes_nothing_and_stays_pending(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, _ = await seeded_user(factory)
    original = repo.existing_hashes

    async def edit_skills_meanwhile(
        session: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        async with factory() as other, other.begin():  # a real, committed transaction
            await other.execute(
                update(Cv).where(Cv.user_id == user).values(skills=["Figma"], updated_at=func.now())
            )
            await status.request_scoring(other, user, reset_failures=True)
        return await original(session, user_id, job_ids)

    monkeypatch.setattr(repo, "existing_hashes", edit_skills_meanwhile)

    first = await run_user_scoring(factory, user, SETTINGS)

    assert first.outcome is RunOutcome.ABORTED and first.pending_after is True
    assert await stored_matches(factory, user) == []  # scores of the old skills were not written
    async with factory() as session:
        assert (await session.get_one(MatchStatus, user)).running_since is None

    monkeypatch.setattr(repo, "existing_hashes", original)
    second = await run_user_scoring(factory, user, SETTINGS)  # the follow-up run
    assert (second.outcome, second.pending_after) == (RunOutcome.COMPLETED, False)
    assert len(await stored_matches(factory, user)) == 2


async def test_preferences_saved_while_the_run_computes_also_abort_it(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import JobPreferences

    user, _ = await seeded_user(factory)
    original = repo.existing_hashes

    async def save_preferences_meanwhile(
        session: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        async with factory() as other, other.begin():
            other.add(JobPreferences(user_id=user, min_salary=None))
            await status.request_scoring(other, user, reset_failures=True)
        return await original(session, user_id, job_ids)

    monkeypatch.setattr(repo, "existing_hashes", save_preferences_meanwhile)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert result.outcome is RunOutcome.ABORTED
    assert await stored_matches(factory, user) == []


# --- failures ----------------------------------------------------------------------------------


async def test_a_bug_is_counted_with_a_fixed_code_and_its_message_is_never_logged(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    user, _ = await seeded_user(factory)

    async def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("quoted skill: Python Django secret-canary")

    monkeypatch.setattr(match_service, "score_jobs_for_user", broken)
    caplog.set_level(logging.DEBUG)

    result = await run_user_scoring(factory, user, SETTINGS)

    assert (result.outcome, result.pending_after) == (RunOutcome.FAILED, False)
    state = await view(factory, user)
    assert (state.failure_count, state.error) == (1, "unexpected_error")
    assert "secret-canary" not in caplog.text
    assert "RuntimeError" in caplog.text  # the type is logged, the message is not
    async with factory() as session:
        assert (await session.get_one(MatchStatus, user)).running_since is None


async def test_three_failed_runs_put_the_user_in_the_failed_state(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, _ = await seeded_user(factory)

    async def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("x")

    monkeypatch.setattr(match_service, "score_jobs_for_user", broken)

    for _ in range(status.MAX_FAILURES):
        await run_user_scoring(factory, user, SETTINGS)

    assert (await view(factory, user)).state is ScoringState.FAILED


async def test_a_database_outage_is_raised_for_celery_to_retry_and_clears_the_run(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, _ = await seeded_user(factory)

    async def down(*args: object, **kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection lost"))

    monkeypatch.setattr(match_service, "score_jobs_for_user", down)

    with pytest.raises(OperationalError):
        await run_user_scoring(factory, user, SETTINGS)

    async with factory() as session:
        stored = await session.get_one(MatchStatus, user)
    assert stored.running_since is None  # the retry is not refused as "already running"
    assert stored.failure_count == 0  # an outage is not the user's scoring failing


# --- a short write transaction -----------------------------------------------------------------


async def test_the_scores_are_computed_before_the_lock_and_the_locked_part_is_short(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django"])
    await two_jobs(db_session, user)
    await status.request_scoring(db_session, user)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None

    events: list[str] = []
    original = match_service._score_candidates

    def marked(*args: object, **kwargs: object) -> object:
        events.append("COMPUTE")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    def record(conn: object, cursor: object, statement: str, *rest: object) -> None:
        events.append("LOCK" if "FOR SHARE" in statement else statement)

    monkeypatch.setattr(match_service, "_score_candidates", marked)
    engine = db_session.bind.sync_engine  # type: ignore[union-attr]
    event.listen(engine, "before_cursor_execute", record)
    try:
        summary = await score_jobs_for_user(db_session, user, SETTINGS, ticket)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert summary.completed
    compute, first_lock = events.index("COMPUTE"), events.index("LOCK")
    assert compute < first_lock  # everything slow happened before anything was locked
    before_lock = events[:first_lock]
    assert any("FROM jobs" in line for line in before_lock)  # candidates were read earlier
    locked = events[first_lock:]
    assert not any("FROM jobs" in line for line in locked)  # and not again while holding locks
    # Two locks, the upsert, the finish and nothing else of note: a handful of statements.
    assert len(locked) <= 6, locked


# --- removing matches that no longer apply -----------------------------------------------------


async def scored_with_two_matches(db_session: AsyncSession) -> uuid.UUID:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django", "PostgreSQL"])
    await two_jobs(db_session, user)
    await status.request_scoring(db_session, user)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None
    await score_jobs_for_user(db_session, user, SETTINGS, ticket)
    assert (
        len((await db_session.scalars(select(JobMatch).where(JobMatch.user_id == user))).all()) == 2
    )
    return user


async def match_count(db_session: AsyncSession, user: uuid.UUID) -> int:
    rows = await db_session.scalars(select(JobMatch).where(JobMatch.user_id == user))
    return len(rows.all())


async def next_ticket(db_session: AsyncSession, user: uuid.UUID) -> status.RunTicket:
    await status.request_scoring(db_session, user)
    ticket = await status.begin_run(db_session, user)
    assert ticket is not None
    return ticket


async def test_a_complete_run_removes_matches_outside_its_candidates(
    db_session: AsyncSession,
) -> None:
    user = await scored_with_two_matches(db_session)
    narrow = make_settings(match_candidates=1)

    summary = await score_jobs_for_user(
        db_session, user, narrow, await next_ticket(db_session, user)
    )

    assert (summary.completed, summary.candidates, summary.removed) == (True, 1, 1)
    assert await match_count(db_session, user) == 1


async def test_a_run_without_a_ticket_never_removes_anything(db_session: AsyncSession) -> None:
    user = await scored_with_two_matches(db_session)

    summary = await score_jobs_for_user(db_session, user, make_settings(match_candidates=1))

    assert (summary.candidates, summary.removed) == (1, 0)
    assert await match_count(db_session, user) == 2


async def test_scoring_one_job_never_removes_the_others(db_session: AsyncSession) -> None:
    user = await scored_with_two_matches(db_session)
    job_id = (await db_session.scalars(select(Job.id).limit(1))).one()

    await score_job_for_user(db_session, user, job_id, SETTINGS)

    assert await match_count(db_session, user) == 2


async def test_an_aborted_run_removes_nothing(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = await scored_with_two_matches(db_session)
    ticket = await next_ticket(db_session, user)
    original = repo.lock_inputs

    async def changed(session: AsyncSession, user_id: uuid.UUID) -> repo.InputsToken:
        token = await original(session, user_id)
        return repo.InputsToken(None, token.cv_embedded_at, token.cv_text_hash, None)

    monkeypatch.setattr(repo, "lock_inputs", changed)

    summary = await score_jobs_for_user(db_session, user, make_settings(match_candidates=1), ticket)

    assert summary.completed is False and summary.removed == 0
    assert await match_count(db_session, user) == 2
    stored = (
        await db_session.execute(select(MatchStatus).where(MatchStatus.user_id == user))
    ).scalar_one()
    assert stored.running_since is None


async def test_a_run_with_no_candidates_keeps_the_old_matches(db_session: AsyncSession) -> None:
    """No candidate usually means the jobs have no vector of the current model yet, not that the
    stored matches are wrong, so they are not wiped."""
    user = await scored_with_two_matches(db_session)
    await db_session.execute(update(Job).values(embedding_model="another-model@1"))
    ticket = await next_ticket(db_session, user)

    summary = await score_jobs_for_user(db_session, user, SETTINGS, ticket)

    assert (summary.completed, summary.candidates, summary.removed) == (True, 0, 0)
    assert await match_count(db_session, user) == 2
