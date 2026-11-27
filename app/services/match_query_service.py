"""Reading the stored matches, and asking for a re-score. Reads never compute anything.

The list and the detail come straight from `job_matches` (joined to the user's own jobs), whatever
their age. How current they are is reported separately, as the scoring state, so a client can show
the previous scores with an "updating" hint while a re-score is pending.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.matching.reasons import NOTE_MESSAGES, MatchReasons, Note
from app.models import Job, JobMatch
from app.repositories import match_repository as repo
from app.services import match_status_service as status
from app.services.match_status_service import MatchScoringQueue, ScoringState, ScoringView


class RescoreOutcome(StrEnum):
    QUEUED = "queued"
    ALREADY_PENDING = "already_pending"
    ALREADY_RUNNING = "already_running"


@dataclass(frozen=True)
class MatchPageData:
    rows: list[tuple[JobMatch, Job]]
    total: int
    scoring: ScoringView


@dataclass(frozen=True)
class MatchDetailData:
    match: JobMatch
    job: Job
    reasons: dict[str, object] | None
    note_messages: list[str]
    scoring: ScoringView


async def list_matches(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    include_filtered: bool,
    min_score: int | None,
    source_id: uuid.UUID | None,
    limit: int,
    offset: int,
) -> MatchPageData:
    """One ranked page (best first) and the scoring state. Never raises for an empty result."""
    rows, total = await repo.page_matches(
        session,
        user_id,
        include_filtered=include_filtered,
        min_score=min_score,
        source_id=source_id,
        limit=limit,
        offset=offset,
    )
    return MatchPageData(rows, total, await status.read_view(session, user_id))


async def get_match(
    session: AsyncSession, user_id: uuid.UUID, job_id: uuid.UUID
) -> MatchDetailData:
    """The stored match of one job. NotFoundError for a job that is not the user's, a job that
    does not exist, and a job that has no match yet: the three are deliberately the same."""
    found = await repo.get_match_with_job(session, user_id, job_id)
    if found is None:
        raise NotFoundError("Match not found")
    match, job = found
    reasons, messages = _explanation(match)
    return MatchDetailData(match, job, reasons, messages, await status.read_view(session, user_id))


def _stored_reasons(match: JobMatch) -> MatchReasons | None:
    """The stored explanation, checked against the current contract."""
    try:
        return MatchReasons.model_validate(match.reasons)
    except ValidationError:  # written by an older scorer: the next run replaces it
        return None


def note_codes(match: JobMatch) -> list[Note] | None:
    """The notes of a stored match, or None if its explanation cannot be read."""
    reasons = _stored_reasons(match)
    return None if reasons is None else list(reasons.notes)


def _explanation(match: JobMatch) -> tuple[dict[str, object] | None, list[str]]:
    """The stored explanation as returned by the detail, with a sentence per note."""
    reasons = _stored_reasons(match)
    if reasons is None:
        return None, []
    return reasons.to_json(), [NOTE_MESSAGES[note] for note in reasons.notes]


async def get_scoring(session: AsyncSession, user_id: uuid.UUID) -> ScoringView:
    return await status.read_view(session, user_id)


async def request_rescore(
    session: AsyncSession, queue: MatchScoringQueue, user_id: uuid.UUID
) -> tuple[RescoreOutcome, ScoringView]:
    """Ask for a fresh scoring run of the user's matches.

    NotFoundError without a CV. A run in progress is left alone. Otherwise the request is stored
    and a task queued BEFORE the commit (the task starts a few seconds later, after the commit): if
    the queue is unreachable the request is rolled back and QueueUnavailableError is raised, so
    the answer never says "queued" for a request that was not.
    """
    view = await status.read_view(session, user_id)
    if view.state is ScoringState.NO_CV:
        raise NotFoundError("No CV uploaded")
    if view.state is ScoringState.RUNNING:
        return RescoreOutcome.ALREADY_RUNNING, view
    outcome = RescoreOutcome.ALREADY_PENDING
    if view.state is not ScoringState.PENDING:
        await status.request_scoring(session, user_id, reset_failures=True)
        outcome = RescoreOutcome.QUEUED
    try:
        await queue.enqueue_match_scoring(user_id)
    except Exception:
        await session.rollback()
        raise
    await session.commit()
    return outcome, await status.read_view(session, user_id)
