"""Where a user's matches stand: asking for a re-score, running one, and reporting the state.

Everything here is a short UPDATE or INSERT on `match_status` (see `app/models/match_status.py`).
`request_scoring` does not commit: it belongs in the same transaction as the change that made the
matches stale, so the change and the request land together or not at all. The rows hold counters,
timestamps and fixed codes only (no CV text, no skills, no notes), and this module imports nothing
that could reach a model or the network.

Python idiom: `StrEnum` is an enum whose members are also plain strings, so `ScoringState.PENDING`
serialises to `"pending"` in JSON without any conversion.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import QueueUnavailableError
from app.models import Cv, Job, JobMatch, MatchStatus, Source

logger = logging.getLogger(__name__)

# A user whose scoring failed this many times in a row is left alone (until they change something
# or ask for a re-score): a bug must not turn into a loop.
MAX_FAILURES = 3
# A run that started longer ago than this and never finished is a run that died.
RUNNING_STALE = timedelta(minutes=5)

ERROR_UNEXPECTED = "unexpected_error"  # the only code so far; never an exception message


class ScoringState(StrEnum):
    UP_TO_DATE = "up_to_date"  # the stored matches reflect every change made so far
    PENDING = "pending"  # a re-score is requested and has not finished; matches may be stale
    RUNNING = "running"  # a re-score is running now
    NO_CV = "no_cv"  # there is no CV, so there is nothing to score
    FAILED = "failed"  # scoring failed MAX_FAILURES times in a row; POST /matches/rescore retries


@dataclass(frozen=True)
class ScoringView:
    """What the API tells a client about scoring. No text, only a state and some numbers."""

    state: ScoringState
    requested_at: datetime | None
    finished_at: datetime | None
    failure_count: int
    max_attempts: int
    error: str | None


@dataclass(frozen=True)
class RunTicket:
    """A started run: which user, and the request number it will satisfy when it finishes."""

    user_id: uuid.UUID
    seq: int


class MatchScoringQueue(Protocol):
    """What the API needs to ask for a scoring run (satisfied by the Celery queue)."""

    async def enqueue_match_scoring(self, user_id: uuid.UUID) -> None: ...


async def publish_request(queue: MatchScoringQueue, user_id: uuid.UUID) -> bool:
    """Publish a scoring task after the request was stored. A failure only logs: the stored
    request is what counts, and the sweep queues it if this message is lost."""
    try:
        await queue.enqueue_match_scoring(user_id)
    except QueueUnavailableError:
        logger.warning("Scoring for user %s was not queued; the sweep will", user_id)
        return False
    return True


# --- asking for a re-score ----------------------------------------------------------------------


async def request_scoring(
    session: AsyncSession, user_id: uuid.UUID, *, reset_failures: bool = False
) -> None:
    """Record that this user's matches are stale. Does not commit.

    One upsert: the first request creates the row, later ones raise `requested_seq`. A change the
    user made (a new CV, skills, preferences, an explicit re-score) clears the failure count so a
    broken state can be retried; a background change (a job that was embedded) does not.
    """
    changed: dict[str, object] = {
        "requested_seq": MatchStatus.requested_seq + 1,
        "requested_at": func.clock_timestamp(),
        "updated_at": func.now(),
    }
    if reset_failures:
        changed |= {"failure_count": 0, "last_error": None}
    statement = (
        insert(MatchStatus)
        .values(user_id=user_id)
        .on_conflict_do_update(index_elements=[MatchStatus.user_id], set_=changed)
    )
    await session.execute(statement)


async def clear_matches_and_request(session: AsyncSession, user_id: uuid.UUID) -> None:
    """The user's CV was deleted: remove their matches and request a run that settles the state
    (it finds no CV and finishes). Does not commit."""
    await session.execute(delete(JobMatch).where(JobMatch.user_id == user_id))
    await request_scoring(session, user_id, reset_failures=True)


async def request_scoring_for_job(session: AsyncSession, job_id: uuid.UUID) -> uuid.UUID | None:
    """Request a re-score for the owner of this job. Does not commit. None if the job is gone."""
    owner = await session.scalar(
        select(Source.user_id).join(Job, Job.source_id == Source.id).where(Job.id == job_id)
    )
    if owner is not None:
        await request_scoring(session, owner)
    return owner


# --- running ------------------------------------------------------------------------------------


async def begin_run(session: AsyncSession, user_id: uuid.UUID) -> RunTicket | None:
    """Mark a run as started and commit. None when there is nothing to do or a run is in progress.

    The one UPDATE refuses while another run is in progress (started less than RUNNING_STALE ago),
    so two runs cannot overlap even if the Redis lock was lost. The ticket holds the request number
    seen *now*: a change made after this moment raises the number past it.
    """
    stale_before = func.clock_timestamp() - RUNNING_STALE
    statement = (
        update(MatchStatus)
        .where(
            MatchStatus.user_id == user_id,
            (MatchStatus.running_since.is_(None)) | (MatchStatus.running_since < stale_before),
        )
        .values(running_since=func.clock_timestamp(), updated_at=func.now())
        .returning(MatchStatus.requested_seq)
    )
    seq = await session.scalar(statement)
    await session.commit()
    return None if seq is None else RunTicket(user_id, int(seq))


async def finish_run(
    session: AsyncSession,
    ticket: RunTicket,
    *,
    fingerprint: str | None,
    candidates: int | None,
    written: int | None,
) -> None:
    """Record that the run satisfied `ticket.seq`. Does not commit (it shares the write's
    transaction). A request made during the run keeps `requested_seq` ahead: still pending."""
    values: dict[str, object] = {
        "finished_seq": func.greatest(MatchStatus.finished_seq, ticket.seq),
        "finished_at": func.clock_timestamp(),
        "running_since": None,
        "failure_count": 0,
        "last_error": None,
        "last_candidates": candidates,
        "last_written": written,
        "updated_at": func.now(),
    }
    if fingerprint is not None:
        values["fingerprint"] = fingerprint
    await session.execute(
        update(MatchStatus).where(MatchStatus.user_id == ticket.user_id).values(**values)
    )


async def abort_run(session: AsyncSession, ticket: RunTicket) -> None:
    """The run wrote nothing (its inputs changed meanwhile): clear it, leave the request pending.
    Does not commit."""
    await session.execute(
        update(MatchStatus)
        .where(MatchStatus.user_id == ticket.user_id)
        .values(running_since=None, updated_at=func.now())
    )


async def record_failure(session: AsyncSession, user_id: uuid.UUID, code: str) -> None:
    """Count a failed run (a fixed code, never an exception message) and commit."""
    await session.execute(
        update(MatchStatus)
        .where(MatchStatus.user_id == user_id)
        .values(
            failure_count=MatchStatus.failure_count + 1,
            last_error=code,
            running_since=None,
            updated_at=func.now(),
        )
    )
    await session.commit()


async def is_pending(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """True while a request is open and the user has not used up their attempts."""
    row = (
        await session.execute(
            select(MatchStatus.requested_seq, MatchStatus.finished_seq, MatchStatus.failure_count)
            .where(MatchStatus.user_id == user_id)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    return (
        row is not None
        and row.requested_seq > row.finished_seq
        and row.failure_count < MAX_FAILURES
    )


# --- reading ------------------------------------------------------------------------------------


def _is_running(running_since: datetime | None, now: datetime) -> bool:
    return running_since is not None and now - running_since < RUNNING_STALE


async def read_view(session: AsyncSession, user_id: uuid.UUID) -> ScoringView:
    """The state the API reports. Cheap: two small reads, no recomputation."""
    has_cv = bool(await session.scalar(select(exists().where(Cv.user_id == user_id))))
    row = (
        await session.execute(select(MatchStatus).where(MatchStatus.user_id == user_id))
    ).scalar_one_or_none()
    if not has_cv:
        return _view(ScoringState.NO_CV, row)
    if row is None:  # a CV from before scoring existed: the sweep will request it
        return _view(ScoringState.PENDING, None)
    now = datetime.now(UTC)
    if row.failure_count >= MAX_FAILURES:
        return _view(ScoringState.FAILED, row)
    if row.requested_seq > row.finished_seq:
        running = _is_running(row.running_since, now)
        return _view(ScoringState.RUNNING if running else ScoringState.PENDING, row)
    return _view(ScoringState.UP_TO_DATE, row)


def _view(state: ScoringState, row: MatchStatus | None) -> ScoringView:
    return ScoringView(
        state=state,
        requested_at=row.requested_at if row else None,
        finished_at=row.finished_at if row else None,
        failure_count=row.failure_count if row else 0,
        max_attempts=MAX_FAILURES,
        error=row.last_error if row and row.failure_count else None,
    )
