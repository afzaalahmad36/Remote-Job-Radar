"""One scoring run for one user, as the Celery task needs it: start, score, finish, report.

The task adds the Redis lock and the publishing; everything that decides what the database says
about the run lives here, so it can be tested without Celery or Redis. A run is several short
transactions on separate sessions (start, read, write): none stays open while scores are computed.
"""

import logging
import uuid
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.services import match_service
from app.services import match_status_service as status

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


class RunOutcome(StrEnum):
    COMPLETED = "completed"  # scored and written
    ABORTED = "aborted"  # the CV or preferences changed while scoring: nothing written
    NO_CV = "no_cv"  # the user has no CV (deleted meanwhile): nothing to score
    SKIPPED = "skipped"  # nothing was requested, or another run is in progress
    FAILED = "failed"  # a bug: counted, and the sweep stops after MAX_FAILURES


@dataclass(frozen=True)
class RunResult:
    outcome: RunOutcome
    # True when a request is still open after the run: a change arrived meanwhile, or the run was
    # aborted. The task then queues one more run, so nothing waits for the next sweep.
    pending_after: bool


async def run_user_scoring(
    factory: SessionFactory, user_id: uuid.UUID, settings: Settings
) -> RunResult:
    """Score one user's jobs and record the result in `match_status`.

    A database or Redis outage is not handled here: it propagates so Celery retries the task, and
    the run is cleared on a best-effort basis so the retry is not refused as "already running".
    Any other exception is a bug: it is counted (a fixed code, never the message) and not raised,
    so the task does not retry a failure that would repeat.
    """
    async with factory() as session:
        ticket = await status.begin_run(session, user_id)
    if ticket is None:
        return RunResult(RunOutcome.SKIPPED, False)
    try:
        async with factory() as session:
            summary = await match_service.score_jobs_for_user(session, user_id, settings, ticket)
        outcome = RunOutcome.COMPLETED if summary.completed else RunOutcome.ABORTED
    except NotFoundError:  # no CV: the request is satisfied, there is just nothing to do
        async with factory() as session:
            await status.finish_run(session, ticket, fingerprint=None, candidates=0, written=0)
            await session.commit()
        outcome = RunOutcome.NO_CV
    except (OperationalError, InterfaceError):
        await _clear_run(factory, ticket)
        raise
    except Exception as error:  # a bug in scoring: record it, never its message or traceback
        # Both can quote values (a skill name, an explanation), so only the type is logged.
        logger.error("Scoring failed for user %s: %s", user_id, type(error).__name__)
        async with factory() as session:
            await status.record_failure(session, user_id, status.ERROR_UNEXPECTED)
        return RunResult(RunOutcome.FAILED, False)
    async with factory() as session:
        pending = await status.is_pending(session, user_id)
    return RunResult(outcome, pending)


async def _clear_run(factory: SessionFactory, ticket: status.RunTicket) -> None:
    try:
        async with factory() as session:
            await status.abort_run(session, ticket)
            await session.commit()
    except SQLAlchemyError:  # the database is still down: the stale window frees the run later
        logger.warning("Could not clear the scoring run of user %s", ticket.user_id)
