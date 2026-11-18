import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api_deps import (
    CurrentUserDep,
    MatchQueueDep,
    PageDep,
    SessionDep,
    rate_limit_rescore,
)
from app.schemas.match import (
    MatchDetailOut,
    MatchJobOut,
    MatchOut,
    MatchPage,
    RescoreAcceptedOut,
    ScoringOut,
)
from app.services import match_query_service
from app.services.match_status_service import ScoringView

# Every endpoint takes CurrentUserDep, so all of them answer 401 without a valid access token.
router = APIRouter(prefix="/matches", tags=["matches"])


def _scoring_out(view: ScoringView) -> ScoringOut:
    return ScoringOut(
        state=view.state,
        requested_at=view.requested_at,
        finished_at=view.finished_at,
        failure_count=view.failure_count,
        max_attempts=view.max_attempts,
        error=view.error,
    )


@router.get("")
async def list_matches(
    user: CurrentUserDep,
    session: SessionDep,
    page: PageDep,
    min_score: Annotated[int | None, Query(ge=0, le=100)] = None,
    include_filtered: bool = False,
    source_id: uuid.UUID | None = None,
) -> MatchPage:
    """Your jobs ranked for your CV, best first. The scores are the stored ones: reading never
    recomputes, so while `scoring.state` is `pending` or `running` they may be a little behind.

    A job that fails one of your hard filters (remote policy, seniority, salary) is hidden unless
    you pass `include_filtered=true`. `source_id` limits the list to one of your sources.
    """
    data = await match_query_service.list_matches(
        session,
        user.id,
        include_filtered=include_filtered,
        min_score=min_score,
        source_id=source_id,
        limit=page.limit,
        offset=page.offset,
    )
    return MatchPage(
        items=[
            MatchOut(
                job=MatchJobOut.model_validate(job),
                score=match.score,
                filters_failed=match.filters_failed,
                missing_skills=list(match.missing_skills),
                scorer_version=match.scorer_version,
                updated_at=match.updated_at,
            )
            for match, job in data.rows
        ],
        total=data.total,
        limit=page.limit,
        offset=page.offset,
        scoring=_scoring_out(data.scoring),
    )


@router.get("/status")
async def get_scoring_status(user: CurrentUserDep, session: SessionDep) -> ScoringOut:
    """Is scoring finished? A cheap read of the state, for polling after a CV upload."""
    return _scoring_out(await match_query_service.get_scoring(session, user.id))


@router.post(
    "/rescore",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit_rescore)],
)
async def rescore(
    user: CurrentUserDep, session: SessionDep, queue: MatchQueueDep
) -> RescoreAcceptedOut:
    """Ask for your matches to be scored again (the scores also refresh by themselves after a CV,
    skills, preferences or job change). Returns 202: the run happens in the background; follow it
    with `GET /matches/status`. `status` is `queued`, `already_pending` or `already_running`. No
    CV: 404. If the queue is down: 503 and nothing was changed."""
    outcome, view = await match_query_service.request_rescore(session, queue, user.id)
    return RescoreAcceptedOut(status=outcome, scoring=_scoring_out(view))


@router.get("/{job_id}")
async def get_match(job_id: uuid.UUID, user: CurrentUserDep, session: SessionDep) -> MatchDetailOut:
    """The stored explanation of one match. 404 when the job is not yours, does not exist, or has
    no match yet (use `scoring` in the list to tell which)."""
    data = await match_query_service.get_match(session, user.id, job_id)
    return MatchDetailOut(
        job=MatchJobOut.model_validate(data.job),
        score=data.match.score,
        filters_failed=data.match.filters_failed,
        missing_skills=list(data.match.missing_skills),
        scorer_version=data.match.scorer_version,
        updated_at=data.match.updated_at,
        reasons=data.reasons,
        note_messages=data.note_messages,
        scoring=_scoring_out(data.scoring),
    )
