"""What the match endpoints return.

Every field is listed on purpose: the job's public post fields, the score and the stored
explanation. There is no field that could carry CV text, CV skills or preference notes, and a test
pins the exact key set of each response model so one cannot be added by accident.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from app.schemas.pagination import Page
from app.services.match_query_service import RescoreOutcome
from app.services.match_status_service import ScoringState


class ScoringOut(BaseModel):
    """Whether the stored matches are current. Counters and a fixed code, nothing else."""

    state: ScoringState
    requested_at: datetime | None  # when the latest re-score was requested
    finished_at: datetime | None  # when the latest run finished
    failure_count: int  # failed runs in a row
    max_attempts: int  # after this many the sweep stops; POST /matches/rescore tries again
    error: str | None  # a fixed code such as "unexpected_error", never a message


class MatchJobOut(BaseModel):
    """The job as shown next to a score: public post fields only, no post text."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_id: uuid.UUID
    title: str | None
    company: str | None
    stack: list[str]
    seniority: Seniority | None
    remote_policy: RemotePolicy | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    posted_at: datetime | None
    apply_email: str | None
    apply_url: str | None


class MatchOut(BaseModel):
    """One ranked job."""

    job: MatchJobOut
    score: int  # 0-100: a rank for this CV, not a probability (docs/MATCHING.md)
    filters_failed: bool  # a hard filter failed, so the score is capped
    missing_skills: list[str]  # the job's own stack entries the CV lacks
    scorer_version: str
    updated_at: datetime  # when this score was last rewritten


class MatchPage(Page[MatchOut]):
    """A page of matches, with the state of scoring so a client can say "updating"."""

    scoring: ScoringOut


class MatchDetailOut(BaseModel):
    """One match with its stored explanation."""

    job: MatchJobOut
    score: int
    filters_failed: bool
    missing_skills: list[str]
    scorer_version: str
    updated_at: datetime
    # The explanation exactly as stored (`MatchReasons`); None if a row was written by an older
    # scorer whose shape this version no longer accepts. It holds job skill names, enum values,
    # numbers and fixed codes: no CV text.
    reasons: dict[str, Any] | None
    note_messages: list[str]  # one sentence per note in `reasons`
    scoring: ScoringOut


class RescoreAcceptedOut(BaseModel):
    # queued: a new request was recorded; already_pending: one was open and a task was
    # queued again; already_running: a run is in progress and nothing was changed.
    status: RescoreOutcome
    scoring: ScoringOut
