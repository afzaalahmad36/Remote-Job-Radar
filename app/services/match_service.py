"""Score a user's jobs: read the stored inputs, score each candidate, store what changed.

No language model, no model of any kind, no network: the vectors were made earlier by the embedding
worker and are only read here. The CV is read as its skills and its vector; the CV text and the
preference notes are never selected (a test inspects the SQL), and the logs carry counts only.

Python idiom: the session is passed in. A run is three steps: read (and end that transaction),
compute (pure CPU, nothing locked), then one short write transaction that locks the CV and
preferences rows, checks they did not change, writes and commits.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.matching.inputs_hash import match_inputs_hash, scoring_fingerprint
from app.matching.scoring import (
    SCORER_VERSION,
    CvFacts,
    JobFacts,
    PreferenceFacts,
    ScoringConfig,
    score_match,
)
from app.repositories import match_repository as repo
from app.repositories.match_repository import CandidateRow, CvRow, PreferenceRow
from app.services import match_status_service as status
from app.services.match_status_service import RunTicket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreSummary:
    """What one run did, in counts."""

    mode: Literal[
        "vector", "recent"
    ]  # nearest by cosine distance, or newest (the CV has no vector)
    candidates: int
    written: int  # new matches and matches whose inputs changed
    unchanged: int  # inputs_hash equal: nothing written
    unscorable: int  # nothing to compare: no row (a row that existed is removed)
    # False when the CV or the preferences changed while the scores were computed: nothing was
    # written, and the change that caused it already requested another run.
    completed: bool = True
    removed: int = 0  # matches deleted because their job is no longer among the candidates


def scoring_config(settings: Settings) -> ScoringConfig:
    return ScoringConfig(
        weight_skills=settings.match_weight_skills,
        weight_embedding=settings.match_weight_embedding,
        similarity_floor=settings.match_similarity_floor,
        similarity_ceiling=settings.match_similarity_ceiling,
        failed_filter_cap=settings.match_failed_filter_cap,
    )


def _vector_id(model: str | None, text_hash: str | None, model_id: str) -> str | None:
    """Identity of a vector, only when the current model made it (a stale one is not used)."""
    if model != model_id or text_hash is None:
        return None
    return f"{model}|{text_hash}"


def _cv_facts(cv: CvRow, model_id: str) -> CvFacts:
    usable = cv.vector is not None
    return CvFacts(
        skills=cv.skills,
        vector_id=_vector_id(cv.embedding_model, cv.embedding_text_hash, model_id)
        if usable
        else None,
    )


def _preference_facts(row: PreferenceRow | None) -> PreferenceFacts | None:
    if row is None:
        return None
    return PreferenceFacts(
        seniority_levels=row.seniority_levels,
        remote_policies=row.remote_policies,
        min_salary=row.min_salary,
        salary_currency=row.salary_currency,
        salary_period=row.salary_period,
    )


def _job_facts_and_cosine(
    candidate: CandidateRow, cv: CvFacts, model_id: str
) -> tuple[JobFacts, float | None]:
    """The job's facts, and its cosine similarity to the CV when both vectors are usable."""
    job_vector = _vector_id(candidate.embedding_model, candidate.embedding_text_hash, model_id)
    cosine = (
        1.0 - candidate.distance
        if cv.vector_id is not None and job_vector is not None and candidate.distance is not None
        else None
    )
    facts = JobFacts(
        stack=candidate.stack,
        seniority=candidate.seniority,
        remote_policy=candidate.remote_policy,
        salary_min=candidate.salary_min,
        salary_max=candidate.salary_max,
        salary_currency=candidate.salary_currency,
        salary_period=candidate.salary_period,
        text_start=candidate.text_start,
        # Only a vector that took part in the score belongs to its inputs.
        vector_id=job_vector if cosine is not None else None,
        raw_text_hash=candidate.raw_text_hash,
        rules_version=candidate.rules_version,
    )
    return facts, cosine


@dataclass
class _Plan:
    """What a run read and what it computed, ready to be written in one short transaction."""

    user_id: uuid.UUID
    mode: Literal["vector", "recent"]
    candidate_ids: list[uuid.UUID]
    rows: list[dict[str, object]]
    stale: list[uuid.UUID]  # jobs that can no longer be scored but still have a match row
    unchanged: int
    unscorable: int
    token: repo.InputsToken  # which CV and preferences the scores were computed from
    full: bool  # all of the user's candidates (not one job): a complete picture


def _score_candidates(
    candidates: list[CandidateRow],
    stored: dict[uuid.UUID, str],
    cv: CvFacts,
    preferences: PreferenceFacts | None,
    settings: Settings,
) -> tuple[list[dict[str, object]], list[uuid.UUID], int, int]:
    """Score every candidate whose inputs changed. Pure CPU, no database, no lock held."""
    config = scoring_config(settings)
    rows: list[dict[str, object]] = []
    stale: list[uuid.UUID] = []
    unchanged = unscorable = 0
    for candidate in candidates:
        job, cosine = _job_facts_and_cosine(candidate, cv, settings.embedding_model_id)
        digest = match_inputs_hash(cv=cv, preferences=preferences, job=job, config=config)
        if stored.get(candidate.job_id) == digest:
            unchanged += 1
            continue
        result = score_match(cv=cv, preferences=preferences, job=job, cosine=cosine, config=config)
        if result is None:
            unscorable += 1
            if candidate.job_id in stored:
                stale.append(candidate.job_id)
            continue
        rows.append(
            {
                "job_id": candidate.job_id,
                "score": result.score,
                "filters_failed": result.filters_failed,
                "missing_skills": result.missing_skills,
                "reasons": result.reasons.to_json(),
                "scorer_version": SCORER_VERSION,
                "inputs_hash": digest,
            }
        )
    return rows, stale, unchanged, unscorable


async def _prepare(
    session: AsyncSession, user_id: uuid.UUID, settings: Settings, job_id: uuid.UUID | None
) -> _Plan:
    """Read the inputs, then compute the scores. The read transaction ends before any computing.

    Nothing is locked here: the rows are read as they are and a token records which versions of
    the CV and the preferences were read, so the write step can tell if they changed meanwhile.
    """
    cv_row = await repo.load_cv_row(session, user_id)
    if cv_row is None:
        raise NotFoundError("No CV uploaded")
    cv = _cv_facts(cv_row, settings.embedding_model_id)
    candidates = await repo.find_candidates(
        session,
        user_id,
        cv_vector=cv_row.vector if cv.vector_id is not None else None,
        model_id=settings.embedding_model_id,
        limit=1 if job_id is not None else settings.match_candidates,
        job_id=job_id,
    )
    if job_id is not None and not candidates:
        raise NotFoundError("Job not found")
    preference_row = await repo.load_preference_row(session, user_id)
    stored = await repo.existing_hashes(session, user_id, [c.job_id for c in candidates])
    await session.commit()  # the reading is over: do not hold a transaction while computing
    rows, stale, unchanged, unscorable = _score_candidates(
        candidates, stored, cv, _preference_facts(preference_row), settings
    )
    return _Plan(
        user_id=user_id,
        mode="vector" if cv.vector_id is not None else "recent",
        candidate_ids=[c.job_id for c in candidates],
        rows=rows,
        stale=stale,
        unchanged=unchanged,
        unscorable=unscorable,
        token=repo.token_of(cv_row, preference_row),
        full=job_id is None,
    )


async def _write(
    session: AsyncSession, plan: _Plan, settings: Settings, ticket: RunTicket | None
) -> ScoreSummary:
    """The short write transaction: lock the inputs, check them, write, finish, commit.

    The scores are already computed, so the locks (FOR SHARE on the CV and preferences rows) are
    held for a handful of statements only. If either row changed since it was read, nothing is
    written and the run is cleared: the change raised the request number, so the user stays
    pending and another run follows.

    Matches of jobs outside this run's candidates are deleted only when the run is complete (all
    of the user's candidates, inputs unchanged, a ticket from the status table) and there was at
    least one candidate: an empty list usually means the jobs have no vectors yet, not that the
    old matches are wrong.
    """
    summary = ScoreSummary(
        plan.mode, len(plan.candidate_ids), 0, plan.unchanged, plan.unscorable, completed=False
    )
    if await repo.lock_inputs(session, plan.user_id) != plan.token:
        if ticket is not None:
            await status.abort_run(session, ticket)
        await session.commit()
        return summary
    written = await repo.upsert_matches(session, plan.user_id, plan.rows)
    await repo.delete_matches(session, plan.user_id, plan.stale)
    removed = 0
    if ticket is not None and plan.full and plan.candidate_ids:
        removed = await repo.delete_matches_outside(session, plan.user_id, plan.candidate_ids)
    if ticket is not None:
        await status.finish_run(
            session,
            ticket,
            fingerprint=scoring_fingerprint(scoring_config(settings)),
            candidates=len(plan.candidate_ids),
            written=written,
        )
    await session.commit()
    return ScoreSummary(
        plan.mode,
        len(plan.candidate_ids),
        written,
        plan.unchanged,
        plan.unscorable,
        completed=True,
        removed=removed,
    )


async def score_jobs_for_user(
    session: AsyncSession,
    user_id: uuid.UUID,
    settings: Settings,
    ticket: RunTicket | None = None,
) -> ScoreSummary:
    """Score the user's `MATCH_CANDIDATES` nearest jobs (newest ones when the CV has no vector).

    Raises NotFoundError when the user has no CV. A job whose inputs did not change since the last
    run is not written again. With a `ticket` (from `match_status_service.begin_run`) the run also
    removes matches outside its candidates and records that it finished; without one it only
    writes scores.
    """
    started = time.monotonic()
    plan = await _prepare(session, user_id, settings, None)
    summary = await _write(session, plan, settings, ticket)
    logger.info(  # counts only: never a skill, a preference or an explanation
        "Scoring run: user=%s mode=%s candidates=%d written=%d unchanged=%d unscorable=%d "
        "removed=%d completed=%s seconds=%.2f",
        user_id,
        summary.mode,
        summary.candidates,
        summary.written,
        summary.unchanged,
        summary.unscorable,
        summary.removed,
        summary.completed,
        time.monotonic() - started,
    )
    return summary


async def score_job_for_user(
    session: AsyncSession, user_id: uuid.UUID, job_id: uuid.UUID, settings: Settings
) -> ScoreSummary:
    """Score one job for the user. A job that is not an extracted job of the user's own sources
    (another user's, a missing id, a post that is not a job) is NotFoundError, never a 403."""
    plan = await _prepare(session, user_id, settings, job_id)
    return await _write(session, plan, settings, None)
