"""The SQL behind job scoring: what to read, and how to store a match.

Every function takes the `user_id` and uses it in its WHERE clause (jobs through their source's
owner, matches through their own `user_id`), and there is no variant without it: another user's
rows cannot be reached from here by accident.

Privacy: the CV is read as `skills` plus the vector columns only. `cvs.text` is never selected, and
the preferences are read without `notes`. A test inspects the compiled statements.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, null, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.matching.scoring import ENGLISH_SAMPLE_CHARS
from app.models import (
    Cv,
    ExtractionStatus,
    Job,
    JobMatch,
    JobPreferences,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
    Source,
)


@dataclass(frozen=True)
class CvRow:
    """The CV columns scoring may read. There is no text here by construction."""

    skills: list[str]
    vector: list[float] | None
    embedding_model: str | None
    embedding_text_hash: str | None
    # Timestamps only: they tell the write step whether the CV changed while scoring ran.
    updated_at: datetime
    embedded_at: datetime | None


@dataclass(frozen=True)
class PreferenceRow:
    """The preference columns scoring may read. There are no notes here by construction."""

    seniority_levels: list[Seniority]
    remote_policies: list[RemotePolicy]
    min_salary: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    updated_at: datetime  # tells the write step whether the preferences changed meanwhile


@dataclass(frozen=True)
class InputsToken:
    """Which version of the CV and the preferences a run read, as timestamps and a vector hash.

    Read again under a lock just before writing: any difference means the user changed something
    while the scores were computed, and the run writes nothing.
    """

    cv_updated_at: datetime | None
    cv_embedded_at: datetime | None
    cv_text_hash: str | None
    preferences_updated_at: datetime | None


@dataclass(frozen=True)
class CandidateRow:
    """One candidate job as plain values, with its cosine distance to the CV when there is one."""

    job_id: uuid.UUID
    stack: list[str]
    seniority: Seniority | None
    remote_policy: RemotePolicy | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    text_start: str
    embedding_model: str | None
    embedding_text_hash: str | None
    raw_text_hash: str | None
    rules_version: str | None
    distance: float | None  # pgvector's cosine distance (0 = same direction), None without a vector


async def load_cv_row(session: AsyncSession, user_id: uuid.UUID) -> CvRow | None:
    """The user's CV skills and vector identity, or None when there is no CV."""
    row = (
        await session.execute(
            select(
                Cv.skills,
                Cv.embedding,
                Cv.embedding_model,
                Cv.embedding_text_hash,
                Cv.updated_at,
                Cv.embedded_at,
            ).where(Cv.user_id == user_id)
        )
    ).one_or_none()
    if row is None:
        return None
    vector = None if row.embedding is None else [float(x) for x in row.embedding]
    return CvRow(
        list(row.skills),
        vector,
        row.embedding_model,
        row.embedding_text_hash,
        row.updated_at,
        row.embedded_at,
    )


async def load_preference_row(session: AsyncSession, user_id: uuid.UUID) -> PreferenceRow | None:
    """The user's saved preferences (without the notes), or None when they never saved any."""
    row = (
        await session.execute(
            select(
                JobPreferences.seniority_levels,
                JobPreferences.remote_policies,
                JobPreferences.min_salary,
                JobPreferences.salary_currency,
                JobPreferences.salary_period,
                JobPreferences.updated_at,
            ).where(JobPreferences.user_id == user_id)
        )
    ).one_or_none()
    if row is None:
        return None
    return PreferenceRow(
        list(row.seniority_levels),
        list(row.remote_policies),
        row.min_salary,
        row.salary_currency,
        row.salary_period,
        row.updated_at,
    )


def token_of(cv: CvRow | None, preferences: PreferenceRow | None) -> InputsToken:
    """The token of what a run read."""
    return InputsToken(
        cv_updated_at=cv.updated_at if cv else None,
        cv_embedded_at=cv.embedded_at if cv else None,
        cv_text_hash=cv.embedding_text_hash if cv else None,
        preferences_updated_at=preferences.updated_at if preferences else None,
    )


async def lock_inputs(session: AsyncSession, user_id: uuid.UUID) -> InputsToken:
    """Read the token again and lock the CV and preferences rows against changes (FOR SHARE).

    Anyone who wants to change either row (a new upload, saved skills or preferences, the
    embedding of the CV) waits until this transaction ends, which is a few statements away: the
    scores were already computed. Only the token's columns are selected, never the text.
    """
    cv = (
        await session.execute(
            select(Cv.updated_at, Cv.embedded_at, Cv.embedding_text_hash)
            .where(Cv.user_id == user_id)
            .with_for_update(read=True)
        )
    ).one_or_none()
    preferences_updated_at = await session.scalar(
        select(JobPreferences.updated_at)
        .where(JobPreferences.user_id == user_id)
        .with_for_update(read=True)
    )
    return InputsToken(
        cv_updated_at=cv.updated_at if cv else None,
        cv_embedded_at=cv.embedded_at if cv else None,
        cv_text_hash=cv.embedding_text_hash if cv else None,
        preferences_updated_at=preferences_updated_at,
    )


async def find_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    cv_vector: list[float] | None,
    model_id: str,
    limit: int,
    job_id: uuid.UUID | None = None,
) -> list[CandidateRow]:
    """The user's extracted jobs to score.

    With a CV vector: the jobs that have a vector of `model_id`, nearest first (cosine distance,
    exact scan). Without one: the newest jobs. With `job_id`: just that job, if it is the user's.
    A job of another user's source can never appear: the join filters on `sources.user_id`.
    """
    distance = (
        Job.embedding.cosine_distance(cv_vector).label("distance")
        if cv_vector is not None
        else null().label("distance")
    )
    statement = (
        select(
            Job.id,
            Job.stack,
            Job.seniority,
            Job.remote_policy,
            Job.salary_min,
            Job.salary_max,
            Job.salary_currency,
            Job.salary_period,
            func.left(Job.raw_text, ENGLISH_SAMPLE_CHARS).label("text_start"),
            Job.embedding_model,
            Job.embedding_text_hash,
            Job.raw_text_hash,
            Job.extraction_rules_version,
            distance,
        )
        .join(Source, Source.id == Job.source_id)
        .where(Source.user_id == user_id, Job.extraction_status == ExtractionStatus.EXTRACTED)
        .limit(limit)
    )
    if job_id is not None:
        statement = statement.where(Job.id == job_id)
    elif cv_vector is not None:
        statement = statement.where(
            Job.embedding.is_not(None), Job.embedding_model == model_id
        ).order_by(distance, Job.id)
    else:
        statement = statement.order_by(Job.created_at.desc(), Job.id)
    rows = (await session.execute(statement)).all()
    return [
        CandidateRow(
            job_id=r.id,
            stack=list(r.stack),
            seniority=r.seniority,
            remote_policy=r.remote_policy,
            salary_min=r.salary_min,
            salary_max=r.salary_max,
            salary_currency=r.salary_currency,
            salary_period=r.salary_period,
            text_start=r.text_start or "",
            embedding_model=r.embedding_model,
            embedding_text_hash=r.embedding_text_hash,
            raw_text_hash=r.raw_text_hash,
            rules_version=r.extraction_rules_version,
            distance=None if r.distance is None else float(r.distance),
        )
        for r in rows
    ]


async def existing_hashes(
    session: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
) -> dict[uuid.UUID, str]:
    """The inputs_hash already stored for this user's matches of these jobs."""
    if not job_ids:
        return {}
    rows = await session.execute(
        select(JobMatch.job_id, JobMatch.inputs_hash).where(
            JobMatch.user_id == user_id, JobMatch.job_id.in_(job_ids)
        )
    )
    return {job_id: digest for job_id, digest in rows}


async def upsert_matches(
    session: AsyncSession, user_id: uuid.UUID, rows: list[dict[str, Any]]
) -> int:
    """Store matches in one statement; a row whose inputs_hash did not change is left alone.

    Each dict holds job_id, score, filters_failed, missing_skills, reasons, scorer_version and
    inputs_hash. Rows go in a fixed order (by job id) so two concurrent runs cannot deadlock.
    Returns how many rows were inserted or changed.
    """
    if not rows:
        return 0
    ordered = sorted(rows, key=lambda row: row["job_id"])
    inserted = insert(JobMatch).values([{**row, "user_id": user_id} for row in ordered])
    excluded = inserted.excluded
    statement = inserted.on_conflict_do_update(
        index_elements=[JobMatch.user_id, JobMatch.job_id],
        set_={
            "score": excluded.score,
            "filters_failed": excluded.filters_failed,
            "missing_skills": excluded.missing_skills,
            "reasons": excluded.reasons,
            "scorer_version": excluded.scorer_version,
            "inputs_hash": excluded.inputs_hash,
            "updated_at": func.now(),
        },
        where=JobMatch.inputs_hash != excluded.inputs_hash,
    ).returning(JobMatch.job_id)
    return len((await session.execute(statement)).all())


async def delete_matches(
    session: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
) -> None:
    """Remove this user's matches for jobs that can no longer be scored (nothing to compare)."""
    if job_ids:
        await session.execute(
            delete(JobMatch).where(JobMatch.user_id == user_id, JobMatch.job_id.in_(job_ids))
        )


async def delete_matches_outside(
    session: AsyncSession, user_id: uuid.UUID, keep_job_ids: list[uuid.UUID]
) -> int:
    """Remove this user's matches for every job NOT in `keep_job_ids`; returns how many.

    Only for a complete run: those rows were scored for an earlier CV or earlier jobs and no
    longer say anything about the current candidates.
    """
    result = await session.execute(
        delete(JobMatch)
        .where(JobMatch.user_id == user_id, JobMatch.job_id.not_in(keep_job_ids))
        .returning(JobMatch.job_id)
    )
    return len(result.all())


async def delete_all_matches(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Remove every match of this user (their CV was deleted: there is nothing to rank against)."""
    await session.execute(delete(JobMatch).where(JobMatch.user_id == user_id))


def _visible_conditions(
    user_id: uuid.UUID, include_filtered: bool, min_score: int | None, source_id: uuid.UUID | None
) -> list[Any]:
    """The WHERE conditions of the listing: this user's matches of jobs they still have."""
    conditions: list[Any] = [
        JobMatch.user_id == user_id,
        Source.user_id == user_id,
        Job.extraction_status == ExtractionStatus.EXTRACTED,
    ]
    if not include_filtered:
        conditions.append(JobMatch.filters_failed.is_(False))
    if min_score is not None:
        conditions.append(JobMatch.score >= min_score)
    if source_id is not None:
        conditions.append(Source.id == source_id)
    return conditions


async def page_matches(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    include_filtered: bool,
    min_score: int | None,
    source_id: uuid.UUID | None,
    limit: int,
    offset: int,
) -> tuple[list[tuple[JobMatch, Job]], int]:
    """One page of the ranked list with its jobs, and how many rows match in all.

    Best score first, then the newest job. A `source_id` that is not the user's gives an empty
    page: the join on `sources.user_id` cannot reach another user's source.
    """
    conditions = _visible_conditions(user_id, include_filtered, min_score, source_id)
    rows = await session.execute(
        select(JobMatch, Job)
        .join(Job, Job.id == JobMatch.job_id)
        .join(Source, Source.id == Job.source_id)
        .where(*conditions)
        .order_by(JobMatch.score.desc(), Job.created_at.desc(), JobMatch.id)
        .limit(limit)
        .offset(offset)
    )
    total = await session.scalar(
        select(func.count())
        .select_from(JobMatch)
        .join(Job, Job.id == JobMatch.job_id)
        .join(Source, Source.id == Job.source_id)
        .where(*conditions)
    )
    return [(match, job) for match, job in rows.all()], int(total or 0)


async def get_match_with_job(
    session: AsyncSession, user_id: uuid.UUID, job_id: uuid.UUID
) -> tuple[JobMatch, Job] | None:
    """This user's match and its job, or None (also for a job that belongs to someone else)."""
    row = (
        await session.execute(
            select(JobMatch, Job)
            .join(Job, Job.id == JobMatch.job_id)
            .join(Source, Source.id == Job.source_id)
            .where(
                JobMatch.user_id == user_id,
                JobMatch.job_id == job_id,
                Source.user_id == user_id,
                Job.extraction_status == ExtractionStatus.EXTRACTED,
            )
        )
    ).one_or_none()
    return None if row is None else (row[0], row[1])


async def list_matches(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    include_filtered: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[JobMatch]:
    """The user's matches, best first. Matches that failed a hard filter are hidden unless asked.

    A match is returned only while its job is still an extracted job of one of the user's sources.
    """
    statement = (
        select(JobMatch)
        .join(Job, Job.id == JobMatch.job_id)
        .join(Source, Source.id == Job.source_id)
        .where(
            JobMatch.user_id == user_id,
            Source.user_id == user_id,
            Job.extraction_status == ExtractionStatus.EXTRACTED,
        )
        .order_by(JobMatch.score.desc(), Job.created_at.desc(), JobMatch.id)
        .limit(limit)
        .offset(offset)
    )
    if not include_filtered:
        statement = statement.where(JobMatch.filters_failed.is_(False))
    return list((await session.scalars(statement)).all())


async def get_match(
    session: AsyncSession, user_id: uuid.UUID, job_id: uuid.UUID
) -> JobMatch | None:
    """This user's match for a job, or None (also when the job belongs to someone else)."""
    result = await session.scalars(
        select(JobMatch).where(JobMatch.user_id == user_id, JobMatch.job_id == job_id)
    )
    return result.one_or_none()
