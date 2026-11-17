"""Scoring a user's jobs against a real Postgres (pgvector included), with no model.

Vectors come from `FakeEmbedder`: texts that share words are closer, so a CV about Python and Django
is nearer to a Python/Django job than to a marketing one. These tests check the plan the service
follows (nearest first, only comparable vectors, write only what changed, never cross users), not
the quality of any embedding.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, event, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.extraction.text_hash import raw_text_hash
from app.matching.embeddings import text_hash
from app.matching.fake_embedder import FakeEmbedder
from app.matching.reasons import MatchReasons
from app.matching.scoring import SCORER_VERSION
from app.models import (
    Cv,
    CvFormat,
    ExtractionStatus,
    Job,
    JobMatch,
    JobPreferences,
    RemotePolicy,
    Seniority,
    Source,
    SourceType,
    User,
)
from app.repositories import match_repository as repo
from app.services.extraction_types import SessionFactory
from app.services.match_service import score_job_for_user, score_jobs_for_user
from tests.settings_helpers import settings_with

NOW = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def make_settings(**overrides: object) -> Settings:
    """Fake embeddings, and a floor/ceiling that map the fake cosines straight onto 0..1."""
    values: dict[str, object] = {
        "embedding_provider": "fake",
        "match_similarity_floor": 0.0,
        "match_similarity_ceiling": 1.0,
    }
    return settings_with(**(values | overrides))


SETTINGS = make_settings()
MODEL = SETTINGS.embedding_model_id

CV_TEXT = "Backend engineer with Python Django PostgreSQL api experience building services."
PYTHON_POST = (
    "We are looking for a backend engineer to build the Python Django PostgreSQL api for our team. "
    "You will work with the people who design the product and we will support you."
)
MARKETING_POST = (
    "We are looking for a marketing manager to run the brand social media campaigns for our team. "
    "You will work with the people who design the logo and we will support you."
)


def vector_of(body: str, model: str = MODEL) -> list[float]:
    return FakeEmbedder(model).embed([body])[0]


async def add_user(session: AsyncSession) -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@extraction.test", hashed_password="x")
    session.add(user)
    await session.flush()
    return user.id


async def add_cv(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    skills: list[str],
    body: str = CV_TEXT,
    embedded: bool = True,
    model: str = MODEL,
) -> Cv:
    cv = Cv(
        user_id=user_id,
        text=body,
        source_format=CvFormat.TEXT,
        char_count=len(body),
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        skills=skills,
    )
    if embedded:
        cv.embedding = vector_of(body, model)
        cv.embedding_model = model
        cv.embedding_text_hash = text_hash(model, body)
        cv.embedded_at = NOW
    session.add(cv)
    await session.flush()
    return cv


async def add_job(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    body: str,
    stack: list[str],
    embedded: bool = True,
    model: str = MODEL,
    status: ExtractionStatus = ExtractionStatus.EXTRACTED,
    **fields: object,
) -> Job:
    source = Source(user_id=user_id, type=SourceType.RSS, name="feed")
    session.add(source)
    await session.flush()
    job = Job(
        source_id=source.id,
        external_id=str(uuid.uuid4()),
        raw_text=body,
        raw_text_hash=raw_text_hash(body),
        title="Engineer",
        stack=stack,
        extraction_status=status,
        extraction_rules_version="rules_v3",
        **fields,
    )
    if embedded:
        job.embedding = vector_of(body, model)
        job.embedding_model = model
        job.embedding_text_hash = text_hash(model, body)
        job.embedded_at = NOW
    session.add(job)
    await session.flush()
    return job


async def two_jobs(session: AsyncSession, user_id: uuid.UUID) -> tuple[Job, Job]:
    near = await add_job(
        session, user_id, body=PYTHON_POST, stack=["Python", "Django", "PostgreSQL", "Docker"]
    )
    far = await add_job(session, user_id, body=MARKETING_POST, stack=["Figma", "Notion"])
    return near, far


async def matches(session: AsyncSession, user_id: uuid.UUID) -> dict[uuid.UUID, JobMatch]:
    rows = await repo.list_matches(session, user_id, include_filtered=True, limit=500)
    return {row.job_id: row for row in rows}


# --- the happy path ---


async def test_it_scores_the_nearest_jobs_first_and_stores_the_explanation(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django", "PostgreSQL"])
    near, far = await two_jobs(db_session, user)

    summary = await score_jobs_for_user(db_session, user, SETTINGS)

    assert (summary.mode, summary.candidates, summary.written) == ("vector", 2, 2)
    assert (summary.unchanged, summary.unscorable) == (0, 0)
    stored = await matches(db_session, user)
    assert stored[near.id].score > stored[far.id].score
    match = stored[near.id]
    assert match.scorer_version == SCORER_VERSION
    assert len(match.inputs_hash) == 64
    assert match.missing_skills == ["Docker"]
    assert match.filters_failed is False
    assert MatchReasons.model_validate(match.reasons).score == match.score
    ranked = await repo.list_matches(db_session, user)
    assert [row.job_id for row in ranked] == [near.id, far.id]


async def test_retrieval_is_nearest_first_and_limited(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    cv = await add_cv(db_session, user, skills=["Python"])
    near, far = await two_jobs(db_session, user)
    assert cv.embedding is not None

    both = await repo.find_candidates(
        db_session, user, cv_vector=list(cv.embedding), model_id=MODEL, limit=10
    )
    one = await repo.find_candidates(
        db_session, user, cv_vector=list(cv.embedding), model_id=MODEL, limit=1
    )

    assert [c.job_id for c in both] == [near.id, far.id]
    assert [c.job_id for c in one] == [near.id]
    assert both[0].distance is not None and both[1].distance is not None
    assert both[0].distance < both[1].distance


async def test_the_candidate_limit_is_a_setting(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    near, _far = await two_jobs(db_session, user)

    summary = await score_jobs_for_user(db_session, user, make_settings(match_candidates=1))

    assert summary.candidates == 1
    assert set(await matches(db_session, user)) == {near.id}


# --- writing only what changed ---


async def test_a_second_run_with_unchanged_inputs_writes_nothing(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django"])
    await two_jobs(db_session, user)
    await score_jobs_for_user(db_session, user, SETTINGS)
    before = {job_id: row.inputs_hash for job_id, row in (await matches(db_session, user)).items()}

    again = await score_jobs_for_user(db_session, user, SETTINGS)

    assert (again.written, again.unchanged) == (0, 2)
    after = {job_id: row.inputs_hash for job_id, row in (await matches(db_session, user)).items()}
    assert after == before


async def test_a_changed_cv_skill_list_rewrites_the_matches(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    cv = await add_cv(db_session, user, skills=["Python"])
    await two_jobs(db_session, user)
    await score_jobs_for_user(db_session, user, SETTINGS)

    await db_session.execute(update(Cv).where(Cv.id == cv.id).values(skills=["Python", "Docker"]))
    again = await score_jobs_for_user(db_session, user, SETTINGS)

    assert again.written == 2
    stored = await matches(db_session, user)
    assert any("Docker" not in row.missing_skills for row in stored.values())


async def test_preferences_change_the_matches_but_their_notes_do_not(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    await two_jobs(db_session, user)
    prefs = JobPreferences(user_id=user, remote_policies=[], notes="first note")
    db_session.add(prefs)
    await db_session.flush()
    first = await score_jobs_for_user(db_session, user, SETTINGS)

    await db_session.execute(
        update(JobPreferences).where(JobPreferences.id == prefs.id).values(notes="a different note")
    )
    notes_only = await score_jobs_for_user(db_session, user, SETTINGS)
    await db_session.execute(
        update(JobPreferences)
        .where(JobPreferences.id == prefs.id)
        .values(remote_policies=[RemotePolicy.REMOTE])
    )
    real_change = await score_jobs_for_user(db_session, user, SETTINGS)

    assert first.written == 2
    assert (notes_only.written, notes_only.unchanged) == (0, 2)
    assert real_change.written == 2


async def test_a_changed_setting_rescores_everything(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    await two_jobs(db_session, user)
    await score_jobs_for_user(db_session, user, SETTINGS)

    again = await score_jobs_for_user(db_session, user, make_settings(match_failed_filter_cap=10))

    assert again.written == 2


async def test_two_runs_at_the_same_time_leave_one_row_per_job(factory: SessionFactory) -> None:
    async with factory() as session, session.begin():
        user = await add_user(session)
        await add_cv(session, user, skills=["Python", "Django"])
        await two_jobs(session, user)

    async def run() -> None:
        async with factory() as session:
            await score_jobs_for_user(session, user, SETTINGS)

    await asyncio.gather(run(), run(), run())

    async with factory() as session:
        count = await session.scalar(
            select(text("count(*)")).select_from(JobMatch).where(JobMatch.user_id == user)
        )
    assert count == 2


# --- which jobs are candidates ---


async def test_only_comparable_extracted_jobs_of_this_user_are_candidates(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    other = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    good = await add_job(db_session, user, body=PYTHON_POST, stack=["Python"])
    await add_job(db_session, user, body=PYTHON_POST, stack=["Python"], model="other@1")
    await add_job(db_session, user, body=PYTHON_POST, stack=["Python"], embedded=False)
    await add_job(
        db_session, user, body=PYTHON_POST, stack=["Python"], status=ExtractionStatus.PENDING
    )
    await add_job(
        db_session, user, body=PYTHON_POST, stack=["Python"], status=ExtractionStatus.NOT_A_JOB
    )
    await add_job(db_session, other, body=PYTHON_POST, stack=["Python"])

    summary = await score_jobs_for_user(db_session, user, SETTINGS)

    assert summary.candidates == 1
    assert set(await matches(db_session, user)) == {good.id}


async def test_without_a_cv_vector_the_newest_jobs_are_scored_on_skills_alone(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django"], embedded=False)
    near, _far = await two_jobs(db_session, user)

    summary = await score_jobs_for_user(db_session, user, SETTINGS)

    assert summary.mode == "recent"
    assert summary.candidates == 2
    reasons = (await matches(db_session, user))[near.id].reasons
    assert reasons["notes"] == ["no_embedding"]
    assert "embedding" not in reasons["components"]
    assert reasons["weights"]["effective"] == {"skills": 1.0, "embedding": 0.0}


async def test_a_cv_vector_from_another_model_counts_as_no_vector(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"], model="other@1")
    await two_jobs(db_session, user)

    summary = await score_jobs_for_user(db_session, user, SETTINGS)

    assert summary.mode == "recent"


async def test_no_cv_is_not_found(db_session: AsyncSession) -> None:
    user = await add_user(db_session)

    with pytest.raises(NotFoundError, match="No CV uploaded"):
        await score_jobs_for_user(db_session, user, SETTINGS)


async def test_a_job_that_can_no_longer_be_scored_loses_its_row_and_gets_none(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    cv = await add_cv(db_session, user, skills=["Python"])
    job = await add_job(db_session, user, body=PYTHON_POST, stack=[])
    first = await score_jobs_for_user(db_session, user, SETTINGS)
    assert (first.mode, first.written) == ("vector", 1)  # no stack: scored on the embedding alone

    await db_session.execute(
        update(Cv)
        .where(Cv.id == cv.id)
        .values(embedding=None, embedding_model=None, embedding_text_hash=None, embedded_at=None)
    )
    second = await score_jobs_for_user(db_session, user, SETTINGS)

    assert second.mode == "recent"
    assert second.unscorable == 1
    assert job.id not in await matches(db_session, user)


# --- hard filters ---


async def test_a_failed_filter_caps_the_score_and_hides_the_match_unless_asked(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python", "Django", "PostgreSQL"])
    db_session.add(JobPreferences(user_id=user, remote_policies=[RemotePolicy.REMOTE]))
    onsite = await add_job(
        db_session,
        user,
        body=PYTHON_POST,
        stack=["Python", "Django", "PostgreSQL"],
        remote_policy=RemotePolicy.ONSITE,
    )
    remote = await add_job(
        db_session,
        user,
        body=PYTHON_POST + " Remote.",
        stack=["Python", "Django", "PostgreSQL"],
        remote_policy=RemotePolicy.REMOTE,
    )

    await score_jobs_for_user(db_session, user, SETTINGS)

    shown = await repo.list_matches(db_session, user)
    everything = await repo.list_matches(db_session, user, include_filtered=True)
    assert [row.job_id for row in shown] == [remote.id]
    assert {row.job_id for row in everything} == {onsite.id, remote.id}
    capped = next(row for row in everything if row.job_id == onsite.id)
    assert capped.filters_failed is True and capped.score <= 30
    assert capped.reasons["cap_applied"] is True


async def test_unknown_filters_do_not_hide_a_match(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    db_session.add(
        JobPreferences(user_id=user, seniority_levels=[Seniority.SENIOR])
    )  # the jobs below state no seniority
    await two_jobs(db_session, user)

    await score_jobs_for_user(db_session, user, SETTINGS)

    assert len(await repo.list_matches(db_session, user)) == 2


# --- another user's data ---


async def test_a_user_never_sees_or_scores_another_users_matches(db_session: AsyncSession) -> None:
    alice = await add_user(db_session)
    bob = await add_user(db_session)
    await add_cv(db_session, alice, skills=["Python"])
    await add_cv(db_session, bob, skills=["Python"])
    alices_job, _ = await two_jobs(db_session, alice)
    await score_jobs_for_user(db_session, alice, SETTINGS)

    assert await repo.list_matches(db_session, bob, include_filtered=True) == []
    assert await repo.get_match(db_session, bob, alices_job.id) is None
    with pytest.raises(NotFoundError, match="Job not found"):
        await score_job_for_user(db_session, bob, alices_job.id, SETTINGS)
    assert await repo.existing_hashes(db_session, bob, [alices_job.id]) == {}
    assert await repo.list_matches(db_session, alice)  # her own are still there


async def test_a_match_row_that_points_at_someone_elses_job_is_never_listed(
    db_session: AsyncSession,
) -> None:
    # Nothing in the code writes such a row; this proves the read side does not trust the table.
    alice = await add_user(db_session)
    bob = await add_user(db_session)
    alices_job, _ = await two_jobs(db_session, alice)
    db_session.add(
        JobMatch(
            user_id=bob,
            job_id=alices_job.id,
            score=99,
            reasons={},
            scorer_version="x",
            inputs_hash="0" * 64,
        )
    )
    await db_session.flush()

    assert await repo.list_matches(db_session, bob, include_filtered=True) == []


async def test_scoring_one_job_scores_just_that_job(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    near, far = await two_jobs(db_session, user)

    summary = await score_job_for_user(db_session, user, near.id, SETTINGS)

    assert summary.candidates == 1
    assert set(await matches(db_session, user)) == {near.id}
    assert far.id not in await matches(db_session, user)


async def test_scoring_one_job_that_is_missing_or_not_extracted_is_not_found(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    pending = await add_job(
        db_session, user, body=PYTHON_POST, stack=["Python"], status=ExtractionStatus.PENDING
    )

    for job_id in (uuid.uuid4(), pending.id):
        with pytest.raises(NotFoundError):
            await score_job_for_user(db_session, user, job_id, SETTINGS)


# --- the table's own rules ---


def match_row(user_id: uuid.UUID, job_id: uuid.UUID, **fields: object) -> JobMatch:
    values: dict[str, object] = {
        "score": 50,
        "reasons": {},
        "scorer_version": "x",
        "inputs_hash": "0" * 64,
    }
    return JobMatch(user_id=user_id, job_id=job_id, **(values | fields))


@pytest.mark.parametrize(
    "fields",
    [{"score": 101}, {"score": -1}, {"missing_skills": [f"s{i}" for i in range(31)]}],
    ids=["score above 100", "score below 0", "31 missing skills"],
)
async def test_the_database_refuses_an_out_of_range_row(
    db_session: AsyncSession, fields: dict[str, object]
) -> None:
    user = await add_user(db_session)
    job, _ = await two_jobs(db_session, user)

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(match_row(user, job.id, **fields))


async def test_a_user_has_one_match_per_job(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    job, _ = await two_jobs(db_session, user)
    db_session.add(match_row(user, job.id))
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(match_row(user, job.id))


async def test_deleting_a_job_or_a_user_deletes_their_matches(db_session: AsyncSession) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    near, far = await two_jobs(db_session, user)
    await score_jobs_for_user(db_session, user, SETTINGS)

    await db_session.execute(delete(Job).where(Job.id == near.id))
    assert set(await matches(db_session, user)) == {far.id}
    await db_session.execute(delete(User).where(User.id == user))
    count = await db_session.scalar(select(text("count(*)")).select_from(JobMatch))
    assert count == 0


# --- privacy ---


async def test_the_cv_text_and_the_preference_notes_are_never_selected(
    db_session: AsyncSession,
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    db_session.add(JobPreferences(user_id=user, notes="private words"))
    await two_jobs(db_session, user)
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    engine = db_session.bind.sync_engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        await score_jobs_for_user(db_session, user, SETTINGS)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    from_cv = [s for s in selects if "FROM cvs" in s]
    from_prefs = [s for s in selects if "FROM job_preferences" in s]
    assert from_cv and from_prefs  # the check below is about real statements
    assert all("cvs.text" not in s and "cvs.*" not in s for s in from_cv)
    assert all("notes" not in s and "job_preferences.*" not in s for s in from_prefs)
    assert not any("FROM cvs" in s and "RETURNING" in s for s in statements)


async def test_no_cv_text_preference_note_or_skill_leaks_into_logs_or_the_stored_explanation(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    canary_text = "CANARY-CV-TEXT-7f3a"
    canary_skill = "CanarySkill7f3a"
    canary_note = "CANARY-NOTE-7f3a"
    user = await add_user(db_session)
    await add_cv(
        db_session,
        user,
        skills=["Python", canary_skill],
        body=f"{CV_TEXT} {canary_text}",
    )
    db_session.add(JobPreferences(user_id=user, notes=canary_note))
    await two_jobs(db_session, user)

    with caplog.at_level(logging.DEBUG):
        await score_jobs_for_user(db_session, user, SETTINGS)

    seen = " ".join(r.getMessage() + str(r.args) for r in caplog.records)
    stored = str(
        [(m.reasons, m.missing_skills) for m in (await matches(db_session, user)).values()]
    )
    for canary in (canary_text, canary_skill, canary_note):
        assert canary not in seen
        assert canary not in stored
    assert any("Scoring run" in r.getMessage() for r in caplog.records)


async def test_the_log_line_carries_counts_only(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    user = await add_user(db_session)
    await add_cv(db_session, user, skills=["Python"])
    await two_jobs(db_session, user)

    with caplog.at_level(logging.INFO, logger="app.services.match_service"):
        await score_jobs_for_user(db_session, user, SETTINGS)

    (record,) = [r for r in caplog.records if r.name == "app.services.match_service"]
    message = record.getMessage()
    assert message.startswith(f"Scoring run: user={user} mode=vector candidates=2 written=2")
    for word in ("Python", "Django", "Engineer", "marketing"):
        assert word not in message
