"""The match endpoints: ranked listing, detail, scoring status, re-score, and the triggers.

The API runs on the rollback-per-test session, so scores written here are the real service's
(fake embeddings, no model) and disappear after each test. Reading never recomputes: every test
that expects "previous scores while pending" writes them first and then changes the state.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_deps import get_settings
from app.core.config import Settings
from app.core.exceptions import QueueUnavailableError
from app.core.rate_limit import RateLimiter
from app.models import JobMatch, JobPreferences, MatchStatus, RemotePolicy, User
from app.schemas.match import (
    MatchDetailOut,
    MatchJobOut,
    MatchOut,
    MatchPage,
    RescoreAcceptedOut,
    ScoringOut,
)
from app.services import match_status_service as status
from app.services.match_service import score_jobs_for_user
from tests.auth_helpers import auth_headers
from tests.fakes import FakeTaskQueue
from tests.integration.test_match_service import (
    PYTHON_POST,
    SETTINGS,
    add_cv,
    add_job,
    add_user,
    two_jobs,
)

MATCHES = "/api/v1/matches"
EMAIL = "matcher@example.com"


async def signed_in(
    client: AsyncClient, session: AsyncSession, email: str = EMAIL
) -> tuple[dict[str, str], uuid.UUID]:
    headers = await auth_headers(client, email)
    user_id = await session.scalar(select(User.id).where(User.email == email))
    assert user_id is not None
    return headers, user_id


async def scored_user(
    client: AsyncClient, session: AsyncSession, email: str = EMAIL
) -> tuple[dict[str, str], uuid.UUID, list[uuid.UUID]]:
    """A signed-in user with a CV, two jobs and their stored matches (near job first)."""
    headers, user_id = await signed_in(client, session, email)
    await add_cv(session, user_id, skills=["Python", "Django", "PostgreSQL"])
    near, far = await two_jobs(session, user_id)
    await score_jobs_for_user(session, user_id, SETTINGS)
    return headers, user_id, [near.id, far.id]


async def settle(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Make the status say up to date (a request, a run that finished it)."""
    await status.request_scoring(session, user_id)
    ticket = await status.begin_run(session, user_id)
    assert ticket is not None
    await status.finish_run(session, ticket, fingerprint="f", candidates=2, written=2)
    await session.commit()


# --- the ranked list ---------------------------------------------------------------------------


async def test_the_list_is_ranked_best_first_and_says_how_current_it_is(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id, (near, far) = await scored_user(db_client, db_session)
    await settle(db_session, user_id)

    response = await db_client.get(MATCHES, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert [item["job"]["id"] for item in body["items"]] == [str(near), str(far)]
    assert body["items"][0]["score"] > body["items"][1]["score"]
    assert (body["total"], body["limit"], body["offset"]) == (2, 20, 0)
    assert body["scoring"]["state"] == "up_to_date"
    first = body["items"][0]
    assert first["missing_skills"] == ["Docker"] and first["filters_failed"] is False
    assert first["job"]["stack"] == ["Python", "Django", "PostgreSQL", "Docker"]


async def test_a_user_without_a_cv_gets_an_empty_list_and_the_no_cv_state(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, _ = await signed_in(db_client, db_session)

    body = (await db_client.get(MATCHES, headers=headers)).json()

    assert body["items"] == [] and body["total"] == 0
    assert body["scoring"]["state"] == "no_cv"


async def test_pagination_walks_the_ranking_without_gaps_or_repeats(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    await add_cv(db_session, user_id, skills=["Python"])
    for index in range(5):
        await add_job(db_session, user_id, body=PYTHON_POST + f" {index}", stack=["Python"])
    await score_jobs_for_user(db_session, user_id, SETTINGS)

    pages = [
        (await db_client.get(MATCHES, params={"limit": 2, "offset": o}, headers=headers)).json()
        for o in (0, 2, 4)
    ]

    assert [len(page["items"]) for page in pages] == [2, 2, 1]
    assert {page["total"] for page in pages} == {5}
    ids = [item["job"]["id"] for page in pages for item in page["items"]]
    assert len(ids) == len(set(ids)) == 5
    scores = [item["score"] for page in pages for item in page["items"]]
    assert scores == sorted(scores, reverse=True)


async def test_min_score_keeps_only_the_better_matches(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, _, (near, far) = await scored_user(db_client, db_session)
    scores = {
        item["job"]["id"]: item["score"]
        for item in (await db_client.get(MATCHES, headers=headers)).json()["items"]
    }
    threshold = scores[str(near)]
    assert scores[str(far)] < threshold

    body = (await db_client.get(MATCHES, params={"min_score": threshold}, headers=headers)).json()

    assert [item["job"]["id"] for item in body["items"]] == [str(near)]
    assert body["total"] == 1


async def test_jobs_that_fail_a_hard_filter_are_hidden_unless_asked_for(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    await add_cv(db_session, user_id, skills=["Python"])
    db_session.add(JobPreferences(user_id=user_id, remote_policies=[RemotePolicy.REMOTE]))
    ok = await add_job(
        db_session, user_id, body=PYTHON_POST, stack=["Python"], remote_policy=RemotePolicy.REMOTE
    )
    bad = await add_job(
        db_session,
        user_id,
        body=PYTHON_POST + " office",
        stack=["Python"],
        remote_policy=RemotePolicy.ONSITE,
    )
    await score_jobs_for_user(db_session, user_id, SETTINGS)

    shown = (await db_client.get(MATCHES, headers=headers)).json()
    everything = (
        await db_client.get(MATCHES, params={"include_filtered": "true"}, headers=headers)
    ).json()

    assert [i["job"]["id"] for i in shown["items"]] == [str(ok.id)] and shown["total"] == 1
    assert {i["job"]["id"] for i in everything["items"]} == {str(ok.id), str(bad.id)}
    failed = next(i for i in everything["items"] if i["job"]["id"] == str(bad.id))
    assert failed["filters_failed"] is True and failed["score"] <= SETTINGS.match_failed_filter_cap


async def test_the_list_can_be_limited_to_one_of_your_sources(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, _, (near, far) = await scored_user(db_client, db_session)
    source_of_far = (await db_client.get(MATCHES, headers=headers)).json()["items"][1]["job"][
        "source_id"
    ]

    body = (
        await db_client.get(MATCHES, params={"source_id": source_of_far}, headers=headers)
    ).json()

    assert [item["job"]["id"] for item in body["items"]] == [str(far)]
    assert body["total"] == 1


async def test_a_source_that_is_not_yours_gives_an_empty_page_not_an_error(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    alice_headers, _, _ = await scored_user(db_client, db_session, "alice@example.com")
    bob_headers, _ = await signed_in(db_client, db_session, "bob@example.com")
    alices_source = (await db_client.get(MATCHES, headers=alice_headers)).json()["items"][0]["job"][
        "source_id"
    ]

    response = await db_client.get(
        MATCHES, params={"source_id": alices_source}, headers=bob_headers
    )

    assert response.status_code == 200
    assert response.json()["items"] == [] and response.json()["total"] == 0


@pytest.mark.parametrize(
    "params", [{"min_score": 101}, {"min_score": -1}, {"limit": 0}, {"limit": 101}, {"offset": -1}]
)
async def test_bad_query_values_are_a_422(
    params: dict[str, int], db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, _ = await signed_in(db_client, db_session)

    assert (await db_client.get(MATCHES, params=params, headers=headers)).status_code == 422


# --- previous scores while a re-score is pending -----------------------------------------------


async def test_while_a_rescore_is_pending_the_previous_scores_are_still_returned(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await settle(db_session, user_id)
    before = (await db_client.get(MATCHES, headers=headers)).json()["items"]

    await status.request_scoring(db_session, user_id)  # e.g. the user edited a skill

    after = (await db_client.get(MATCHES, headers=headers)).json()
    assert after["scoring"]["state"] == "pending"
    assert after["items"] == before  # the same rows: reading recomputes nothing
    assert after["scoring"]["requested_at"] is not None


# --- the detail --------------------------------------------------------------------------------


async def test_the_detail_returns_the_explanation_exactly_as_stored(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id, (near, _) = await scored_user(db_client, db_session)
    stored = await db_session.scalar(
        select(JobMatch).where(JobMatch.user_id == user_id, JobMatch.job_id == near)
    )
    assert stored is not None

    response = await db_client.get(f"{MATCHES}/{near}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["reasons"] == stored.reasons  # byte for byte what was stored
    assert body["score"] == stored.score and body["scorer_version"] == stored.scorer_version
    assert body["job"]["id"] == str(near)
    assert body["note_messages"] == []  # no notes on this match


async def test_the_detail_words_each_note_with_its_fixed_sentence(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    await add_cv(db_session, user_id, skills=["Python"])
    db_session.add(JobPreferences(user_id=user_id, remote_policies=[RemotePolicy.REMOTE]))
    job = await add_job(
        db_session, user_id, body=PYTHON_POST, stack=["Python"], remote_policy=RemotePolicy.ONSITE
    )
    await score_jobs_for_user(db_session, user_id, SETTINGS)

    body = (await db_client.get(f"{MATCHES}/{job.id}", headers=headers)).json()

    assert body["reasons"]["notes"] == ["filters_failed"]
    assert body["note_messages"] == ["A hard filter failed, so the score is capped."]


async def test_an_explanation_from_an_older_scorer_is_returned_as_null_not_as_an_error(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id, (near, _) = await scored_user(db_client, db_session)
    await db_session.execute(
        update(JobMatch)
        .where(JobMatch.user_id == user_id, JobMatch.job_id == near)
        .values(reasons={"an": "older shape"})
    )

    body = (await db_client.get(f"{MATCHES}/{near}", headers=headers)).json()

    assert body["reasons"] is None and body["note_messages"] == []
    assert body["score"] is not None  # the score itself is still there


async def test_another_users_job_a_missing_job_and_an_unscored_job_are_the_same_404(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    alice_headers, _, (alices_job, _) = await scored_user(
        db_client, db_session, "alice@example.com"
    )
    bob_headers, bob_id = await signed_in(db_client, db_session, "bob@example.com")
    await add_cv(db_session, bob_id, skills=["Python"])
    unscored = await add_job(db_session, bob_id, body=PYTHON_POST, stack=["Python"])

    responses = [
        await db_client.get(f"{MATCHES}/{alices_job}", headers=bob_headers),  # not his
        await db_client.get(f"{MATCHES}/{uuid.uuid4()}", headers=bob_headers),  # does not exist
        await db_client.get(f"{MATCHES}/{unscored.id}", headers=bob_headers),  # no match yet
    ]

    assert [r.status_code for r in responses] == [404, 404, 404]  # never 403
    assert len({r.text for r in responses}) == 1  # nothing tells the three apart
    assert (
        await db_client.get(f"{MATCHES}/{alices_job}", headers=alice_headers)
    ).status_code == 200


async def test_users_only_ever_see_their_own_matches(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    alice_headers, _, alice_jobs = await scored_user(db_client, db_session, "alice@example.com")
    bob_headers, _, bob_jobs = await scored_user(db_client, db_session, "bob@example.com")

    alice = (await db_client.get(MATCHES, headers=alice_headers)).json()
    bob = (await db_client.get(MATCHES, headers=bob_headers)).json()

    assert {i["job"]["id"] for i in alice["items"]} == {str(j) for j in alice_jobs}
    assert {i["job"]["id"] for i in bob["items"]} == {str(j) for j in bob_jobs}
    assert not {i["job"]["id"] for i in alice["items"]} & {i["job"]["id"] for i in bob["items"]}


async def test_a_job_that_stopped_being_extracted_leaves_the_list(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    from app.models import ExtractionStatus, Job

    headers, _, (near, far) = await scored_user(db_client, db_session)
    await db_session.execute(
        update(Job).where(Job.id == far).values(extraction_status=ExtractionStatus.NOT_A_JOB)
    )

    body = (await db_client.get(MATCHES, headers=headers)).json()

    assert [i["job"]["id"] for i in body["items"]] == [str(near)]
    assert (await db_client.get(f"{MATCHES}/{far}", headers=headers)).status_code == 404


# --- the status --------------------------------------------------------------------------------


async def test_the_status_endpoint_walks_through_the_states(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id = await signed_in(db_client, db_session)

    async def state() -> dict[str, Any]:
        response = await db_client.get(f"{MATCHES}/status", headers=headers)
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    assert (await state())["state"] == "no_cv"
    await add_cv(db_session, user_id, skills=["Python"])
    assert (await state())["state"] == "pending"  # a CV nobody has scored yet
    await status.request_scoring(db_session, user_id)
    ticket = await status.begin_run(db_session, user_id)
    assert (await state())["state"] == "running"
    assert ticket is not None
    await status.finish_run(db_session, ticket, fingerprint="f", candidates=0, written=0)
    done = await state()
    assert done["state"] == "up_to_date" and done["finished_at"] is not None
    for _ in range(status.MAX_FAILURES):
        await status.record_failure(db_session, user_id, status.ERROR_UNEXPECTED)
    failed = await state()
    assert failed["state"] == "failed"
    assert (failed["failure_count"], failed["max_attempts"]) == (3, 3)
    assert failed["error"] == "unexpected_error"


async def test_a_run_that_died_is_reported_as_pending_not_running(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    await add_cv(db_session, user_id, skills=["Python"])
    await status.request_scoring(db_session, user_id)
    await status.begin_run(db_session, user_id)
    long_ago = datetime.now(UTC) - status.RUNNING_STALE - timedelta(minutes=1)
    await db_session.execute(
        update(MatchStatus).where(MatchStatus.user_id == user_id).values(running_since=long_ago)
    )

    body = (await db_client.get(f"{MATCHES}/status", headers=headers)).json()

    assert body["state"] == "pending"


# --- the response schemas ----------------------------------------------------------------------


def test_the_response_models_have_exactly_these_fields() -> None:
    """A new field must be added here on purpose: none of them may be able to carry CV text,
    CV skills or preference notes."""
    assert set(ScoringOut.model_fields) == {
        "state",
        "requested_at",
        "finished_at",
        "failure_count",
        "max_attempts",
        "error",
    }
    assert set(MatchJobOut.model_fields) == {
        "id",
        "source_id",
        "title",
        "company",
        "stack",
        "seniority",
        "remote_policy",
        "salary_min",
        "salary_max",
        "salary_currency",
        "salary_period",
        "posted_at",
        "apply_email",
        "apply_url",
    }
    assert set(MatchOut.model_fields) == {
        "job",
        "score",
        "filters_failed",
        "missing_skills",
        "scorer_version",
        "updated_at",
    }
    assert set(MatchPage.model_fields) == {"items", "total", "limit", "offset", "scoring"}
    assert set(MatchDetailOut.model_fields) == {
        "job",
        "score",
        "filters_failed",
        "missing_skills",
        "scorer_version",
        "updated_at",
        "reasons",
        "note_messages",
        "scoring",
    }
    assert set(RescoreAcceptedOut.model_fields) == {"status", "scoring"}


async def test_the_json_the_endpoints_return_has_exactly_those_keys(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, (near, _) = await scored_user(db_client, db_session)

    listing = (await db_client.get(MATCHES, headers=headers)).json()
    detail = (await db_client.get(f"{MATCHES}/{near}", headers=headers)).json()
    state = (await db_client.get(f"{MATCHES}/status", headers=headers)).json()
    rescore = (await db_client.post(f"{MATCHES}/rescore", headers=headers)).json()

    assert set(listing) == set(MatchPage.model_fields)
    assert set(listing["items"][0]) == set(MatchOut.model_fields)
    assert set(listing["items"][0]["job"]) == set(MatchJobOut.model_fields)
    assert set(detail) == set(MatchDetailOut.model_fields)
    assert set(detail["job"]) == set(MatchJobOut.model_fields)
    assert set(state) == set(ScoringOut.model_fields)
    assert set(rescore) == set(RescoreAcceptedOut.model_fields)
    assert set(rescore["scoring"]) == set(ScoringOut.model_fields)
    # The post's text is not among the job fields, and neither is anything of the CV.
    for body in (listing, detail):
        assert "raw_text" not in str(body)
    assert set(detail["reasons"]) == {
        "scorer_version",
        "score",
        "cap_applied",
        "weights",
        "components",
        "skills",
        "filters",
        "notes",
    }


# --- asking for a re-score ---------------------------------------------------------------------


async def test_rescore_records_a_request_and_queues_a_task(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await settle(db_session, user_id)

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["scoring"]["state"] == "pending"
    assert task_queue.match_enqueued == [user_id]  # only a user id is queued


async def test_rescore_while_a_request_is_open_queues_again_without_a_new_request(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await status.request_scoring(db_session, user_id)

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.json()["status"] == "already_pending"
    assert task_queue.match_enqueued == [user_id]  # in case the first message was lost
    row = (await db_session.execute(select(MatchStatus))).scalar_one()
    assert row.requested_seq == 1  # no second request was recorded


async def test_rescore_during_a_run_changes_nothing(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await status.request_scoring(db_session, user_id)
    await status.begin_run(db_session, user_id)

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.status_code == 202 and response.json()["status"] == "already_running"
    assert task_queue.match_enqueued == []


async def test_rescore_without_a_cv_is_a_404(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, _ = await signed_in(db_client, db_session)

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.status_code == 404
    assert task_queue.match_enqueued == []


async def test_rescore_after_three_failures_clears_them_and_tries_again(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await status.request_scoring(db_session, user_id)
    for _ in range(status.MAX_FAILURES):
        await status.record_failure(db_session, user_id, status.ERROR_UNEXPECTED)

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.json()["status"] == "queued"
    assert response.json()["scoring"]["state"] == "pending"
    assert response.json()["scoring"]["failure_count"] == 0


async def test_rescore_with_the_queue_down_is_a_503_and_changes_nothing(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await settle(db_session, user_id)
    task_queue.fail_match_enqueue_with = QueueUnavailableError("down")

    response = await db_client.post(f"{MATCHES}/rescore", headers=headers)

    assert response.status_code == 503 and response.json()["code"] == "queue_unavailable"
    state = (await db_client.get(f"{MATCHES}/status", headers=headers)).json()
    assert state["state"] == "up_to_date"  # no request was stored for a task that was not queued


async def test_rescore_has_a_rate_limit_of_its_own(
    app: FastAPI, db_client: AsyncClient, db_session: AsyncSession
) -> None:
    app.state.rescore_rate_limiter = RateLimiter(max_attempts=2, window_seconds=60)
    headers, user_id, _ = await scored_user(db_client, db_session)
    await settle(db_session, user_id)

    codes = [
        (await db_client.post(f"{MATCHES}/rescore", headers=headers)).status_code for _ in range(3)
    ]

    assert codes[:2] == [202, 202] and codes[2] == 429
    # reading is not limited by that bucket
    assert (await db_client.get(MATCHES, headers=headers)).status_code == 200


async def test_every_endpoint_needs_a_login(db_client: AsyncClient) -> None:
    job = uuid.uuid4()

    codes = [
        (await db_client.get(MATCHES)).status_code,
        (await db_client.get(f"{MATCHES}/status")).status_code,
        (await db_client.get(f"{MATCHES}/{job}")).status_code,
        (await db_client.post(f"{MATCHES}/rescore")).status_code,
    ]

    assert codes == [401, 401, 401, 401]


# --- the triggers, through the API -------------------------------------------------------------


async def put_cv(
    client: AsyncClient, headers: dict[str, str], text: str = "Python developer."
) -> Any:
    return await client.put(
        "/api/v1/cv", content=text, headers=headers | {"content-type": "text/plain"}
    )


async def test_uploading_a_cv_with_embeddings_off_requests_scoring_at_once(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id = await signed_in(db_client, db_session)

    assert (await put_cv(db_client, headers)).status_code == 200

    assert task_queue.match_enqueued == [user_id]
    row = (await db_session.execute(select(MatchStatus))).scalar_one()
    assert row.requested_seq == 1  # stored in the upload's own transaction


async def test_uploading_a_cv_with_embeddings_on_leaves_scoring_to_the_embedding_task(
    app: FastAPI,
    settings: Settings,
    db_client: AsyncClient,
    db_session: AsyncSession,
    task_queue: FakeTaskQueue,
) -> None:
    on = settings.model_copy(update={"embedding_provider": "fake"})
    app.dependency_overrides[get_settings] = lambda: on
    headers, user_id = await signed_in(db_client, db_session)

    assert (await put_cv(db_client, headers)).status_code == 200

    assert task_queue.cv_embedding_enqueued == [user_id]
    assert task_queue.match_enqueued == []  # embed_cv asks for the re-score when it is done
    row = (await db_session.execute(select(MatchStatus))).scalar_one()
    assert row.requested_seq == 1  # but the request itself is stored, so the sweep can find it


async def test_saving_skills_and_preferences_each_request_scoring(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    await put_cv(db_client, headers)
    task_queue.match_enqueued.clear()

    skills = await db_client.put("/api/v1/cv/skills", json={"skills": ["Python"]}, headers=headers)
    prefs = await db_client.put(
        "/api/v1/preferences", json={"remote_policies": ["remote"]}, headers=headers
    )

    assert (skills.status_code, prefs.status_code) == (200, 200)
    assert task_queue.match_enqueued == [user_id, user_id]
    row = (await db_session.execute(select(MatchStatus))).scalar_one()
    assert row.requested_seq == 3  # the upload, the skills, the preferences


async def test_a_failed_trigger_publish_never_fails_the_change_and_the_request_stays(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id = await signed_in(db_client, db_session)
    task_queue.fail_match_enqueue_with = QueueUnavailableError("down")

    saved = await db_client.put(
        "/api/v1/preferences", json={"remote_policies": ["remote"]}, headers=headers
    )

    assert saved.status_code == 200
    row = (await db_session.execute(select(MatchStatus))).scalar_one()
    assert row.requested_seq > row.finished_seq  # stored: the sweep queues it


async def test_deleting_the_cv_removes_the_matches_and_settles_the_state(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    assert (await db_client.get(MATCHES, headers=headers)).json()["total"] == 2

    assert (await db_client.delete("/api/v1/cv", headers=headers)).status_code == 204

    body = (await db_client.get(MATCHES, headers=headers)).json()
    assert body["items"] == [] and body["total"] == 0 and body["scoring"]["state"] == "no_cv"
    assert (await db_session.scalars(select(JobMatch))).all() == []
    assert task_queue.match_enqueued == [user_id]


async def test_uploading_a_new_cv_keeps_the_old_scores_visible_until_the_rescore(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers, user_id, _ = await scored_user(db_client, db_session)
    await settle(db_session, user_id)
    before = (await db_client.get(MATCHES, headers=headers)).json()["items"]

    assert (
        await put_cv(db_client, headers, "A different CV about Rust and Go.")
    ).status_code == 200

    body = (await db_client.get(MATCHES, headers=headers)).json()
    assert body["scoring"]["state"] == "pending"
    assert body["items"] == before  # still the previous scores, flagged as pending


async def test_the_status_row_is_removed_with_the_user(db_session: AsyncSession) -> None:
    user_id = await add_user(db_session)
    await status.request_scoring(db_session, user_id)

    await db_session.execute(delete(User).where(User.id == user_id))

    assert (await db_session.scalars(select(MatchStatus))).all() == []


async def test_with_embeddings_off_the_scores_are_skills_only_and_the_explanation_says_so(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    from tests.integration.test_match_service import make_settings

    headers, user_id = await signed_in(db_client, db_session)
    await add_cv(db_session, user_id, skills=["Python", "Django", "PostgreSQL"], embedded=False)
    near = await add_job(
        db_session,
        user_id,
        body=PYTHON_POST,
        stack=["Python", "Django", "PostgreSQL", "Docker"],
        embedded=False,
    )
    await score_jobs_for_user(db_session, user_id, make_settings(embedding_provider="none"))

    body = (await db_client.get(f"{MATCHES}/{near.id}", headers=headers)).json()

    assert body["score"] == 75  # 3 of the 4 job skills, and nothing else: coverage 0.75 x 100
    reasons = body["reasons"]
    assert reasons["notes"] == ["no_embedding"]
    assert reasons["weights"]["effective"] == {"skills": 1.0, "embedding": 0.0}
    assert "embedding" not in reasons["components"]
    assert body["note_messages"] == ["No comparable embedding: the score is skills only."]
