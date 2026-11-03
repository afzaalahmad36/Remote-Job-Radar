"""POST /api/v1/jobs/{id}/extract: who may ask, which jobs qualify, the retry limit, and what
happens when the queue is down. The queue is a fake: nothing is published."""

import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import QueueUnavailableError
from app.core.rate_limit import RateLimiter
from app.models import ExtractionMethod, ExtractionStatus, Job, LlmEnrichment, Source
from tests.auth_helpers import auth_headers
from tests.fakes import FakeTaskQueue

JOBS = "/api/v1/jobs"
FAILED = {
    "extraction_status": ExtractionStatus.EXTRACTION_FAILED,
    "llm_enrichment": LlmEnrichment.SKIPPED,
    "extraction_method": ExtractionMethod.RULES,
    "extraction_error": "post too short for the model",
}


@pytest.fixture
async def ann(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "ann@example.com")


@pytest.fixture
async def bob(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "bob@example.com")


async def new_job(
    client: AsyncClient,
    session: AsyncSession,
    headers: dict[str, str],
    **fields: Any,
) -> uuid.UUID:
    """A job in a new source of the user behind `headers`, committed (so a rollback inside the
    endpoint cannot take the setup with it)."""
    created = await client.post(
        "/api/v1/sources",
        json={
            "type": "rss",
            "name": "feed",
            "config": {"url": f"https://example.com/{uuid.uuid4()}"},
        },
        headers=headers,
    )
    assert created.status_code == 201
    job = Job(
        source_id=uuid.UUID(created.json()["id"]),
        external_id="post-1",
        raw_text="Hiring a Marketing Manager",
        **fields,
    )
    session.add(job)
    await session.commit()
    return job.id


async def extract(client: AsyncClient, headers: dict[str, str], job_id: Any) -> Response:
    return await client.post(f"{JOBS}/{job_id}/extract", headers=headers)


async def state(session: AsyncSession, job_id: uuid.UUID) -> Job:
    session.expire_all()
    return (await session.scalars(select(Job).where(Job.id == job_id))).one()


# --- accepting the request ---------------------------------------------------------------------


async def test_a_failed_job_is_reset_and_queued(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 202
    assert response.json() == {"job_id": str(job_id), "status": "queued"}
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    )
    assert (job.extraction_error, job.extraction_method, job.extraction_retries) == (None, None, 1)
    assert task_queue.extraction_enqueued == [(job_id, 2.0)]  # a short delay lets the commit land


async def test_a_job_whose_model_step_was_skipped_can_be_retried_too(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(
        db_client,
        db_session,
        ann,
        extraction_status=ExtractionStatus.EXTRACTED,
        title="Marketing Manager",
        llm_enrichment=LlmEnrichment.SKIPPED,
        extraction_method=ExtractionMethod.RULES,
    )

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 202 and response.json()["status"] == "queued"
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.PENDING,
        LlmEnrichment.NOT_NEEDED,
    )
    assert job.title == "Marketing Manager"  # the rules phase rewrites it; nothing is thrown away
    assert len(task_queue.extraction_enqueued) == 1


@pytest.mark.parametrize(
    ("status", "enrichment"),
    [
        (ExtractionStatus.PENDING, LlmEnrichment.NOT_NEEDED),  # not processed yet
        (ExtractionStatus.EXTRACTED, LlmEnrichment.NOT_NEEDED),
        (ExtractionStatus.EXTRACTED, LlmEnrichment.PENDING),  # waiting for the model already
        (ExtractionStatus.EXTRACTED, LlmEnrichment.DONE),
        (ExtractionStatus.NOT_A_JOB, LlmEnrichment.DONE),
    ],
)
async def test_any_other_job_is_not_retryable_and_nothing_changes(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
    status: ExtractionStatus,
    enrichment: LlmEnrichment,
) -> None:
    job_id = await new_job(
        db_client,
        db_session,
        ann,
        extraction_status=status,
        llm_enrichment=enrichment,
        title="Kept title",
        extraction_method=ExtractionMethod.RULES,
    )

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 409 and response.json()["code"] == "not_retryable"
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.llm_enrichment, job.extraction_retries) == (
        status,
        enrichment,
        0,
    )
    assert task_queue.extraction_enqueued == []


async def test_a_run_in_progress_is_reported_not_restarted_and_costs_no_retry(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)
    task_queue.extracting.add(job_id)

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 202
    assert response.json() == {"job_id": str(job_id), "status": "already_running"}
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.extraction_retries) == (
        ExtractionStatus.EXTRACTION_FAILED,
        0,
    )
    assert task_queue.extraction_enqueued == []


# --- who may ask -------------------------------------------------------------------------------


async def test_another_users_job_looks_exactly_like_a_missing_one(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    bob: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)

    theirs = await extract(db_client, bob, job_id)
    missing = await extract(db_client, bob, uuid.uuid4())

    assert theirs.status_code == missing.status_code == 404
    assert theirs.json() == missing.json() == {"detail": "Job not found", "code": "not_found"}
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.extraction_retries) == (
        ExtractionStatus.EXTRACTION_FAILED,
        0,
    )
    assert task_queue.extraction_enqueued == []


async def test_it_needs_a_login(db_client: AsyncClient) -> None:
    response = await db_client.post(f"{JOBS}/{uuid.uuid4()}/extract")

    assert response.status_code == 401


async def test_a_malformed_id_is_a_validation_error(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.post(f"{JOBS}/not-a-uuid/extract", headers=ann)

    assert response.status_code == 422


# --- the retry limit ---------------------------------------------------------------------------


async def test_five_retries_are_allowed_and_the_sixth_is_refused(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
    app: FastAPI,
) -> None:
    app.state.extract_rate_limiter = RateLimiter(max_attempts=100, window_seconds=60)
    job_id = await new_job(db_client, db_session, ann, **FAILED)

    for expected in range(1, 6):
        assert (await extract(db_client, ann, job_id)).status_code == 202
        assert (await state(db_session, job_id)).extraction_retries == expected
        await db_session.execute(  # the job fails again
            update(Job).where(Job.id == job_id).values(**FAILED)
        )
        await db_session.commit()

    sixth = await extract(db_client, ann, job_id)

    assert sixth.status_code == 409 and sixth.json()["code"] == "retries_exhausted"
    assert (await state(db_session, job_id)).extraction_retries == 5
    assert len(task_queue.extraction_enqueued) == 5


async def test_a_job_already_at_the_limit_is_refused_at_once(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, extraction_retries=5, **FAILED)

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 409 and response.json()["code"] == "retries_exhausted"
    assert task_queue.extraction_enqueued == []


async def test_the_limit_is_counted_in_extraction_retries_not_extraction_attempts(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    """The system's own re-runs raise attempts a lot; they must not use up the user's retries."""
    job_id = await new_job(db_client, db_session, ann, extraction_attempts=40, **FAILED)

    assert (await extract(db_client, ann, job_id)).status_code == 202

    job = await state(db_session, job_id)
    assert (job.extraction_attempts, job.extraction_retries) == (40, 1)


# --- the queue is down: the reset is rolled back -----------------------------------------------


async def test_a_failed_enqueue_rolls_the_reset_back_and_uses_no_retry(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)
    task_queue.fail_extraction_enqueue_with = QueueUnavailableError("The queue is unavailable")

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 503 and response.json()["code"] == "queue_unavailable"
    job = await state(db_session, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTION_FAILED,
        LlmEnrichment.SKIPPED,
    )
    assert (job.extraction_retries, job.extraction_method) == (0, ExtractionMethod.RULES)
    assert job.extraction_error == "post too short for the model"
    assert task_queue.extraction_enqueued == []

    task_queue.fail_extraction_enqueue_with = None  # the queue comes back; the user tries again
    assert (await extract(db_client, ann, job_id)).status_code == 202
    assert (await state(db_session, job_id)).extraction_retries == 1  # one retry, not two


async def test_an_unexpected_queue_error_also_rolls_back(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
    app: FastAPI,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)
    task_queue.fail_extraction_enqueue_with = RuntimeError("something odd")
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(f"{JOBS}/{job_id}/extract", headers=ann)

    assert response.status_code == 500
    assert (await state(db_session, job_id)).extraction_retries == 0


async def test_checking_for_a_run_in_progress_fails_with_503_when_the_queue_is_down(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    job_id = await new_job(db_client, db_session, ann, **FAILED)
    task_queue.fail_with = QueueUnavailableError("The queue is unavailable")

    response = await extract(db_client, ann, job_id)

    assert response.status_code == 503
    assert (await state(db_session, job_id)).extraction_retries == 0


# --- rate limit, in its own bucket -------------------------------------------------------------


async def test_retries_are_rate_limited_per_user_in_their_own_bucket(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    bob: dict[str, str],
    app: FastAPI,
) -> None:
    app.state.extract_rate_limiter = RateLimiter(max_attempts=2, window_seconds=60)
    ann_job = await new_job(db_client, db_session, ann, **FAILED)
    bob_job = await new_job(db_client, db_session, bob, **FAILED)

    first = await extract(db_client, ann, ann_job)
    second = await extract(db_client, ann, ann_job)
    third = await extract(db_client, ann, ann_job)
    bobs = await extract(db_client, bob, bob_job)

    assert (first.status_code, second.status_code) == (202, 409)  # the second is not retryable now
    assert third.status_code == 429 and third.json()["code"] == "rate_limited"
    assert "Retry-After" in third.headers
    assert bobs.status_code == 202  # another user's bucket is untouched


async def test_syncs_and_extraction_retries_do_not_share_a_bucket(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    app: FastAPI,
    task_queue: FakeTaskQueue,
) -> None:
    app.state.sync_rate_limiter = RateLimiter(max_attempts=1, window_seconds=60)
    app.state.extract_rate_limiter = RateLimiter(max_attempts=1, window_seconds=60)
    job_id = await new_job(db_client, db_session, ann, **FAILED)
    source_id = (await db_session.scalars(select(Source.id))).first()
    assert source_id is not None

    sync_ok = await db_client.post(f"/api/v1/sources/{source_id}/sync", headers=ann)
    extract_ok = await extract(db_client, ann, job_id)
    sync_blocked = await db_client.post(f"/api/v1/sources/{source_id}/sync", headers=ann)

    assert (sync_ok.status_code, extract_ok.status_code, sync_blocked.status_code) == (
        202,
        202,
        429,
    )
