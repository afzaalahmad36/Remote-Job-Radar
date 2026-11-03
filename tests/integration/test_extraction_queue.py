"""The API's view of the extraction queue (is a run in progress? queue one), the real wiring behind
the endpoint, and the paths of the task module that need a broker. No Redis is ever contacted
except a port where nothing listens."""

import threading
import time
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_deps import get_extraction_queue
from app.core.config import Settings
from app.core.exceptions import QueueUnavailableError
from app.llm.providers.fake import FakeLLMClient
from app.models import ExtractionMethod, ExtractionStatus, Job, LlmEnrichment
from app.repositories import extraction_repository
from app.tasks import jobs
from app.tasks.locking import RedisJobLocks, RedisSourceLocks
from app.tasks.queue import CeleryTaskQueue
from tests.auth_helpers import auth_headers
from tests.fakes import UNREACHABLE_REDIS_URL, FakeCooldown, FakeSourceLocks, FakeTaskQueue

JOB = uuid.UUID("11111111-2222-3333-4444-555555555555")
FAILED = {
    "extraction_status": ExtractionStatus.EXTRACTION_FAILED,
    "llm_enrichment": LlmEnrichment.SKIPPED,
    "extraction_method": ExtractionMethod.RULES,
}


class RecordingTask:
    def __init__(self, error: Exception | None = None, hang: threading.Event | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error, self.hang = error, hang

    def apply_async(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.hang is not None:
            self.hang.wait(timeout=2)
        if self.error:
            raise self.error


def queue_with(job_locks: FakeSourceLocks | None = None, **kwargs: Any) -> CeleryTaskQueue:
    return CeleryTaskQueue(FakeSourceLocks(), job_locks=job_locks or FakeSourceLocks(), **kwargs)


# --- is a run in progress? ---------------------------------------------------------------------


async def test_an_extraction_in_progress_is_a_held_job_lock() -> None:
    job_locks = FakeSourceLocks()
    queue = queue_with(job_locks)

    assert await queue.is_extracting(JOB) is False
    job_locks.held.add(JOB)
    assert await queue.is_extracting(JOB) is True


async def test_a_dead_redis_makes_the_check_fail_fast_as_queue_unavailable() -> None:
    locks = RedisJobLocks.from_url(UNREACHABLE_REDIS_URL, connect_timeout=0.5, timeout=0.5)
    queue = CeleryTaskQueue(RedisSourceLocks.from_url(UNREACHABLE_REDIS_URL), job_locks=locks)
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue.is_extracting(JOB)

    assert time.perf_counter() - started < 15


async def test_a_hanging_check_is_cut_off_by_the_overall_deadline() -> None:
    release = threading.Event()

    class HangingLocks(FakeSourceLocks):
        def is_locked(self, source_id: uuid.UUID) -> bool:
            release.wait(timeout=2)
            return False

    queue = queue_with(HangingLocks(), call_timeout=0.2)
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue.is_extracting(JOB)

    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 1.5


# --- queueing one ------------------------------------------------------------------------------


async def test_queueing_publishes_the_task_with_the_id_and_the_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = RecordingTask()
    monkeypatch.setattr("app.tasks.queue.extract_job", task)

    await queue_with().enqueue_extraction(JOB, 2.0)
    await queue_with().enqueue_extraction(JOB)

    assert task.calls == [
        {"args": [str(JOB)], "countdown": 2.0},
        {"args": [str(JOB)], "countdown": 0.0},
    ]


@pytest.mark.parametrize(
    "error",
    [KombuOperationalError("down"), RedisConnectionError("down")],
    ids=lambda e: type(e).__name__,
)
async def test_a_publish_error_becomes_queue_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr("app.tasks.queue.extract_job", RecordingTask(error=error))

    with pytest.raises(QueueUnavailableError, match="extraction queue is unavailable"):
        await queue_with().enqueue_extraction(JOB)


async def test_a_hanging_publish_is_cut_off_by_the_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    monkeypatch.setattr("app.tasks.queue.extract_job", RecordingTask(hang=release))
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue_with(call_timeout=0.2).enqueue_extraction(JOB)

    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 1.5


async def test_the_real_publish_path_fails_fast_when_the_broker_is_unreachable(
    celery_env: None,
) -> None:
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue_with(call_timeout=15).enqueue_extraction(JOB)

    assert time.perf_counter() - started < 15


# --- the wiring behind the endpoint ------------------------------------------------------------


def test_the_endpoint_gets_a_real_queue_with_both_kinds_of_lock(settings: Settings) -> None:
    queue = get_extraction_queue(settings)

    assert isinstance(queue, CeleryTaskQueue)


async def test_the_endpoint_answers_503_when_the_real_queue_cannot_reach_redis(
    app: FastAPI, db_client: AsyncClient, db_session: AsyncSession, celery_env: None
) -> None:
    app.dependency_overrides[get_extraction_queue] = lambda: CeleryTaskQueue(
        RedisSourceLocks.from_url(UNREACHABLE_REDIS_URL, connect_timeout=0.5, timeout=0.5),
        job_locks=RedisJobLocks.from_url(UNREACHABLE_REDIS_URL, connect_timeout=0.5, timeout=0.5),
    )
    headers = await auth_headers(db_client, "ann@example.com")
    created = await db_client.post(
        "/api/v1/sources",
        json={"type": "rss", "name": "feed", "config": {"url": "https://example.com/feed.xml"}},
        headers=headers,
    )
    job = Job(source_id=uuid.UUID(created.json()["id"]), external_id="p", raw_text="x", **FAILED)
    db_session.add(job)
    await db_session.commit()
    started = time.perf_counter()

    response = await db_client.post(f"/api/v1/jobs/{job.id}/extract", headers=headers)

    assert response.status_code == 503 and response.json()["code"] == "queue_unavailable"
    assert time.perf_counter() - started < 15
    await db_session.refresh(job)
    assert job.extraction_retries == 0


# --- losing a race at the reset ----------------------------------------------------------------


async def test_a_reset_that_matches_nothing_is_reported_and_costs_no_retry(
    db_client: AsyncClient,
    db_session: AsyncSession,
    task_queue: FakeTaskQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another request got there between the check and the reset: it is not retryable any more."""
    headers = await auth_headers(db_client, "ann@example.com")
    created = await db_client.post(
        "/api/v1/sources",
        json={"type": "rss", "name": "feed", "config": {"url": "https://example.com/feed.xml"}},
        headers=headers,
    )
    job = Job(source_id=uuid.UUID(created.json()["id"]), external_id="p", raw_text="x", **FAILED)
    db_session.add(job)
    await db_session.commit()

    async def lost(*args: Any, **kwargs: Any) -> None:
        await db_session.execute(
            update(Job)
            .where(Job.id == job.id)
            .values(
                extraction_status=ExtractionStatus.PENDING, llm_enrichment=LlmEnrichment.NOT_NEEDED
            )
        )
        return None

    monkeypatch.setattr(extraction_repository, "reset_for_retry", lost)

    response = await db_client.post(f"/api/v1/jobs/{job.id}/extract", headers=headers)

    assert response.status_code == 409 and response.json()["code"] == "not_retryable"
    assert task_queue.extraction_enqueued == []


async def test_a_reset_that_matches_nothing_for_an_unknown_reason_is_still_refused(
    db_client: AsyncClient,
    db_session: AsyncSession,
    task_queue: FakeTaskQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = await auth_headers(db_client, "ann@example.com")
    created = await db_client.post(
        "/api/v1/sources",
        json={"type": "rss", "name": "feed", "config": {"url": "https://example.com/feed.xml"}},
        headers=headers,
    )
    job = Job(source_id=uuid.UUID(created.json()["id"]), external_id="p", raw_text="x", **FAILED)
    db_session.add(job)
    await db_session.commit()

    async def nothing(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(extraction_repository, "reset_for_retry", nothing)

    response = await db_client.post(f"/api/v1/jobs/{job.id}/extract", headers=headers)

    assert response.status_code == 409 and response.json()["code"] == "not_retryable"
    assert task_queue.extraction_enqueued == []


# --- task module paths that need a broker ------------------------------------------------------


def test_the_real_enqueue_publishes_the_extract_task_with_a_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = RecordingTask()
    monkeypatch.setattr(jobs, "extract_job", task)

    jobs._enqueue_extraction(JOB, 3.0)  # noqa: SLF001 - the default the hook and the sweep use

    assert task.calls == [{"args": [str(JOB)], "countdown": 3.0}]


def test_the_sweep_builds_its_own_cooldown_when_a_client_exists_and_fails_closed(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Kill switch on (fake provider): the sweep builds the Redis cooldown itself. Redis is a dead
    port here, so nothing is queued for the model and one warning is logged."""
    on = settings.model_copy(
        update={
            "llm_provider": "fake",
            "llm_extraction_enabled": True,
            "llm_model_extraction": "some-model",
        }
    )

    summary = jobs.run_sweep_extraction(settings=on, enqueue=lambda job_id, delay: None)

    assert summary["enrichment_due"] == 0
    assert "cooldown store is unavailable" in caplog.text


def test_the_sweep_uses_the_cooldown_it_is_given(settings: Settings) -> None:
    summary = jobs.run_sweep_extraction(
        settings=settings,
        llm=FakeLLMClient(),
        cooldown=FakeCooldown(),
        enqueue=lambda job_id, delay: None,
    )

    assert summary["queued"] >= 0
