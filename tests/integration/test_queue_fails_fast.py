"""Proof that a Redis outage turns into a quick 503 and not a hanging request (or a hung CI job).

No Redis is needed or used: the "broker" is a port nothing listens on, or a local server that
never answers. The time limits below are generous (they must hold on slow CI machines) but are
orders of magnitude below "retry for hours".
"""

import threading
import time
import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api_deps import get_task_queue
from app.core.exceptions import QueueUnavailableError
from app.tasks.celery_app import QUEUE_CALL_TIMEOUT_SECONDS
from app.tasks.locking import RedisSourceLocks
from app.tasks.queue import CeleryTaskQueue
from tests.auth_helpers import auth_headers
from tests.fakes import UNREACHABLE_REDIS_URL, FakeSourceLocks

SOURCE = uuid.UUID("11111111-2222-3333-4444-555555555555")
FAST_ENOUGH_SECONDS = 15  # a hang is hours; a refused connection is milliseconds to a few seconds
BROKER_WORDS = ("redis", "6399", "kombu", "celery", "127.0.0.1", "connection", "errno", "refused")


def real_queue_on_a_dead_redis(call_timeout: float = QUEUE_CALL_TIMEOUT_SECONDS) -> CeleryTaskQueue:
    locks = RedisSourceLocks.from_url(UNREACHABLE_REDIS_URL, connect_timeout=0.5, timeout=0.5)
    return CeleryTaskQueue(locks, call_timeout=call_timeout)


async def test_checking_for_a_running_sync_fails_fast_when_redis_is_unreachable() -> None:
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await real_queue_on_a_dead_redis().is_syncing(SOURCE)

    assert time.perf_counter() - started < FAST_ENOUGH_SECONDS


async def test_publishing_fails_fast_when_the_broker_is_unreachable(celery_env: None) -> None:
    # The REAL Celery publish path (celery_env points the broker at the dead port): connection
    # timeout plus the bounded publish retry policy, then a clean error.
    queue = CeleryTaskQueue(FakeSourceLocks(), call_timeout=FAST_ENOUGH_SECONDS)
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError) as raised:
        await queue.enqueue_sync(SOURCE)

    assert time.perf_counter() - started < FAST_ENOUGH_SECONDS
    assert raised.value.detail == "The sync queue is unavailable"


async def test_a_call_that_hangs_is_cut_off_by_the_overall_deadline() -> None:
    release = threading.Event()

    class HangingLocks(FakeSourceLocks):
        def is_locked(self, source_id: uuid.UUID) -> bool:
            release.wait(timeout=2)  # like a DNS lookup that has not failed yet
            return False

    queue = CeleryTaskQueue(HangingLocks(), call_timeout=0.2)
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue.is_syncing(SOURCE)

    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 1.5  # answered at the 0.2 s deadline, not after the 2 s the call needs


async def test_a_hanging_publish_is_cut_off_by_the_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    class HangingTask:
        def apply_async(self, **kwargs: object) -> None:
            release.wait(timeout=2)

    monkeypatch.setattr("app.tasks.queue.ingest_source", HangingTask())
    queue = CeleryTaskQueue(FakeSourceLocks(), call_timeout=0.2)
    started = time.perf_counter()

    with pytest.raises(QueueUnavailableError):
        await queue.enqueue_sync(SOURCE)

    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 1.5


async def test_the_sync_endpoint_answers_503_within_seconds_when_redis_is_down(
    app: FastAPI, db_client: AsyncClient, celery_env: None
) -> None:
    app.dependency_overrides[get_task_queue] = lambda: real_queue_on_a_dead_redis()
    headers = await auth_headers(db_client, "ann@example.com")
    created = await db_client.post(
        "/api/v1/sources",
        json={"type": "rss", "name": "feed", "config": {"url": "https://example.com/feed.xml"}},
        headers=headers,
    )
    source_id = created.json()["id"]
    started = time.perf_counter()

    response = await db_client.post(f"/api/v1/sources/{source_id}/sync", headers=headers)

    assert time.perf_counter() - started < FAST_ENOUGH_SECONDS
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The sync queue is unavailable",
        "code": "queue_unavailable",
    }
    body = response.text.lower()
    assert not any(word in body for word in BROKER_WORDS)  # nothing about the broker leaks
