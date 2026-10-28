import threading
import uuid
from typing import Any

import pytest
from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api_deps import get_task_queue
from app.core.config import Settings
from app.core.exceptions import QueueUnavailableError
from app.services.sync_service import SyncRequestStatus, request_sync
from app.tasks.queue import CeleryTaskQueue
from tests.fakes import FakeSourceLocks, FakeTaskQueue

SOURCE = uuid.UUID("11111111-2222-3333-4444-555555555555")


class BrokenLocks(FakeSourceLocks):
    def is_locked(self, source_id: uuid.UUID) -> bool:
        raise RedisConnectionError("cannot connect to redis-internal:6379")


class FakeIngestTask:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[dict[str, Any], int]] = []

    def apply_async(self, **kwargs: Any) -> None:
        self.calls.append((kwargs, threading.get_ident()))
        if self.error:
            raise self.error


# --- the real queue adapter ----------------------------------------------------------------------


async def test_is_syncing_reads_the_lock() -> None:
    locks = FakeSourceLocks()
    queue = CeleryTaskQueue(locks)

    assert await queue.is_syncing(SOURCE) is False
    locks.held.add(SOURCE)
    assert await queue.is_syncing(SOURCE) is True


async def test_enqueue_publishes_the_ingest_task_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = FakeIngestTask()
    monkeypatch.setattr("app.tasks.queue.ingest_source", task)

    await CeleryTaskQueue(FakeSourceLocks()).enqueue_sync(SOURCE)

    ((kwargs, thread_id),) = task.calls
    assert kwargs == {"args": [str(SOURCE)]}  # the id as a string: JSON-serializable
    assert thread_id != threading.get_ident()  # blocking broker I/O ran in a worker thread


@pytest.mark.parametrize(
    "error",
    [KombuOperationalError("connection to redis-internal:6379 refused"), RedisConnectionError("x")],
)
async def test_a_broker_failure_becomes_a_generic_queue_unavailable_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr("app.tasks.queue.ingest_source", FakeIngestTask(error))

    with pytest.raises(QueueUnavailableError) as raised:
        await CeleryTaskQueue(FakeSourceLocks()).enqueue_sync(SOURCE)

    assert raised.value.detail == "The sync queue is unavailable"  # nothing about hosts or ports
    assert "redis-internal" not in raised.value.detail


async def test_a_redis_failure_while_checking_the_lock_is_also_queue_unavailable() -> None:
    with pytest.raises(QueueUnavailableError) as raised:
        await CeleryTaskQueue(BrokenLocks()).is_syncing(SOURCE)

    assert "redis-internal" not in raised.value.detail


# --- the sync request logic ----------------------------------------------------------------------


async def test_a_request_queues_a_sync_when_none_is_running() -> None:
    queue = FakeTaskQueue()

    assert await request_sync(queue, SOURCE) is SyncRequestStatus.QUEUED
    assert queue.enqueued == [SOURCE]


async def test_a_request_is_answered_with_already_running_and_queues_nothing() -> None:
    queue = FakeTaskQueue()
    queue.syncing.add(SOURCE)

    assert await request_sync(queue, SOURCE) is SyncRequestStatus.ALREADY_RUNNING
    assert queue.enqueued == []


def test_the_default_queue_is_the_celery_one_on_the_settings_redis(settings: Settings) -> None:
    queue = get_task_queue(settings)  # tests normally replace this; building it opens no connection

    assert isinstance(queue, CeleryTaskQueue)


def test_the_statuses_are_the_documented_strings() -> None:
    assert [s.value for s in SyncRequestStatus] == ["queued", "already_running"]
