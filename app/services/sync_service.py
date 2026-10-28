import uuid
from enum import StrEnum
from typing import Protocol


class TaskQueue(Protocol):
    """What this service needs from the background-job system (a `Protocol` is an interface:
    anything with these two methods qualifies, which also lets tests pass a fake)."""

    async def is_syncing(self, source_id: uuid.UUID) -> bool: ...

    async def enqueue_sync(self, source_id: uuid.UUID) -> None: ...


class SyncRequestStatus(StrEnum):
    QUEUED = "queued"
    ALREADY_RUNNING = "already_running"


async def request_sync(queue: TaskQueue, source_id: uuid.UUID) -> SyncRequestStatus:
    """Ask for a sync of the source, unless one is already running.

    "Already running" is an answer, not an error: the caller wanted fresh data and a sync is
    already producing it. The check is best-effort: two requests can both see "not running" and
    both queue a task, but the per-source lock makes the second task skip itself.
    """
    if await queue.is_syncing(source_id):
        return SyncRequestStatus.ALREADY_RUNNING
    await queue.enqueue_sync(source_id)
    return SyncRequestStatus.QUEUED
