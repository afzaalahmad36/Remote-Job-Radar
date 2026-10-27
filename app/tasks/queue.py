import asyncio
import uuid

from kombu.exceptions import KombuError
from redis.exceptions import RedisError

from app.core.exceptions import QueueUnavailableError
from app.tasks.jobs import ingest_source
from app.tasks.locking import SourceLocks


class CeleryTaskQueue:
    """The API's view of the background jobs, backed by Celery and Redis.

    Satisfies `services.sync_service.TaskQueue`. Both redis-py and Celery's publisher are
    blocking libraries, and the API is async: calling them directly would freeze the event loop
    (every other request) while they wait on the network, so `asyncio.to_thread` runs each call
    in a worker thread instead.
    """

    def __init__(self, locks: SourceLocks) -> None:
        self._locks = locks

    async def is_syncing(self, source_id: uuid.UUID) -> bool:
        try:
            return await asyncio.to_thread(self._locks.is_locked, source_id)
        except RedisError as error:
            raise QueueUnavailableError("The sync queue is unavailable") from error

    async def enqueue_sync(self, source_id: uuid.UUID) -> None:
        try:
            await asyncio.to_thread(ingest_source.apply_async, args=[str(source_id)])
        except (KombuError, RedisError) as error:  # the broker is down or unreachable
            raise QueueUnavailableError("The sync queue is unavailable") from error
