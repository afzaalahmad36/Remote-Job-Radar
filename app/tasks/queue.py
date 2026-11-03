import asyncio
import uuid

from kombu.exceptions import KombuError
from redis.exceptions import RedisError

from app.core.exceptions import QueueUnavailableError
from app.tasks.celery_app import QUEUE_CALL_TIMEOUT_SECONDS
from app.tasks.jobs import extract_job, ingest_source
from app.tasks.locking import SourceLocks


class CeleryTaskQueue:
    """The API's view of the background jobs, backed by Celery and Redis.

    Satisfies `services.sync_service.TaskQueue`. Both redis-py and Celery's publisher are
    blocking libraries, and the API is async: calling them directly would freeze the event loop
    (every other request) while they wait on the network, so `asyncio.to_thread` runs each call
    in a worker thread instead.

    Every operation also has an overall deadline (`call_timeout`). The client timeouts bound the
    socket operations, but not everything (a slow DNS lookup is not covered), and a request must
    not hang while Redis is down: past the deadline we answer "unavailable". The abandoned
    thread finishes (or fails) on its own and its result is ignored.
    """

    def __init__(
        self,
        locks: SourceLocks,
        call_timeout: float = QUEUE_CALL_TIMEOUT_SECONDS,
        job_locks: SourceLocks | None = None,
    ) -> None:
        self._locks = locks
        self._job_locks = job_locks  # only needed for the extraction methods below
        self._call_timeout = call_timeout

    async def is_syncing(self, source_id: uuid.UUID) -> bool:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._locks.is_locked, source_id), self._call_timeout
            )
        except (RedisError, TimeoutError) as error:
            raise QueueUnavailableError("The sync queue is unavailable") from error

    async def enqueue_sync(self, source_id: uuid.UUID) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(ingest_source.apply_async, args=[str(source_id)]),
                self._call_timeout,
            )
        except (KombuError, RedisError, TimeoutError) as error:  # the broker is unreachable
            raise QueueUnavailableError("The sync queue is unavailable") from error

    async def is_extracting(self, job_id: uuid.UUID) -> bool:
        assert self._job_locks is not None, "this queue was built without job locks"
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._job_locks.is_locked, job_id), self._call_timeout
            )
        except (RedisError, TimeoutError) as error:
            raise QueueUnavailableError("The extraction queue is unavailable") from error

    async def enqueue_extraction(self, job_id: uuid.UUID, countdown: float = 0.0) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(extract_job.apply_async, args=[str(job_id)], countdown=countdown),
                self._call_timeout,
            )
        except (KombuError, RedisError, TimeoutError) as error:
            raise QueueUnavailableError("The extraction queue is unavailable") from error
