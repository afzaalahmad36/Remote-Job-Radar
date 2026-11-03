import logging
import uuid
from functools import lru_cache
from typing import Protocol, Self

from redis import Redis
from redis.exceptions import LockError
from redis.lock import Lock

from app.extraction.limits import JOB_LOCK_TTL_SECONDS
from app.tasks.celery_app import (
    HARD_TIME_LIMIT_SECONDS,
    REDIS_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

LOCK_KEY_PREFIX = "lock:ingest:"
JOB_LOCK_KEY_PREFIX = "lock:extract:"
# Longer than the hard time limit: a task that is still running is always killed before its lock
# can expire, so a live run never loses its lock. The extra 30 s is margin for clock/scheduling.
LOCK_TTL_SECONDS = HARD_TIME_LIMIT_SECONDS + 30


def lock_key(source_id: uuid.UUID) -> str:
    return f"{LOCK_KEY_PREFIX}{source_id}"


class SourceLocks(Protocol):
    """One lock per source, so two runs of the same source never overlap.

    Locks are never waited for: a run that cannot get the lock is skipped, not queued.
    """

    def acquire(self, source_id: uuid.UUID) -> bool: ...

    def release(self, source_id: uuid.UUID) -> None: ...

    def is_locked(self, source_id: uuid.UUID) -> bool: ...


class RedisSourceLocks:
    """SourceLocks stored in Redis.

    Each lock is a key set with "only if it does not exist" and an expiry (redis-py's `Lock`),
    holding a random token that identifies its owner.

    Edge cases:
      * A worker killed while holding a lock (OOM, `kill -9`) cannot release it. The expiry
        does: the lock disappears after LOCK_TTL_SECONDS. Until then runs of that source are
        skipped, and the next scheduled run picks it up. The message that was being processed
        is redelivered (acks_late) and skipped the same way if the lock is still held.
      * The token makes release safe: a run whose lock already expired (and was taken by
        someone else) cannot delete the new owner's lock; releasing then only logs a warning.
      * If Redis is unreachable, acquire/release raise and the task fails (and is retried): it
        never runs unprotected.
    """

    _prefix = LOCK_KEY_PREFIX
    _ttl = LOCK_TTL_SECONDS

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._held: dict[uuid.UUID, Lock] = {}  # the locks this object acquired

    @classmethod
    def from_url(
        cls,
        url: str,
        connect_timeout: float = REDIS_CONNECT_TIMEOUT_SECONDS,
        timeout: float = REDIS_SOCKET_TIMEOUT_SECONDS,
    ) -> Self:
        # Timeouts, so a dead or hung Redis cannot hold a worker or an API request forever.
        return cls(
            Redis.from_url(url, socket_connect_timeout=connect_timeout, socket_timeout=timeout)
        )

    def acquire(self, source_id: uuid.UUID) -> bool:
        lock = self._client.lock(self._key(source_id), timeout=self._ttl, blocking=False)
        if not lock.acquire(blocking=False):
            return False
        self._held[source_id] = lock
        return True

    def release(self, source_id: uuid.UUID) -> None:
        lock = self._held.pop(source_id, None)
        if lock is None:
            return
        try:
            lock.release()
        except LockError:  # it expired first: someone else may own it now, leave it alone
            logger.warning("Lock for source %s expired before it was released", source_id)

    def is_locked(self, source_id: uuid.UUID) -> bool:
        return bool(self._client.exists(self._key(source_id)))

    def _key(self, owner_id: uuid.UUID) -> str:
        return f"{self._prefix}{owner_id}"


class RedisJobLocks(RedisSourceLocks):
    """One lock per job, so two runs of one job's extraction do not both ask the model.

    Only an optimisation: the compare-and-set writes keep the data correct even if two runs
    overlap (for example after a lock expired). Same behaviour as the source locks, including the
    expiry that frees the lock of a worker that was killed; the TTL outlives the hard task limit.
    """

    _prefix = JOB_LOCK_KEY_PREFIX
    _ttl = JOB_LOCK_TTL_SECONDS


@lru_cache
def redis_job_locks_for(redis_url: str) -> RedisJobLocks:
    """One job-lock client per process, like `redis_locks_for`."""
    return RedisJobLocks.from_url(redis_url)


@lru_cache
def redis_locks_for(redis_url: str) -> RedisSourceLocks:
    """One lock client (and so one connection pool) per process, shared by every task and
    request. redis-py clients are thread-safe and reconnect after a fork."""
    return RedisSourceLocks.from_url(redis_url)
