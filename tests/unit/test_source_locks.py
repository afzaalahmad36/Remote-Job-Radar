import logging
import socket
import time
import uuid

import pytest
from redis.exceptions import LockNotOwnedError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.tasks.celery_app import (
    HARD_TIME_LIMIT_SECONDS,
    REDIS_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_TIMEOUT_SECONDS,
)
from app.tasks.locking import (
    LOCK_KEY_PREFIX,
    LOCK_TTL_SECONDS,
    RedisSourceLocks,
    lock_key,
    redis_locks_for,
)
from tests.fakes import UNREACHABLE_REDIS_URL


class FakeRedisLock:
    """Just enough of redis-py's Lock: set-if-absent with an owner token, safe release."""

    def __init__(self, store: dict[str, str], name: str, timeout: float, blocking: bool) -> None:
        self.store, self.name, self.timeout, self.blocking = store, name, timeout, blocking
        self.token = uuid.uuid4().hex

    def acquire(self, blocking: bool | None = None) -> bool:
        if self.name in self.store:
            return False
        self.store[self.name] = self.token
        return True

    def release(self) -> None:
        if self.store.get(self.name) != self.token:
            raise LockNotOwnedError("Cannot release a lock that's no longer owned")
        del self.store[self.name]


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.locks: list[FakeRedisLock] = []

    def lock(self, name: str, timeout: float, blocking: bool = True) -> FakeRedisLock:
        lock = FakeRedisLock(self.store, name, timeout, blocking)
        self.locks.append(lock)
        return lock

    def exists(self, key: str) -> int:
        return int(key in self.store)


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def locks(redis: FakeRedis) -> RedisSourceLocks:
    return RedisSourceLocks(redis)  # type: ignore[arg-type]


SOURCE = uuid.UUID("11111111-2222-3333-4444-555555555555")


def test_the_key_is_per_source() -> None:
    assert lock_key(SOURCE) == f"{LOCK_KEY_PREFIX}{SOURCE}" == f"lock:ingest:{SOURCE}"
    assert lock_key(SOURCE) != lock_key(uuid.uuid4())


def test_the_first_acquire_wins_and_the_second_is_refused(locks: RedisSourceLocks) -> None:
    other_process = RedisSourceLocks(locks._client)  # the same Redis, a different holder

    assert locks.acquire(SOURCE) is True
    assert other_process.acquire(SOURCE) is False


def test_different_sources_do_not_block_each_other(locks: RedisSourceLocks) -> None:
    assert locks.acquire(SOURCE) is True
    assert locks.acquire(uuid.uuid4()) is True


def test_the_lock_is_never_waited_for_and_has_an_expiry(
    locks: RedisSourceLocks, redis: FakeRedis
) -> None:
    locks.acquire(SOURCE)

    lock = redis.locks[0]
    assert lock.blocking is False  # skip, don't queue
    assert lock.timeout == LOCK_TTL_SECONDS  # a dead worker's lock still disappears


def test_the_expiry_outlives_the_longest_possible_run() -> None:
    # A task is killed at the hard time limit, so a running task can never outlive its lock.
    assert LOCK_TTL_SECONDS > HARD_TIME_LIMIT_SECONDS


def test_release_frees_the_lock_for_the_next_run(locks: RedisSourceLocks) -> None:
    locks.acquire(SOURCE)

    locks.release(SOURCE)

    assert locks.is_locked(SOURCE) is False
    assert locks.acquire(SOURCE) is True


def test_is_locked_reflects_the_state_in_redis(locks: RedisSourceLocks) -> None:
    assert locks.is_locked(SOURCE) is False
    locks.acquire(SOURCE)
    assert locks.is_locked(SOURCE) is True


def test_releasing_a_lock_we_never_held_does_nothing(
    locks: RedisSourceLocks, redis: FakeRedis
) -> None:
    redis.store[lock_key(SOURCE)] = "someone-else"

    locks.release(SOURCE)

    assert locks.is_locked(SOURCE) is True  # we did not touch the other holder's lock


def test_a_lock_that_expired_is_not_released_over_its_new_owner(
    locks: RedisSourceLocks, redis: FakeRedis, caplog: pytest.LogCaptureFixture
) -> None:
    locks.acquire(SOURCE)
    del redis.store[lock_key(SOURCE)]  # our lock expired...
    new_owner = RedisSourceLocks(redis)  # type: ignore[arg-type]
    assert new_owner.acquire(SOURCE)  # ...and another run took the source

    with caplog.at_level(logging.WARNING):
        locks.release(SOURCE)  # must neither raise nor free the new owner's lock

    assert locks.is_locked(SOURCE) is True
    assert "expired before it was released" in caplog.text


def test_the_real_client_has_short_connect_and_socket_timeouts() -> None:
    # So a dead or hung Redis cannot hold a worker or an API request for long.
    client = RedisSourceLocks.from_url(UNREACHABLE_REDIS_URL)._client

    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_connect_timeout"] == REDIS_CONNECT_TIMEOUT_SECONDS == 2.0
    assert kwargs["socket_timeout"] == REDIS_SOCKET_TIMEOUT_SECONDS == 5.0


def test_the_timeouts_can_be_overridden_per_client() -> None:
    client = RedisSourceLocks.from_url(
        UNREACHABLE_REDIS_URL, connect_timeout=0.3, timeout=0.4
    )._client

    kwargs = client.connection_pool.connection_kwargs
    assert (kwargs["socket_connect_timeout"], kwargs["socket_timeout"]) == (0.3, 0.4)


def test_a_redis_that_accepts_but_never_answers_fails_within_the_socket_timeout() -> None:
    with socket.socket() as silent_server:  # accepts connections (backlog), never replies
        silent_server.bind(("127.0.0.1", 0))
        silent_server.listen(5)
        url = f"redis://127.0.0.1:{silent_server.getsockname()[1]}/0"
        locks = RedisSourceLocks.from_url(url, connect_timeout=0.3, timeout=0.3)
        started = time.perf_counter()

        with pytest.raises(RedisTimeoutError):
            locks.is_locked(SOURCE)

    assert time.perf_counter() - started < 3  # about 0.3 s, not "until Redis answers"


def test_one_lock_client_is_shared_per_redis_url() -> None:
    first = redis_locks_for(UNREACHABLE_REDIS_URL)
    second = redis_locks_for(UNREACHABLE_REDIS_URL)
    other = redis_locks_for(UNREACHABLE_REDIS_URL.replace("/0", "/1"))

    assert first is second
    assert first is not other
