import uuid

import pytest
from redis.exceptions import LockNotOwnedError

from app.extraction.limits import EXTRACTION_HARD_TIME_LIMIT_SECONDS, JOB_LOCK_TTL_SECONDS
from app.tasks.locking import (
    JOB_LOCK_KEY_PREFIX,
    LOCK_KEY_PREFIX,
    LOCK_TTL_SECONDS,
    RedisJobLocks,
    RedisSourceLocks,
    redis_job_locks_for,
)
from tests.fakes import UNREACHABLE_REDIS_URL

JOB = uuid.UUID("aaaaaaaa-2222-3333-4444-555555555555")


class FakeRedisLock:
    def __init__(self, store: dict[str, str], name: str, timeout: float) -> None:
        self.store, self.name, self.timeout = store, name, timeout
        self.token = uuid.uuid4().hex

    def acquire(self, blocking: bool | None = None) -> bool:
        if self.name in self.store:
            return False
        self.store[self.name] = self.token
        return True

    def release(self) -> None:
        if self.store.get(self.name) != self.token:
            raise LockNotOwnedError("no longer owned")
        del self.store[self.name]


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.locks: list[FakeRedisLock] = []

    def lock(self, name: str, timeout: float, blocking: bool = True) -> FakeRedisLock:
        lock = FakeRedisLock(self.store, name, timeout)
        self.locks.append(lock)
        return lock

    def exists(self, key: str) -> int:
        return int(key in self.store)


def job_locks(redis: FakeRedis) -> RedisJobLocks:
    return RedisJobLocks(redis)  # type: ignore[arg-type]


def test_the_job_lock_has_its_own_key_and_a_120_second_expiry() -> None:
    redis = FakeRedis()

    assert job_locks(redis).acquire(JOB)

    assert list(redis.store) == [f"{JOB_LOCK_KEY_PREFIX}{JOB}"] == [f"lock:extract:{JOB}"]
    assert redis.locks[0].timeout == JOB_LOCK_TTL_SECONDS == 120


def test_the_lock_outlives_the_longest_run_a_task_is_allowed() -> None:
    assert JOB_LOCK_TTL_SECONDS > EXTRACTION_HARD_TIME_LIMIT_SECONDS


def test_a_job_lock_and_a_source_lock_with_the_same_id_never_collide() -> None:
    redis = FakeRedis()
    sources = RedisSourceLocks(redis)  # type: ignore[arg-type]

    assert sources.acquire(JOB) and job_locks(redis).acquire(JOB)

    assert sorted(redis.store) == sorted([f"{LOCK_KEY_PREFIX}{JOB}", f"{JOB_LOCK_KEY_PREFIX}{JOB}"])
    assert redis.locks[0].timeout == LOCK_TTL_SECONDS  # the source lock keeps its own expiry


def test_the_first_acquire_wins_and_is_never_waited_for() -> None:
    redis = FakeRedis()
    first, other_process = job_locks(redis), job_locks(redis)

    assert first.acquire(JOB) is True
    assert other_process.acquire(JOB) is False  # skipped, not queued
    assert other_process.is_locked(JOB) is True


def test_release_frees_the_lock_and_a_later_run_can_take_it() -> None:
    redis = FakeRedis()
    locks = job_locks(redis)
    locks.acquire(JOB)

    locks.release(JOB)

    assert locks.is_locked(JOB) is False
    assert job_locks(redis).acquire(JOB) is True


def test_a_run_whose_lock_expired_cannot_delete_the_new_owners_lock() -> None:
    redis = FakeRedis()
    old_run = job_locks(redis)
    old_run.acquire(JOB)
    redis.store.clear()  # the old run's lock expired...
    new_run = job_locks(redis)
    assert new_run.acquire(JOB)  # ...and another run took the job

    old_run.release(JOB)  # only logs a warning

    assert new_run.is_locked(JOB) is True


def test_releasing_a_lock_that_was_never_taken_does_nothing() -> None:
    job_locks(FakeRedis()).release(JOB)


def test_one_lock_client_is_shared_per_url() -> None:
    redis_job_locks_for.cache_clear()

    first = redis_job_locks_for(UNREACHABLE_REDIS_URL)

    assert isinstance(first, RedisJobLocks)
    assert redis_job_locks_for(UNREACHABLE_REDIS_URL) is first
    redis_job_locks_for.cache_clear()


@pytest.mark.parametrize("cls", [RedisSourceLocks, RedisJobLocks])
def test_both_kinds_build_from_a_url_without_connecting(cls: type[RedisSourceLocks]) -> None:
    assert isinstance(cls.from_url(UNREACHABLE_REDIS_URL), cls)
