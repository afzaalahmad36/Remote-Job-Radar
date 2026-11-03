"""What makes Celery retry the extraction task, and what never does.

The retry comes from `autoretry_for=RETRYABLE_ERRORS` on the `extract_job` task in
`app/tasks/jobs.py` (database and Redis errors only). Celery wraps the task's `run` so that one of
those exceptions calls `task.retry(...)`. These tests run the real task through that wrapper and
spy on `retry`: a provider outcome must come back as a plain result, so the wrapper never sees an
exception and never calls `retry`.
"""

import asyncio
import uuid
from typing import Any

import pytest
from celery.exceptions import Retry
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import InterfaceError, OperationalError

from app.core.config import Settings
from app.llm.base import LLMRateLimited, LLMRejected, LLMRequest, LLMResponse, LLMTransientError
from app.llm.providers.fake import FakeLLMClient
from app.services import enrichment_service
from app.tasks import jobs
from app.tasks.jobs import RETRYABLE_ERRORS
from tests.extraction_helpers import NEEDS_STACK
from tests.fakes import FakeCooldown, FakeSourceLocks
from tests.task_helpers import Seeder


class Hanging(FakeLLMClient):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(30)
        return await super().complete(request)


@pytest.fixture
def retries(monkeypatch: pytest.MonkeyPatch) -> list[BaseException | None]:
    """Spy on `task.retry`, the call Celery's autoretry wrapper makes for a retryable error."""
    calls: list[BaseException | None] = []

    def retry(*args: Any, exc: BaseException | None = None, **kwargs: Any) -> None:
        calls.append(exc)
        raise Retry(exc=exc)  # what Celery raises to stop the current run and requeue it

    monkeypatch.setattr(jobs.extract_job, "retry", retry)
    return calls


@pytest.fixture
def wired(
    celery_env: None, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeSourceLocks, FakeCooldown]:
    """The task as a worker runs it, with the test database and every outside service faked."""
    locks, cooldown = FakeSourceLocks(), FakeCooldown()
    monkeypatch.setattr("app.tasks.jobs.get_settings", lambda: settings)
    monkeypatch.setattr("app.tasks.jobs.redis_job_locks_for", lambda url: locks)
    monkeypatch.setattr("app.tasks.jobs.redis_cooldown_for", lambda url, provider: cooldown)
    return locks, cooldown


def waiting_job(seed: Seeder) -> uuid.UUID:
    return seed.job(seed.source(seed.user()), NEEDS_STACK)


def use_client(monkeypatch: pytest.MonkeyPatch, client: FakeLLMClient) -> None:
    monkeypatch.setattr("app.tasks.jobs.build_client", lambda settings: client)


# --- provider outcomes are results, never exceptions -------------------------------------------


@pytest.mark.parametrize(
    ("script", "outcome"),
    [
        ([LLMRateLimited(30)], "deferred"),  # 429
        ([LLMRateLimited(None)], "deferred"),
        ([LLMTransientError("server error 503")], "deferred"),  # 503
        ([LLMTransientError("no response in time")], "deferred"),  # a timeout from the adapter
        ([LLMTransientError("connect timeout")], "deferred"),
        ([LLMTransientError("connection failed")], "deferred"),
        ([LLMRejected(401)], "deferred"),  # configuration failure: waits behind a long pause
        ([LLMRejected(403)], "deferred"),
        ([LLMRejected(404)], "deferred"),
        ([LLMRejected(400)], "skipped"),  # about this job's content
        (["not json at all"], "skipped"),  # invalid reply twice (the second request repeats it)
        ([""], "skipped"),
    ],
    ids=lambda value: str(value) if isinstance(value, str) else None,
)
def test_a_provider_outcome_never_causes_a_celery_retry(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    outcome: str,
) -> None:
    locks, _ = wired
    client = FakeLLMClient(script)
    use_client(monkeypatch, client)

    result = jobs.extract_job.run(str(waiting_job(seed)))

    assert result == outcome
    assert retries == []  # the autoretry wrapper never saw an exception
    assert 1 <= len(client.requests) <= 2  # and the service, not Celery, decided about asking again
    assert locks.held == set()


def test_a_provider_that_never_answers_ends_as_a_deferral_not_a_retry(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run's own deadline cuts the call off; the task still returns normally."""
    monkeypatch.setattr(enrichment_service, "run_deadline_seconds", lambda _: 0.05)
    use_client(monkeypatch, Hanging())

    result = jobs.extract_job.run(str(waiting_job(seed)))

    assert result == "deferred" and retries == []


def test_even_a_dead_cooldown_store_is_a_deferral_not_a_retry(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locks, cooldown = wired
    cooldown.unavailable = True
    client = FakeLLMClient()
    use_client(monkeypatch, client)

    assert jobs.extract_job.run(str(waiting_job(seed))) == "deferred"

    assert retries == [] and client.requests == []


# --- the errors that DO cause a retry (so the spy is known to work) ----------------------------


def test_a_database_error_does_cause_a_retry(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OperationalError("select", {}, Exception("connection lost"))

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr("app.tasks.jobs.extraction_service.extract_job", broken)

    with pytest.raises(Retry):
        jobs.extract_job.run(str(waiting_job(seed)))

    assert retries == [error]


def test_a_redis_error_does_cause_a_retry(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RedisConnectionError("redis is down")

    class DownLocks(FakeSourceLocks):
        def acquire(self, source_id: uuid.UUID) -> bool:
            raise error

    monkeypatch.setattr("app.tasks.jobs.redis_job_locks_for", lambda url: DownLocks())

    with pytest.raises(Retry):
        jobs.extract_job.run(str(waiting_job(seed)))

    assert retries == [error]


def test_a_bug_is_not_retried_because_running_it_again_would_fail_the_same_way(
    seed: Seeder,
    wired: tuple[FakeSourceLocks, FakeCooldown],
    retries: list[BaseException | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def buggy(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("a bug")

    monkeypatch.setattr("app.tasks.jobs.extraction_service.extract_job", buggy)

    with pytest.raises(ValueError):
        jobs.extract_job.run(str(waiting_job(seed)))

    assert retries == []


def test_the_retryable_errors_are_exactly_database_and_redis_errors() -> None:
    names = {error.__name__ for error in RETRYABLE_ERRORS}

    assert names == {"RedisError", "OperationalError", "InterfaceError"}
    assert OperationalError in RETRYABLE_ERRORS and InterfaceError in RETRYABLE_ERRORS
    # No provider error class, and nothing from app.llm, is in the list.
    assert not any(error.__module__.startswith("app.") for error in RETRYABLE_ERRORS)
