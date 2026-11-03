"""The extraction task and the post-ingest hook, with fakes only. These tests are synchronous on
purpose: the task runs its own event loop with `asyncio.run`, which cannot start inside an async
test."""

import json
import logging
import uuid
from typing import Any

import pytest
from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.extraction.limits import (
    EXTRACTION_HARD_TIME_LIMIT_SECONDS,
    EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
)
from app.llm.base import LLMRateLimited, LLMRejected
from app.llm.providers.fake import DEFAULT_RESPONSE, FakeLLMClient
from app.models import ExtractionMethod, ExtractionStatus, LlmEnrichment
from app.tasks import jobs
from app.tasks.jobs import (
    ENQUEUE_ERRORS,
    RETRYABLE_ERRORS,
    IngestOutcome,
    enqueue_many,
    run_extract_job,
    run_ingest_source,
)
from tests.extraction_helpers import NEEDS_STACK, RICH
from tests.fakes import FakeCooldown, FakeFeedServer, FakeSourceLocks, feed_bytes
from tests.task_helpers import Seeder

URL = "https://example.com/feed.xml"
BASIC = feed_bytes("rss2_basic.xml")


def answer(**fields: Any) -> str:
    return json.dumps({**json.loads(DEFAULT_RESPONSE), **fields})


class RecordingEnqueue:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[tuple[uuid.UUID, float]] = []
        self.fail_with = fail_with

    def __call__(self, job_id: uuid.UUID, delay: float) -> None:
        self.calls.append((job_id, delay))
        if self.fail_with:
            raise self.fail_with


@pytest.fixture
def locks() -> FakeSourceLocks:
    return FakeSourceLocks()


@pytest.fixture
def user_source(seed: Seeder) -> tuple[uuid.UUID, uuid.UUID]:
    user_id = seed.user()
    return user_id, seed.source(user_id)


# --- the task runs the extraction --------------------------------------------------------------


def test_the_task_extracts_a_job_with_a_fake_model_and_releases_its_lock(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, source_id = user_source
    job_id = seed.job(source_id, NEEDS_STACK)
    llm = FakeLLMClient([answer(stack=["Quux"])])

    result = run_extract_job(
        job_id, settings=settings, llm=llm, cooldown=FakeCooldown(), locks=locks
    )

    assert result == "enriched"
    job = seed.job_row(job_id)
    assert job.extraction_method is ExtractionMethod.RULES_LLM and job.stack == ["Quux"]
    assert locks.events == [("acquire", job_id), ("release", job_id)] and locks.held == set()


def test_without_a_model_the_task_still_runs_the_rules(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Default settings: provider none, so no client is built and no Redis cooldown is needed."""
    _, source_id = user_source
    job_id = seed.job(source_id, RICH)

    result = run_extract_job(job_id, settings=settings, locks=locks)

    assert result == "extracted"
    job = seed.job_row(job_id)
    assert (job.extraction_status, job.extraction_method) == (
        ExtractionStatus.EXTRACTED,
        ExtractionMethod.RULES,
    )


def test_the_task_builds_the_real_client_and_the_redis_cooldown_and_fails_closed(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """With the kill switch on, the task builds its own client (here the fake provider) and the
    Redis cooldown. Redis is a dead port in tests, so the pause cannot be read: no request."""
    user_id, source_id = user_source
    on = settings.model_copy(
        update={
            "llm_provider": "fake",
            "llm_extraction_enabled": True,
            "llm_model_extraction": "some-model",
        }
    )
    job_id = seed.job(source_id, NEEDS_STACK)

    result = run_extract_job(job_id, settings=on, locks=locks)

    assert result == "deferred"
    assert seed.usage(user_id) == []  # nothing was reserved or sent
    job = seed.job_row(job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
    )


@pytest.mark.parametrize(
    "error", [LLMRateLimited(30), LLMRejected(401), LLMRejected(400)], ids=lambda e: str(e)
)
def test_a_provider_failure_is_an_outcome_never_an_exception_so_celery_never_retries_it(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
    error: Exception,
) -> None:
    _, source_id = user_source
    job_id = seed.job(source_id, NEEDS_STACK)
    llm = FakeLLMClient([error], strict=True)  # type: ignore[list-item]

    result = run_extract_job(
        job_id, settings=settings, llm=llm, cooldown=FakeCooldown(), locks=locks
    )

    assert result in {"deferred", "skipped"}
    assert len(llm.requests) == 1
    assert locks.held == set()


def test_running_the_task_again_is_harmless(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, source_id = user_source
    job_id = seed.job(source_id, RICH)
    run_extract_job(job_id, settings=settings, locks=locks)
    before = seed.job_row(job_id)

    assert run_extract_job(job_id, settings=settings, locks=locks) == "already_done"

    assert seed.job_row(job_id).extraction_attempts == before.extraction_attempts == 1


def test_a_job_deleted_after_the_task_was_queued_is_skipped_quietly(
    settings: Settings, locks: FakeSourceLocks
) -> None:
    assert run_extract_job(uuid.uuid4(), settings=settings, locks=locks) == "not_found"


def test_a_run_is_skipped_while_another_run_of_the_same_job_holds_the_lock(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, source_id = user_source
    job_id = seed.job(source_id, RICH)
    locks.held.add(job_id)  # another worker is on it

    assert run_extract_job(job_id, settings=settings, locks=locks) == "skipped_locked"

    assert seed.job_row(job_id).extraction_method is None  # untouched
    assert locks.held == {job_id}  # and the other worker's lock was not released by us


def test_other_jobs_are_not_blocked_by_one_jobs_lock(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, source_id = user_source
    busy, free = seed.job(source_id, RICH), seed.job(source_id, RICH + " ")
    locks.held.add(busy)

    assert run_extract_job(free, settings=settings, locks=locks) in {"extracted", "reused"}


def test_the_lock_is_released_even_when_the_database_fails(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_id = user_source
    job_id = seed.job(source_id, RICH)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise OperationalError("select", {}, Exception("db down"))

    monkeypatch.setattr("app.tasks.jobs.extraction_service.extract_job", broken)

    with pytest.raises(OperationalError):  # visible to Celery, which may retry it
        run_extract_job(job_id, settings=settings, locks=locks)

    assert locks.held == set()


def test_the_task_uses_its_own_throwaway_engine_not_the_apps(
    seed: Seeder,
    settings: Settings,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_id = user_source
    created: list[dict[str, Any]] = []

    def spy(url: str, **kwargs: Any) -> Any:
        created.append(kwargs)
        return create_async_engine(url, **kwargs)

    def forbidden() -> None:
        raise AssertionError("the app's cached engine belongs to another event loop")

    monkeypatch.setattr("app.tasks.jobs.create_async_engine", spy)
    monkeypatch.setattr("app.db.session.get_engine", forbidden)

    run_extract_job(seed.job(source_id, RICH), settings=settings, locks=locks)

    assert [kwargs["poolclass"] for kwargs in created] == [NullPool]


# --- the Celery task wrapper is thin -----------------------------------------------------------


def test_the_extract_task_just_calls_the_function_with_a_uuid(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[uuid.UUID] = []

    def fake_run(job_id: uuid.UUID) -> str:
        seen.append(job_id)
        return "extracted"

    monkeypatch.setattr("app.tasks.jobs.run_extract_job", fake_run)
    job_id = uuid.uuid4()

    assert jobs.extract_job.run(str(job_id)) == "extracted"
    assert seen == [job_id]
    with pytest.raises(ValueError):
        jobs.extract_job.run("not-a-uuid")


def test_the_extract_task_retries_only_database_and_redis_errors(celery_env: None) -> None:
    task = jobs.extract_job

    assert set(task.autoretry_for) == set(RETRYABLE_ERRORS)  # nothing about the provider
    assert (
        (task.soft_time_limit, task.time_limit)
        == (
            EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
            EXTRACTION_HARD_TIME_LIMIT_SECONDS,
        )
        == (60, 90)
    )
    assert task.max_retries == 3


# --- the post-ingest hook ----------------------------------------------------------------------


def sync(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    source_id: uuid.UUID,
    enqueue: RecordingEnqueue,
    locks: FakeSourceLocks,
) -> IngestOutcome:
    return run_ingest_source(
        source_id,
        settings=settings,
        fetchers=fetchers,
        locks=locks,
        enqueue_extraction=enqueue,
    )


def test_the_hook_queues_exactly_the_jobs_that_were_inserted(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    feed_server.add(URL, BASIC)
    _, source_id = user_source
    enqueue = RecordingEnqueue()

    sync(seed, settings, fetchers, source_id, enqueue, locks)

    stored = {job.id for job in seed.jobs(source_id)}
    assert len(stored) == 3
    assert {job_id for job_id, _ in enqueue.calls} == stored
    delays = [delay for _, delay in enqueue.calls]
    assert delays == sorted(delays) and delays[0] == 0  # spread out, not all at once


def test_a_second_sync_of_the_same_feed_queues_nothing(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    feed_server.add(URL, BASIC)
    _, source_id = user_source
    sync(seed, settings, fetchers, source_id, RecordingEnqueue(), locks)
    again = RecordingEnqueue()

    sync(seed, settings, fetchers, source_id, again, locks)

    assert again.calls == []  # the posts already existed: nothing was inserted, nothing queued


@pytest.mark.parametrize(
    "error",
    [
        KombuOperationalError("broker down"),
        RedisConnectionError("redis down"),
        ConnectionRefusedError("refused"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_failed_enqueue_never_fails_ingestion(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    feed_server.add(URL, BASIC)
    user_id, source_id = user_source
    enqueue = RecordingEnqueue(fail_with=error)

    with caplog.at_level(logging.WARNING, logger="app.tasks.jobs"):
        outcome = sync(seed, settings, fetchers, source_id, enqueue, locks)

    assert outcome is IngestOutcome.INGESTED
    assert len(seed.jobs(source_id)) == 3  # stored regardless
    assert seed.source_row(source_id).last_error is None
    assert len(enqueue.calls) == 1  # the first failure ends it: no hammering a dead broker
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "3 of 3 extraction tasks were not queued" in record.getMessage()
    assert locks.held == set()


def test_nothing_is_queued_for_a_locked_missing_or_failing_source(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
    user_source: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, source_id = user_source
    enqueue = RecordingEnqueue()
    locks.held.add(source_id)
    assert sync(seed, settings, fetchers, source_id, enqueue, locks) is IngestOutcome.SKIPPED_LOCKED
    locks.held.clear()
    assert (
        sync(seed, settings, fetchers, uuid.uuid4(), enqueue, locks)
        is IngestOutcome.SKIPPED_MISSING
    )
    sync(seed, settings, fetchers, source_id, enqueue, locks)  # the feed server has no such feed

    assert enqueue.calls == []


# --- queueing helper ---------------------------------------------------------------------------


def test_enqueue_many_returns_how_many_were_queued_and_spreads_the_delays() -> None:
    enqueue = RecordingEnqueue()
    ids = [uuid.uuid4() for _ in range(5)]

    assert enqueue_many(ids, enqueue) == 5

    assert [job_id for job_id, _ in enqueue.calls] == ids
    assert enqueue_many([], enqueue) == 0


def test_a_broker_failure_midway_stops_there_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[uuid.UUID] = []

    def flaky(job_id: uuid.UUID, delay: float) -> None:
        calls.append(job_id)
        if len(calls) == 3:
            raise OSError("broken pipe")

    ids = [uuid.uuid4() for _ in range(10)]

    with caplog.at_level(logging.WARNING, logger="app.tasks.jobs"):
        queued = enqueue_many(ids, flaky)

    assert queued == 2 and len(calls) == 3  # nothing was tried after the failure
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "8 of 10" in caplog.records[0].getMessage()


def test_only_connection_type_errors_are_swallowed_by_the_hook() -> None:
    assert issubclass(OSError, ENQUEUE_ERRORS) or OSError in ENQUEUE_ERRORS
    with pytest.raises(ValueError):  # a bug is not hidden as "the queue is down"
        enqueue_many([uuid.uuid4()], RecordingEnqueue(fail_with=ValueError("bug")))
