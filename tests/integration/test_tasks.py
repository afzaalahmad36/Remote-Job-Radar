import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.tasks import jobs
from app.tasks.jobs import (
    SPREAD_STEP_SECONDS,
    SPREAD_WINDOW_SECONDS,
    IngestOutcome,
    run_ingest_all_active_sources,
    run_ingest_source,
    run_purge_expired_refresh_tokens,
    spread_delays,
)
from tests.fakes import FakeFeedServer, FakeSourceLocks, feed_bytes
from tests.task_helpers import Seeder

URL = "https://example.com/feed.xml"
BASIC = feed_bytes("rss2_basic.xml")


@pytest.fixture
def locks() -> FakeSourceLocks:
    return FakeSourceLocks()


def run(
    seed: Seeder, source_id: uuid.UUID, settings: Settings, fetchers: Any, locks: FakeSourceLocks
) -> IngestOutcome:
    return run_ingest_source(source_id, settings=settings, fetchers=fetchers, locks=locks)


# --- ingest one source ---------------------------------------------------------------------------


def test_the_task_syncs_the_source_and_releases_its_lock(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, BASIC)
    source_id = seed.source(seed.user())

    outcome = run(seed, source_id, settings, fetchers, locks)

    assert outcome is IngestOutcome.INGESTED
    assert len(seed.jobs(source_id)) == 3
    assert seed.source_row(source_id).last_synced_at is not None
    assert locks.events == [("acquire", source_id), ("release", source_id)]
    assert locks.held == set()


def test_running_the_task_again_is_harmless(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, BASIC)
    source_id = seed.source(seed.user())

    first = run(seed, source_id, settings, fetchers, locks)
    second = run(seed, source_id, settings, fetchers, locks)  # a retry, or a duplicate delivery

    assert (first, second) == (IngestOutcome.INGESTED, IngestOutcome.INGESTED)
    assert len(seed.jobs(source_id)) == 3  # nothing doubled


def test_new_posts_are_picked_up_on_a_later_run(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    source_id = seed.source(seed.user())
    feed_server.add(URL, BASIC)
    run(seed, source_id, settings, fetchers, locks)
    new_item = b"<item><title>New</title><guid>job-104</guid><description>x</description></item>"
    feed_server.add(URL, BASIC.replace(b"</channel>", new_item + b"</channel>"))

    run(seed, source_id, settings, fetchers, locks)

    assert len(seed.jobs(source_id)) == 4


def test_a_source_deleted_after_the_task_was_queued_is_skipped_quietly(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, BASIC)
    source_id = seed.source(seed.user())
    seed.delete_source(source_id)  # deleted between queueing and running

    outcome = run(seed, source_id, settings, fetchers, locks)

    assert outcome is IngestOutcome.SKIPPED_MISSING
    assert feed_server.requests == []  # nothing was fetched
    assert locks.held == set()  # the lock was still released


def test_an_id_that_never_existed_is_skipped_too(
    seed: Seeder, settings: Settings, fetchers: Any, locks: FakeSourceLocks
) -> None:
    assert run(seed, uuid.uuid4(), settings, fetchers, locks) is IngestOutcome.SKIPPED_MISSING


def test_a_run_is_skipped_while_another_run_of_the_same_source_holds_the_lock(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, BASIC)
    source_id = seed.source(seed.user())
    locks.held.add(source_id)  # a beat run (or a manual sync) is in progress

    outcome = run(seed, source_id, settings, fetchers, locks)

    assert outcome is IngestOutcome.SKIPPED_LOCKED
    assert feed_server.requests == [] and seed.jobs(source_id) == []
    assert locks.held == {source_id}  # the skipper did not release the other run's lock
    assert locks.events == [("acquire", source_id)]


def test_other_sources_are_not_blocked_by_one_sources_lock(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, BASIC)
    user = seed.user()
    busy, free = seed.source(user, name="busy"), seed.source(user, name="free")
    locks.held.add(busy)

    assert run(seed, free, settings, fetchers, locks) is IngestOutcome.INGESTED


def test_a_failing_feed_is_recorded_and_the_task_still_succeeds(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    feed_server.add(URL, status=500)
    source_id = seed.source(seed.user())

    outcome = run(seed, source_id, settings, fetchers, locks)  # must not raise

    assert outcome is IngestOutcome.INGESTED
    assert seed.source_row(source_id).last_error == "HTTP 500"
    assert locks.held == set()


def test_a_timeout_in_the_feed_is_recorded_not_raised(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    feed_server.add_handler(URL, slow)
    source_id = seed.source(seed.user())

    run(seed, source_id, settings, fetchers, locks)

    assert seed.source_row(source_id).last_error == "Timed out"


@pytest.mark.parametrize("error", [RuntimeError("boom"), SoftTimeLimitExceeded()])
def test_the_lock_is_released_even_when_the_sync_blows_up(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    locks: FakeSourceLocks,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    async def explode(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr("app.tasks.jobs.ingestion_service.ingest_source", explode)
    source_id = seed.source(seed.user())

    with pytest.raises(type(error)):  # visible to Celery (retry or failure), not swallowed
        run(seed, source_id, settings, fetchers, locks)

    assert locks.held == set()  # so the next run is not blocked until the lock expires


def test_the_task_uses_its_own_throwaway_engine_not_the_apps(
    seed: Seeder,
    settings: Settings,
    fetchers: Any,
    feed_server: FakeFeedServer,
    locks: FakeSourceLocks,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed_server.add(URL, BASIC)
    created: list[dict[str, Any]] = []

    def spy(url: str, **kwargs: Any) -> Any:
        created.append(kwargs)
        return create_async_engine(url, **kwargs)

    def forbidden() -> None:
        raise AssertionError("the app's cached engine belongs to another event loop")

    monkeypatch.setattr("app.tasks.jobs.create_async_engine", spy)
    monkeypatch.setattr("app.db.session.get_engine", forbidden)
    source_id = seed.source(seed.user())

    run(seed, source_id, settings, fetchers, locks)

    assert [kwargs["poolclass"] for kwargs in created] == [NullPool]  # one engine, no pooling


# --- the Celery task wrappers are thin -----------------------------------------------------------


def test_the_ingest_task_just_calls_the_function_with_a_uuid(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[uuid.UUID] = []

    def fake_run(source_id: uuid.UUID) -> IngestOutcome:
        seen.append(source_id)
        return IngestOutcome.SKIPPED_LOCKED

    monkeypatch.setattr("app.tasks.jobs.run_ingest_source", fake_run)
    source_id = uuid.uuid4()

    result = jobs.ingest_source.run(str(source_id))

    assert seen == [source_id]
    assert result == "skipped_locked"  # a plain string: JSON-serializable for Celery


def test_the_ingest_task_rejects_a_malformed_id(celery_env: None) -> None:
    with pytest.raises(ValueError):
        jobs.ingest_source.run("not-a-uuid")


def test_tasks_retry_only_transient_failures_a_limited_number_of_times(celery_env: None) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError
    from sqlalchemy.exc import OperationalError

    for task in (jobs.ingest_source, jobs.purge_expired_refresh_tokens):
        assert RedisConnectionError in task.autoretry_for or any(
            issubclass(RedisConnectionError, exc) for exc in task.autoretry_for
        )
        assert OperationalError in task.autoretry_for
        assert ValueError not in task.autoretry_for  # a bug is not retried
        assert task.max_retries == 3


# --- fan out -------------------------------------------------------------------------------------


def test_fan_out_queues_one_task_per_active_source_and_none_for_inactive(
    seed: Seeder, settings: Settings
) -> None:
    ann, bob = seed.user(), seed.user()
    a1 = seed.source(ann, name="a1")
    a2 = seed.source(ann, name="a2")
    inactive = seed.source(ann, name="off", active=False)
    b1 = seed.source(bob, name="b1")
    queued: list[tuple[uuid.UUID, float]] = []

    count = run_ingest_all_active_sources(
        lambda sid, delay: queued.append((sid, delay)), settings=settings
    )

    ids = [source_id for source_id, _ in queued]
    assert count == 3
    assert sorted(ids) == sorted([a1, a2, b1])  # every user's active sources, once each
    assert inactive not in ids


def test_fan_out_queues_the_longest_waiting_sources_first(seed: Seeder, settings: Settings) -> None:
    user = seed.user()
    now = datetime.now(UTC)
    recent = seed.source(user, name="recent", last_synced_at=now - timedelta(minutes=5))
    stale = seed.source(user, name="stale", last_synced_at=now - timedelta(days=2))
    never = seed.source(user, name="never")
    queued: list[uuid.UUID] = []

    run_ingest_all_active_sources(lambda sid, delay: queued.append(sid), settings=settings)

    assert queued == [never, stale, recent]


def test_fan_out_spreads_the_delays(seed: Seeder, settings: Settings) -> None:
    user = seed.user()
    for n in range(4):
        seed.source(user, name=f"s{n}")
    delays: list[float] = []

    run_ingest_all_active_sources(lambda sid, delay: delays.append(delay), settings=settings)

    assert delays == [0.0, 1.0, 2.0, 3.0]


def test_fan_out_with_no_sources_queues_nothing(seed: Seeder, settings: Settings) -> None:
    queued: list[Any] = []

    assert run_ingest_all_active_sources(lambda *args: queued.append(args), settings=settings) == 0
    assert queued == []


def test_the_fan_out_task_queues_ingest_tasks_with_delay_and_expiry(
    celery_env: None, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    calls: list[dict[str, Any]] = []

    class FakeIngestTask:
        def apply_async(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    def fake_fan_out(enqueue: Any, *, settings: Settings) -> int:
        for position, source_id in enumerate(ids):
            enqueue(source_id, float(position))
        return len(ids)

    monkeypatch.setattr("app.tasks.jobs.get_settings", lambda: settings)
    monkeypatch.setattr("app.tasks.jobs.run_ingest_all_active_sources", fake_fan_out)
    monkeypatch.setattr("app.tasks.jobs.ingest_source", FakeIngestTask())

    assert jobs.ingest_all_active_sources.run() == 2

    expires = settings.ingest_interval_minutes * 60
    assert calls == [
        {"args": [str(ids[0])], "countdown": 0.0, "expires": expires},
        {"args": [str(ids[1])], "countdown": 1.0, "expires": expires},
    ]


@pytest.mark.parametrize("count", [0, 1, 2, 5, 29, 30, 31, 100, 1000, 20_000])
def test_spread_delays_are_ascending_and_never_exceed_the_window(count: int) -> None:
    delays = spread_delays(count)

    assert len(delays) == count
    assert delays == sorted(delays)
    assert not delays or (delays[0] == 0 and delays[-1] < SPREAD_WINDOW_SECONDS)


def test_a_few_sources_are_one_step_apart_and_many_are_squeezed_into_the_window() -> None:
    assert spread_delays(3) == [0, SPREAD_STEP_SECONDS, 2 * SPREAD_STEP_SECONDS]
    assert spread_delays(60)[1] == SPREAD_WINDOW_SECONDS / 60  # 0.5 s apart


# --- purging refresh tokens ----------------------------------------------------------------------


def test_the_purge_task_deletes_long_expired_tokens_and_keeps_the_rest(
    seed: Seeder, settings: Settings
) -> None:
    user = seed.user()
    now = datetime.now(UTC)
    long_expired = seed.refresh_token(user, now - timedelta(days=30))
    just_expired = seed.refresh_token(user, now - timedelta(days=1))
    valid = seed.refresh_token(user, now + timedelta(days=3))

    deleted = run_purge_expired_refresh_tokens(settings=settings)

    assert deleted >= 1
    assert not seed.token_exists(long_expired)
    assert seed.token_exists(just_expired)  # expired, but still inside the grace period
    assert seed.token_exists(valid)


def test_the_purge_task_wrapper_calls_the_function(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.tasks.jobs.run_purge_expired_refresh_tokens", lambda: 42)

    assert jobs.purge_expired_refresh_tokens.run() == 42
