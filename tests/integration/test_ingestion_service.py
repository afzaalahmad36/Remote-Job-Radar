import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.ingestion.base import FetchError, FetchResult, RawJob
from app.ingestion.rss import MAX_RESPONSE_BYTES
from app.ingestion.text import MAX_TEXT_CHARS
from app.models import ExtractionStatus, Job, Source, SourceType, User
from app.services.ingestion_service import _insert_jobs, ingest_source
from tests.fakes import FakeFeedServer, FakeResolver, feed_bytes

URL = "https://example.com/feed.xml"
NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

LEAKS = ("169.254", "127.0.0.1", "10.0.0.5", "internal.example", "localhost")


async def make_source(session: AsyncSession, url: str = URL, name: str = "feed") -> Source:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="x")
    session.add(user)
    await session.flush()
    source = Source(user_id=user.id, type=SourceType.RSS, name=name, config={"url": url})
    session.add(source)
    await session.flush()
    return source


async def jobs_of(session: AsyncSession, source: Source) -> list[Job]:
    rows = await session.scalars(
        select(Job).where(Job.source_id == source.id).order_by(Job.external_id)
    )
    return list(rows)


# --- storing posts -------------------------------------------------------------------------------


async def test_a_first_sync_stores_the_posts_as_raw_pending_jobs(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(
        URL,
        feed_bytes("rss2_basic.xml"),
        headers={"ETag": '"v1"', "Last-Modified": "Mon, 26 Oct 2026 10:00:00 GMT"},
    )
    source = await make_source(db_session)

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert (summary.fetched, summary.inserted, summary.skipped, summary.error) == (3, 3, 0, None)
    jobs = await jobs_of(db_session, source)
    assert [job.external_id for job in jobs] == ["job-101", "job-102", "job-103"]
    first = jobs[0]
    assert first.raw_text.startswith(
        "Senior Python Engineer at Acme\n\nWe are hiring a Senior Python Engineer."
    )
    assert first.posted_at == datetime(2026, 10, 26, 12, 30, tzinfo=UTC)  # 14:30 +0200, in UTC
    assert first.extraction_status == ExtractionStatus.PENDING
    # Everything the LLM will extract in Phase 3 is still empty.
    for job in jobs:
        assert (job.title, job.company, job.seniority, job.salary_min, job.salary_max) == (
            None,
        ) * 5
        assert (job.salary_currency, job.remote_policy, job.apply_email, job.apply_url) == (
            None,
        ) * 4
        assert job.stack == []


async def test_the_source_records_the_sync(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(
        URL,
        feed_bytes("rss2_basic.xml"),
        headers={"ETag": '"v1"', "Last-Modified": "Mon, 26 Oct 2026 10:00:00 GMT"},
    )
    source = await make_source(db_session)

    await ingest_source(db_session, source, fetchers, now=NOW)

    assert source.last_synced_at == NOW
    assert source.last_error is None
    assert (source.etag, source.last_modified) == ('"v1"', "Mon, 26 Oct 2026 10:00:00 GMT")


async def test_a_second_sync_inserts_nothing(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source = await make_source(db_session)
    await ingest_source(db_session, source, fetchers, now=NOW)

    summary = await ingest_source(db_session, source, fetchers, now=NOW + timedelta(hours=1))

    assert (summary.fetched, summary.inserted, summary.skipped, summary.error) == (3, 0, 3, None)
    assert len(await jobs_of(db_session, source)) == 3


async def test_only_new_posts_are_inserted_on_a_later_sync(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    source = await make_source(db_session)
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    await ingest_source(db_session, source, fetchers, now=NOW)
    new_item = (
        b"<item><title>Brand new</title><guid>job-104</guid>"
        b"<description>Fresh.</description></item>"
    )
    grown = feed_bytes("rss2_basic.xml").replace(b"</channel>", new_item + b"</channel>")
    feed_server.add(URL, grown)

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert (summary.fetched, summary.inserted, summary.skipped) == (4, 1, 3)


async def test_a_post_that_already_exists_is_skipped_not_an_error(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source = await make_source(db_session)
    db_session.add(
        Job(source_id=source.id, external_id="job-102", raw_text="stored by someone else")
    )
    await db_session.flush()

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert (summary.inserted, summary.skipped, summary.error) == (2, 1, None)
    existing = next(j for j in await jobs_of(db_session, source) if j.external_id == "job-102")
    assert existing.raw_text == "stored by someone else"  # never overwritten


async def test_repeated_entries_inside_one_feed_are_inserted_once(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("duplicates.xml"))
    source = await make_source(db_session)

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert (summary.fetched, summary.inserted, summary.skipped) == (5, 3, 2)
    assert [j.external_id for j in await jobs_of(db_session, source)] == [
        "dup-1",
        "https://example.com/jobs/same-link",
        "unique-1",
    ]


async def test_guid_then_link_then_hash_ids_are_stable_across_syncs(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("no_guid.xml"))
    source = await make_source(db_session)

    first = await ingest_source(db_session, source, fetchers, now=NOW)
    second = await ingest_source(db_session, source, fetchers, now=NOW)

    ids = sorted(j.external_id for j in await jobs_of(db_session, source))
    assert ids[0] == "https://example.com/jobs/linked-1"
    assert all(i.startswith("sha256:") for i in ids[1:])
    assert (first.inserted, second.inserted) == (3, 0)  # the hash ids matched the second time


async def test_html_is_stripped_and_a_missing_date_is_stored_as_null(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("html_descriptions.xml"))
    source = await make_source(db_session)

    await ingest_source(db_session, source, fetchers, now=NOW)

    job = (await jobs_of(db_session, source))[0]
    assert job.external_id == "html-1"
    assert "<" not in job.raw_text and "alert" not in job.raw_text
    assert job.posted_at is None


async def test_two_sources_with_the_same_feed_each_get_their_own_jobs(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    first, second = await make_source(db_session), await make_source(db_session)

    a = await ingest_source(db_session, first, fetchers, now=NOW)
    b = await ingest_source(db_session, second, fetchers, now=NOW)

    assert (a.inserted, b.inserted) == (3, 3)  # uniqueness is per source, not global
    assert await db_session.scalar(select(func.count()).select_from(Job)) == 6


# --- conditional GET -----------------------------------------------------------------------------


async def test_a_304_is_a_successful_no_op_that_keeps_the_validators(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, content=feed_bytes("rss2_basic.xml"), headers={"ETag": '"v1"'})

    feed_server.add_handler(URL, handler)
    source = await make_source(db_session)
    await ingest_source(db_session, source, fetchers, now=NOW)
    later = NOW + timedelta(hours=1)

    summary = await ingest_source(db_session, source, fetchers, now=later)

    assert (summary.fetched, summary.inserted, summary.skipped, summary.error) == (0, 0, 0, None)
    assert source.last_synced_at == later  # a 304 still counts as a successful sync
    assert source.etag == '"v1"'  # not wiped
    assert len(await jobs_of(db_session, source)) == 3


async def test_a_304_clears_an_earlier_error(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, status=304)
    source = await make_source(db_session)
    source.last_error = "HTTP 500"
    source.etag = '"v1"'

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert summary.error is None and source.last_error is None


# --- failures never crash and are recorded in a short, sanitized form ----------------------------


def timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("read timed out talking to 10.0.0.5")


@pytest.mark.parametrize(
    ("register", "expected"),
    [
        (lambda s: s.add(URL, b"<html>oops</html>", status=500), "HTTP 500"),
        (lambda s: s.add(URL, b"", status=404), "HTTP 404"),
        (lambda s: s.add_handler(URL, timeout_handler), "Timed out"),
        (
            lambda s: s.add(URL, b"x", headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)}),
            "Response too large",
        ),
        (lambda s: s.add(URL, feed_bytes("malformed.xml")), "Not a valid RSS or Atom feed"),
        (lambda s: s.add(URL, feed_bytes("not_a_feed.html")), "Not a valid RSS or Atom feed"),
    ],
    ids=["http-500", "http-404", "timeout", "oversize", "bad-xml", "html-page"],
)
async def test_a_failure_is_recorded_not_raised(
    db_session: AsyncSession,
    fetchers: dict,
    feed_server: FakeFeedServer,
    register: object,
    expected: str,
) -> None:
    register(feed_server)  # type: ignore[operator]
    source = await make_source(db_session)
    source.etag = '"keep-me"'

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert (summary.fetched, summary.inserted, summary.skipped) == (0, 0, 0)
    assert summary.error == expected
    assert source.last_error == expected
    assert source.last_synced_at is None  # that is the last *successful* sync
    assert source.etag == '"keep-me"'
    assert len(await jobs_of(db_session, source)) == 0


async def test_an_unsafe_url_is_recorded_without_any_request(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    source = await make_source(db_session, url="http://169.254.169.254/latest/meta-data/")

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert summary.error is not None and summary.error.startswith("URL not allowed: ")
    assert feed_server.requests == []
    assert not any(leak in summary.error for leak in LEAKS)


async def test_a_redirect_to_a_private_address_is_recorded_without_leaking_it(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    feed_server.redirect(URL, "http://internal.example.com/secret-admin")
    source = await make_source(db_session)

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert summary.error is not None and summary.error.startswith("Redirect not allowed: ")
    assert source.last_error == summary.error
    assert not any(leak in summary.error for leak in (*LEAKS, "secret-admin"))
    assert feed_server.urls == [URL]


async def test_a_successful_sync_clears_the_previous_error(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    source = await make_source(db_session)
    feed_server.add(URL, status=503)
    await ingest_source(db_session, source, fetchers, now=NOW)
    assert source.last_error == "HTTP 503"
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert summary.error is None
    assert source.last_error is None
    assert source.last_synced_at == NOW


async def test_one_failing_source_does_not_affect_the_others(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer
) -> None:
    broken_url, good_url = "https://example.com/broken.xml", "https://feeds.example.org/rss"
    feed_server.add(broken_url, status=500)
    feed_server.add(good_url, feed_bytes("atom_basic.xml"))
    broken = await make_source(db_session, broken_url)
    good = await make_source(db_session, good_url)

    results = [await ingest_source(db_session, s, fetchers, now=NOW) for s in (broken, good)]

    assert [r.error for r in results] == ["HTTP 500", None]
    assert results[1].inserted == 2
    assert (broken.last_error, good.last_error) == ("HTTP 500", None)
    assert len(await jobs_of(db_session, broken)) == 0


class ExplodingFetcher:
    async def fetch(self, source: Source) -> FetchResult:
        raise RuntimeError("password=hunter2 while talking to 10.0.0.5")


async def test_an_unexpected_error_is_recorded_generically_and_logged(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    source = await make_source(db_session)

    with caplog.at_level(logging.ERROR):
        summary = await ingest_source(
            db_session, source, {SourceType.RSS: ExplodingFetcher()}, now=NOW
        )

    assert summary.error == "Unexpected error while fetching"
    assert source.last_error == "Unexpected error while fetching"  # nothing from the exception
    assert "Unexpected error while fetching source" in caplog.text  # the real cause is in the log


async def test_the_stored_error_is_capped_at_the_column_size(db_session: AsyncSession) -> None:
    class Wordy:
        async def fetch(self, source: Source) -> FetchResult:
            raise FetchError("x" * 1000)

    source = await make_source(db_session)

    summary = await ingest_source(db_session, source, {SourceType.RSS: Wordy()}, now=NOW)

    assert summary.error is not None and len(summary.error) == 500


async def test_no_database_transaction_is_open_while_the_network_is_used(
    db_session: AsyncSession,
) -> None:
    class Spy:
        in_transaction: bool | None = None

        async def fetch(self, source: Source) -> FetchResult:
            Spy.in_transaction = db_session.in_transaction()  # asked from inside the fetch
            return FetchResult()

    source = await make_source(db_session)
    assert db_session.in_transaction()  # the lookups before a sync do start a transaction

    await ingest_source(db_session, source, {SourceType.RSS: Spy()}, now=NOW)

    assert Spy.in_transaction is False  # ...and it was ended before the fetcher ran


async def test_a_fetcher_cannot_store_more_than_the_text_cap(db_session: AsyncSession) -> None:
    class Huge:
        async def fetch(self, source: Source) -> FetchResult:
            return FetchResult(jobs=[RawJob("big", "x" * 100_000, None)], fetched=1)

    source = await make_source(db_session)

    await ingest_source(db_session, source, {SourceType.RSS: Huge()}, now=NOW)

    assert len((await jobs_of(db_session, source))[0].raw_text) == MAX_TEXT_CHARS == 20_000


async def test_two_syncs_at_once_store_each_post_exactly_once(
    test_database_url: str, migrated_db: None
) -> None:
    """Real concurrency: two sessions on two connections, committing for real.

    A barrier makes both syncs finish fetching before either inserts, then both insert the
    same posts (in opposite feed orders). Whichever insert gets a post first stores it; the
    other skips it. This checks the outcome (no duplicates, nothing lost, exact counts). It
    cannot prove the deadlock-avoidance ordering, because the client sends the two statements
    one after the other and they rarely overlap on the server: that is covered by
    `test_rows_are_inserted_in_a_fixed_order...` below.

    Everything created here is committed, so it is removed in `finally` (deleting the user
    cascades to its sources and jobs).
    """
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    posts = [RawJob(f"post-{i:04d}", f"text {i}", None) for i in range(500)]

    class Fetcher:
        def __init__(self, order: list[RawJob], barrier: asyncio.Barrier) -> None:
            self.order, self.barrier = order, barrier

        async def fetch(self, source: Source) -> FetchResult:
            await self.barrier.wait()
            return FetchResult(jobs=self.order, fetched=len(self.order))

    async def sync_with(order: list[RawJob], source_id: uuid.UUID, barrier: asyncio.Barrier) -> int:
        async with sessions() as session:
            source = await session.get(Source, source_id)
            assert source is not None
            summary = await ingest_source(
                session, source, {SourceType.RSS: Fetcher(order, barrier)}
            )
            assert summary.error is None
            return summary.inserted

    user_id: uuid.UUID | None = None
    try:
        async with sessions() as setup:
            user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="x")
            setup.add(user)
            await setup.commit()
            user_id = user.id

        # A fresh source each round, so the two inserts collide on a source with no jobs yet.
        for round_number in range(3):
            async with sessions() as setup:
                source = Source(
                    user_id=user_id, type=SourceType.RSS, name=f"round-{round_number}", config={}
                )
                setup.add(source)
                await setup.commit()
                source_id = source.id
            barrier = asyncio.Barrier(2)
            inserted = await asyncio.gather(
                sync_with(posts, source_id, barrier),
                sync_with(list(reversed(posts)), source_id, barrier),
            )
            async with sessions() as check:
                stored = await check.scalar(
                    select(func.count()).select_from(Job).where(Job.source_id == source_id)
                )
            assert stored == len(posts)  # no duplicates, nothing lost
            assert sum(inserted) == len(posts)  # each post inserted by exactly one of the two
    finally:
        if user_id is not None:
            async with sessions() as cleanup:
                await cleanup.execute(delete(User).where(User.id == user_id))  # cascades
                await cleanup.commit()
        await engine.dispose()


async def test_rows_are_inserted_in_a_fixed_order_so_overlapping_syncs_cannot_deadlock() -> None:
    # Two syncs that insert overlapping rows in *different* orders can each end up waiting for a
    # row the other holds. If every sync inserts in the same (sorted) order, that cannot happen.
    statements: list[Any] = []

    class StubSession:
        async def execute(self, statement: Any) -> Any:
            statements.append(statement)
            return type("Result", (), {"scalars": lambda self: self, "all": lambda self: []})()

    feed_order = [RawJob(f"post-{n}", "text", None) for n in (3, 1, 4, 2, 0)]

    await _insert_jobs(StubSession(), uuid.uuid4(), feed_order)  # type: ignore[arg-type]

    params = statements[0].compile().params
    inserted_order = [params[f"external_id_m{i}"] for i in range(len(feed_order))]
    assert inserted_order == ["post-0", "post-1", "post-2", "post-3", "post-4"]


async def test_a_source_type_without_a_fetcher_is_an_error_not_a_crash(
    db_session: AsyncSession, fetchers: dict
) -> None:
    source = await make_source(db_session)
    source.type = SourceType.TELEGRAM

    summary = await ingest_source(db_session, source, fetchers, now=NOW)

    assert summary.error == "Unsupported source type"


async def test_default_fetchers_are_used_when_none_are_given(db_session: AsyncSession) -> None:
    source = await make_source(
        db_session, url="http://localhost/feed"
    )  # refused before any network

    summary = await ingest_source(db_session, source, now=NOW)

    assert summary.error is not None and summary.error.startswith("URL not allowed")


async def test_the_resolver_is_consulted_for_each_sync(
    db_session: AsyncSession, fetchers: dict, feed_server: FakeFeedServer, resolver: FakeResolver
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source = await make_source(db_session)

    await ingest_source(db_session, source, fetchers, now=NOW)
    await ingest_source(db_session, source, fetchers, now=NOW)

    assert resolver.calls == ["example.com", "example.com"]
