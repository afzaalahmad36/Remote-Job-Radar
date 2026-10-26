import asyncio
import gzip
import time
import uuid
from typing import Any

import httpx
import pytest

from app.ingestion.base import FetchError
from app.ingestion.rss import MAX_ENTRIES, MAX_REDIRECTS, MAX_RESPONSE_BYTES, RssFetcher
from app.models import Source, SourceType
from tests.fakes import FakeFeedServer, FakeResolver, feed_bytes

URL = "https://example.com/feed.xml"
BASIC = feed_bytes("rss2_basic.xml")

# Pieces of private targets that must never appear in an error message shown to the user.
LEAKS = (
    "169.254",
    "127.0.0.1",
    "10.0.0.5",
    "internal.example",
    "2130706433",
    "localhost",
    "passwd",
)


def source_for(url: str = URL, etag: str | None = None, last_modified: str | None = None) -> Source:
    return Source(
        user_id=uuid.uuid4(),
        type=SourceType.RSS,
        name="feed",
        config={"url": url},
        etag=etag,
        last_modified=last_modified,
    )


@pytest.fixture
def fetcher(resolver: FakeResolver, feed_server: FakeFeedServer) -> RssFetcher:
    return RssFetcher(resolver=resolver, transport=feed_server.transport)


async def fetch_error(fetcher: RssFetcher, source: Source | None = None) -> str:
    with pytest.raises(FetchError) as error:
        await fetcher.fetch(source or source_for())
    return str(error.value)


# --- the happy path and conditional GET ----------------------------------------------------------


async def test_fetch_returns_the_posts_and_the_validators(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(
        URL, BASIC, headers={"ETag": '"v1"', "Last-Modified": "Mon, 26 Oct 2026 10:00:00 GMT"}
    )

    result = await fetcher.fetch(source_for())

    assert (result.fetched, len(result.jobs), result.not_modified) == (3, 3, False)
    assert result.etag == '"v1"'
    assert result.last_modified == "Mon, 26 Oct 2026 10:00:00 GMT"


async def test_the_first_request_is_not_conditional_and_identifies_itself(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, BASIC)

    await fetcher.fetch(source_for())

    headers = feed_server.requests[0].headers
    assert "if-none-match" not in headers and "if-modified-since" not in headers
    assert headers["user-agent"].startswith("RemoteJobRadar/")
    assert "rss" in headers["accept"]


async def test_stored_validators_are_sent_and_a_304_is_a_successful_no_op(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, content=BASIC)

    feed_server.add_handler(URL, handler)
    source = source_for(etag='"v1"', last_modified="Mon, 26 Oct 2026 10:00:00 GMT")

    result = await fetcher.fetch(source)

    request = feed_server.requests[0]
    assert request.headers["if-none-match"] == '"v1"'
    assert request.headers["if-modified-since"] == "Mon, 26 Oct 2026 10:00:00 GMT"
    assert (result.not_modified, result.jobs, result.fetched) == (True, [], 0)


async def test_a_200_after_a_conditional_request_returns_the_new_validators(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, BASIC, headers={"ETag": '"v2"'})

    result = await fetcher.fetch(source_for(etag='"v1"'))

    assert (result.not_modified, result.etag, len(result.jobs)) == (False, '"v2"', 3)


async def test_unusable_validators_are_dropped_instead_of_stored(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, BASIC, headers={"ETag": '"' + "x" * 300 + '"'})  # longer than the column

    assert (await fetcher.fetch(source_for())).etag is None


async def test_a_gzip_compressed_feed_is_decoded(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, gzip.compress(BASIC), headers={"Content-Encoding": "gzip"})

    assert len((await fetcher.fetch(source_for())).jobs) == 3


async def test_a_valid_feed_with_no_entries_is_a_success(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("empty.xml"))

    result = await fetcher.fetch(source_for())

    assert (result.fetched, result.jobs) == (0, [])


# --- the URL is checked before the first request -------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/feed",
        "http://127.0.0.1/",
        "http://db/",
        "http://169.254.169.254/",
        "ftp://example.com/x",
    ],
)
async def test_an_unsafe_start_url_is_refused_and_nothing_is_requested(
    fetcher: RssFetcher, feed_server: FakeFeedServer, url: str
) -> None:
    message = await fetch_error(fetcher, source_for(url))

    assert message.startswith("URL not allowed: ")
    assert feed_server.requests == []


async def test_a_start_url_that_resolves_to_a_private_address_is_refused(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    message = await fetch_error(fetcher, source_for("https://internal.example.com/feed"))

    assert message.startswith("URL not allowed: ")
    assert feed_server.requests == []
    assert not any(leak in message for leak in LEAKS)


async def test_the_host_is_resolved_and_checked_again_on_every_fetch(
    fetcher: RssFetcher, feed_server: FakeFeedServer, resolver: FakeResolver
) -> None:
    feed_server.add(URL, BASIC)

    await fetcher.fetch(source_for())
    await fetcher.fetch(source_for())

    assert resolver.calls == ["example.com", "example.com"]  # DNS may change between fetches


# --- redirects: followed by hand, each hop checked -----------------------------------------------


async def test_a_redirect_to_another_public_host_is_followed_and_checked(
    fetcher: RssFetcher, feed_server: FakeFeedServer, resolver: FakeResolver
) -> None:
    feed_server.redirect(URL, "https://feeds.example.org/rss", status=301)
    feed_server.add("https://feeds.example.org/rss", BASIC)

    result = await fetcher.fetch(source_for())

    assert len(result.jobs) == 3
    assert feed_server.urls == [URL, "https://feeds.example.org/rss"]
    assert resolver.calls == ["example.com", "feeds.example.org"]  # the new host was checked too


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_every_redirect_status_is_followed(
    fetcher: RssFetcher, feed_server: FakeFeedServer, status: int
) -> None:
    feed_server.redirect(URL, "https://feeds.example.org/rss", status=status)
    feed_server.add("https://feeds.example.org/rss", BASIC)

    assert len((await fetcher.fetch(source_for())).jobs) == 3


async def test_a_relative_redirect_is_resolved_against_the_current_url(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.redirect(URL, "/v2/feed.xml")
    feed_server.add("https://example.com/v2/feed.xml", BASIC)

    assert len((await fetcher.fetch(source_for())).jobs) == 3


@pytest.mark.parametrize(
    "target",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8000/admin",
        "http://localhost/",
        "http://db/",
        "http://2130706433/",
        "http://[::ffff:127.0.0.1]/",
        "//169.254.169.254/x",  # scheme-relative: inherits https and points at the metadata IP
        "http://internal.example.com/feed",  # a normal-looking host that resolves to 10.0.0.5
        "ftp://example.com/feed",
        "file:///etc/passwd",
        "http://user:pw@example.com/",
        "https://example.com:8443/feed",
    ],
)
async def test_a_redirect_to_a_private_or_unsafe_target_is_blocked_before_any_request(
    fetcher: RssFetcher, feed_server: FakeFeedServer, target: str
) -> None:
    feed_server.redirect(URL, target)

    message = await fetch_error(fetcher)

    assert message.startswith("Redirect not allowed: ")
    assert feed_server.urls == [URL]  # the target was never requested
    assert not any(leak in message for leak in LEAKS)


async def test_three_redirects_are_allowed(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    for i in range(MAX_REDIRECTS):
        feed_server.redirect(f"https://example.com/r{i}", f"https://example.com/r{i + 1}")
    feed_server.add(f"https://example.com/r{MAX_REDIRECTS}", BASIC)

    result = await fetcher.fetch(source_for("https://example.com/r0"))

    assert len(result.jobs) == 3
    assert len(feed_server.requests) == MAX_REDIRECTS + 1


async def test_a_fourth_redirect_is_an_error_and_the_next_url_is_not_requested(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    for i in range(MAX_REDIRECTS + 1):
        feed_server.redirect(f"https://example.com/r{i}", f"https://example.com/r{i + 1}")
    feed_server.add(f"https://example.com/r{MAX_REDIRECTS + 1}", BASIC)

    message = await fetch_error(fetcher, source_for("https://example.com/r0"))

    assert message == "Too many redirects"
    assert len(feed_server.requests) == MAX_REDIRECTS + 1  # r0..r3; r4 was never fetched


async def test_a_redirect_loop_is_an_error(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.redirect(URL, URL)

    assert await fetch_error(fetcher) == "Too many redirects"


async def test_a_redirect_without_a_location_is_an_http_error(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, status=302)

    assert await fetch_error(fetcher) == "HTTP 302"


# --- failures are short, sanitized messages ------------------------------------------------------


@pytest.mark.parametrize("status", [400, 403, 404, 410, 429, 500, 502, 503])
async def test_http_error_statuses_are_reported_by_code(
    fetcher: RssFetcher, feed_server: FakeFeedServer, status: int
) -> None:
    feed_server.add(URL, b"<html>error page with details</html>", status=status)

    assert await fetch_error(fetcher) == f"HTTP {status}"


@pytest.mark.parametrize(
    "exception",
    [httpx.ReadTimeout("read"), httpx.ConnectTimeout("connect"), httpx.PoolTimeout("pool")],
)
async def test_timeouts_are_reported_without_details(
    fetcher: RssFetcher, feed_server: FakeFeedServer, exception: Exception
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    feed_server.add_handler(URL, handler)

    assert await fetch_error(fetcher) == "Timed out"


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectError("connect to 10.0.0.5:5432 failed (internal-db.corp)"),
        httpx.ReadError("connection reset by 10.0.0.5"),
        httpx.RemoteProtocolError("bad response from internal-db.corp"),
        httpx.UnsupportedProtocol("scheme gopher"),
    ],
)
async def test_connection_errors_hide_hostnames_and_addresses(
    fetcher: RssFetcher, feed_server: FakeFeedServer, exception: Exception
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    feed_server.add_handler(URL, handler)

    message = await fetch_error(fetcher)

    assert message == "Could not connect to the server"
    assert "10.0.0.5" not in message and "internal-db" not in message


async def test_an_undecodable_response_is_reported(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("bad gzip data")

    feed_server.add_handler(URL, handler)

    assert await fetch_error(fetcher) == "Response could not be decoded"


async def test_a_response_that_takes_too_long_overall_is_cut_off(
    fetcher: RssFetcher, feed_server: FakeFeedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.ingestion.rss.OVERALL_DEADLINE_SECONDS", 0.05)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)  # each read stays under its own timeout, but the total does not
        return httpx.Response(200, content=BASIC)

    feed_server.add_handler(URL, slow)
    started = time.perf_counter()

    message = await fetch_error(fetcher)

    assert message == "Timed out"
    assert time.perf_counter() - started < 2


# --- size limits ---------------------------------------------------------------------------------


async def test_a_declared_oversize_body_is_refused(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, BASIC, headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)})

    assert await fetch_error(fetcher) == "Response too large"


async def test_a_streamed_body_is_cut_off_at_the_cap_without_reading_the_rest(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    chunk, total_chunks = 64 * 1024, 400  # 25 MB offered, no Content-Length
    sent = 0

    async def body() -> Any:
        nonlocal sent
        for _ in range(total_chunks):
            sent += 1
            yield b"x" * chunk

    feed_server.add_handler(URL, lambda request: httpx.Response(200, content=body()))

    assert await fetch_error(fetcher) == "Response too large"
    assert sent <= MAX_RESPONSE_BYTES // chunk + 2  # it stopped reading soon after the cap


async def test_a_compression_bomb_is_stopped_by_the_decoded_size(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    bomb = gzip.compress(b"0" * (MAX_RESPONSE_BYTES * 5))  # a few KB on the wire
    assert len(bomb) < 50_000
    feed_server.add(URL, bomb, headers={"Content-Encoding": "gzip"})

    assert await fetch_error(fetcher) == "Response too large"


# --- what comes back is not a usable feed --------------------------------------------------------


@pytest.mark.parametrize("name", ["malformed.xml", "not_a_feed.html"])
async def test_a_response_that_is_not_a_feed_is_reported(
    fetcher: RssFetcher, feed_server: FakeFeedServer, name: str
) -> None:
    feed_server.add(URL, feed_bytes(name))

    assert await fetch_error(fetcher) == "Not a valid RSS or Atom feed"


async def test_only_the_first_max_entries_are_taken(
    fetcher: RssFetcher, feed_server: FakeFeedServer
) -> None:
    items = "".join(
        f"<item><title>Job {i}</title><guid>g{i}</guid><description>x</description></item>"
        for i in range(MAX_ENTRIES + 80)
    )
    feed_server.add(
        URL,
        f'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>{items}</channel></rss>',
    )

    result = await fetcher.fetch(source_for())

    assert (result.fetched, len(result.jobs)) == (MAX_ENTRIES, MAX_ENTRIES)


# --- how the HTTP client is configured -----------------------------------------------------------


async def test_the_client_never_follows_redirects_itself_or_trusts_the_environment(
    fetcher: RssFetcher, feed_server: FakeFeedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class SpyClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", SpyClient)
    feed_server.add(URL, BASIC)

    await fetcher.fetch(source_for())

    assert captured["follow_redirects"] is False  # we follow redirects ourselves, checking each
    assert captured["trust_env"] is False  # HTTP(S)_PROXY must not route around the checks
    timeout = captured["timeout"]
    assert (timeout.connect, timeout.read) == (5.0, 10.0)
