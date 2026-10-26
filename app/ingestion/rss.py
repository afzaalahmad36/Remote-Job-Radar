import asyncio
import calendar
import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

import feedparser
import httpx

from app.core.url_safety import Resolver, UnsafeUrlError, system_resolver, validate_public_url
from app.ingestion.base import FetchError, FetchResult, RawJob
from app.ingestion.text import MAX_TEXT_CHARS, html_to_text
from app.models import Source

CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 10.0
OVERALL_DEADLINE_SECONDS = 20.0  # the whole download, however many redirects it takes
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 200
MAX_REDIRECTS = 3
MAX_EXTERNAL_ID_LENGTH = 255  # the jobs.external_id column size
MAX_ETAG_LENGTH = 255  # sources.etag
MAX_LAST_MODIFIED_LENGTH = 100  # sources.last_modified

USER_AGENT = "RemoteJobRadar/0.1 (feed reader for a personal job search)"
ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.5"
)


@dataclass(frozen=True)
class _Download:
    body: bytes = b""
    not_modified: bool = False
    etag: str | None = None
    last_modified: str | None = None


class RssFetcher:
    """Fetches and parses an RSS/Atom feed. We do the HTTP; feedparser only parses the bytes.

    Safety, in order of what happens to a request:
      * the URL passes `url_safety.validate_public_url` before the first request AND before
        every redirect hop (httpx's automatic redirects are off; we follow them ourselves);
      * `trust_env=False`, so proxy settings in the environment cannot route around those checks;
      * connect/read timeouts plus an overall deadline, a cap on the decoded body size and a cap
        on the number of entries, so one slow or huge feed cannot tie up the server;
      * every error becomes a FetchError with fixed, sanitized text (see FetchError).

    Not covered: DNS can answer differently when httpx opens the connection than it did during
    our check (DNS rebinding). Pinning the connection to the checked address is Phase 7 work.
    """

    def __init__(
        self,
        resolver: Resolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._resolver = resolver
        self._transport = transport  # tests pass a MockTransport; None means the real network

    async def fetch(self, source: Source) -> FetchResult:
        url = str(source.config.get("url", ""))
        try:
            async with asyncio.timeout(OVERALL_DEADLINE_SECONDS):
                download = await self._download(url, source.etag, source.last_modified)
        except TimeoutError as error:
            raise FetchError("Timed out") from error
        if download.not_modified:
            return FetchResult(not_modified=True)
        # Parsing is CPU work (up to ~2 MB of XML): a worker thread keeps the event loop free.
        fetched, jobs = await asyncio.to_thread(parse_feed, download.body)
        return FetchResult(
            jobs=jobs, fetched=fetched, etag=download.etag, last_modified=download.last_modified
        )

    async def _download(
        self, start_url: str, etag: str | None, last_modified: str | None
    ) -> _Download:
        headers = {"User-Agent": USER_AGENT, "Accept": ACCEPT}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
            write=READ_TIMEOUT_SECONDS,
            pool=CONNECT_TIMEOUT_SECONDS,
        )
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=timeout, follow_redirects=False, trust_env=False
            ) as client:
                url = start_url
                redirects = 0
                while True:  # ends by returning the body or raising; every hop is checked first
                    await self._check_url(url, is_redirect=redirects > 0)
                    async with client.stream("GET", url, headers=headers) as response:
                        # Not `is_redirect`: that is true for every 3xx, including 304 (which has
                        # no Location) and 3xx responses that forgot the header.
                        if not response.has_redirect_location:
                            return await _read(response)
                        if redirects == MAX_REDIRECTS:
                            raise FetchError("Too many redirects")
                        redirects += 1
                        url = urljoin(url, response.headers["location"])  # may be relative
        except httpx.TimeoutException as error:
            raise FetchError("Timed out") from error
        except httpx.DecodingError as error:
            raise FetchError("Response could not be decoded") from error
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            # No details in the message: they can include hostnames and addresses.
            raise FetchError("Could not connect to the server") from error

    async def _check_url(self, url: str, is_redirect: bool) -> None:
        try:
            await validate_public_url(url, self._resolver)
        except UnsafeUrlError as error:
            # `error.detail` is fixed text from url_safety (it never contains the URL or an IP).
            prefix = "Redirect not allowed" if is_redirect else "URL not allowed"
            raise FetchError(f"{prefix}: {error.detail}") from error


async def _read(response: httpx.Response) -> _Download:
    """Turn the final response into a body, enforcing the size cap while streaming."""
    if response.status_code == 304:  # "not modified": a successful no-op
        return _Download(not_modified=True)
    if not 200 <= response.status_code < 300:
        raise FetchError(f"HTTP {response.status_code}")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
        raise FetchError("Response too large")
    chunks: list[bytes] = []
    size = 0
    # aiter_bytes yields *decoded* bytes, so the cap also stops compressed "bombs" that are tiny
    # on the wire. We stop reading as soon as the cap is passed.
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise FetchError("Response too large")
        chunks.append(chunk)
    return _Download(
        body=b"".join(chunks),
        etag=_validator(response.headers.get("etag"), MAX_ETAG_LENGTH),
        last_modified=_validator(response.headers.get("last-modified"), MAX_LAST_MODIFIED_LENGTH),
    )


def _validator(value: str | None, max_length: int) -> str | None:
    """Keep an ETag / Last-Modified value only if it fits its column and is plain printable ASCII
    (it is sent back verbatim in a request header next time)."""
    if value and len(value) <= max_length and value.isascii() and value.isprintable():
        return value
    return None


def parse_feed(body: bytes) -> tuple[int, list[RawJob]]:
    """Parse feed bytes into (entries seen, valid unique posts). Raises FetchError if not a feed."""
    # Pass a file-like object, never the bytes or a string: feedparser.parse() treats a str with an
    # http/ftp/file scheme as a URL to fetch itself, and any other value as a possible local file
    # path to open, either of which would bypass our SSRF checks.
    parsed = feedparser.parse(io.BytesIO(body))
    entries = parsed.get("entries", [])[:MAX_ENTRIES]
    # `version` is missing (not just empty) when the input was empty or unreadable.
    if not entries and (parsed.get("bozo") or not parsed.get("version")):
        raise FetchError("Not a valid RSS or Atom feed")  # malformed XML, or not a feed at all
    unique: dict[str, RawJob] = {}
    for entry in entries:
        job = _entry_to_job(entry)
        if job is not None:
            unique.setdefault(job.external_id, job)  # a repeated id inside one feed keeps the first
    return len(entries), list(unique.values())


def _entry_to_job(entry: Any) -> RawJob | None:
    title = html_to_text(str(entry.get("title") or ""))
    body = html_to_text(_entry_html(entry))
    raw_text = f"{title}\n\n{body}" if title and body else title or body
    if not raw_text:
        return None  # nothing to store
    return RawJob(
        external_id=_external_id(entry),
        raw_text=raw_text[:MAX_TEXT_CHARS],
        posted_at=_posted_at(entry),
    )


def _entry_html(entry: Any) -> str:
    content = entry.get("content")  # Atom <content> and RSS <content:encoded>
    if content:
        return str(content[0].get("value") or "")
    return str(entry.get("summary") or "")  # RSS <description> and Atom <summary>


def _external_id(entry: Any) -> str:
    """The post's stable id: its GUID, else its link, else a hash of title + link + date."""
    candidate = _clean(entry.get("id")) or _clean(entry.get("link"))
    if candidate is None:
        parts = (
            _clean(entry.get("title")),
            "",
            _clean(_date_field(entry, "published") or _date_field(entry, "updated")),
        )
        candidate = "\n".join(part or "" for part in parts)
        return _sha256(candidate)
    if len(candidate) > MAX_EXTERNAL_ID_LENGTH:  # does not fit the column: use its hash instead
        return _sha256(candidate)
    return candidate


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _date_field(entry: Any, key: str) -> Any:
    """Read a date key exactly as stored. feedparser's own `.get()` silently falls back from a
    missing `updated_parsed` to `published_parsed` (and warns about it); we choose explicitly."""
    return dict.get(entry, key)


def _posted_at(entry: Any) -> datetime | None:
    # feedparser has already converted the feed's date (with its own timezone offset) to UTC.
    parsed = _date_field(entry, "published_parsed") or _date_field(entry, "updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
    except (OverflowError, ValueError, OSError, TypeError):
        return None  # an impossible date in the feed
