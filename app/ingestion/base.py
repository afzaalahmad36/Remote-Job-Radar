from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.models import Source


class FetchError(Exception):
    """A source could not be fetched. The message is stored in `sources.last_error` and shown to
    the user, so it must be SHORT and SANITIZED: no resolved IPs, internal hostnames, URLs with
    credentials or stack traces. Fetchers build these messages themselves from fixed text."""


@dataclass(frozen=True)
class RawJob:
    """One post from a source, before LLM extraction (Phase 3 fills in the structured fields)."""

    external_id: str  # stable id of the post within its source: the dedupe key
    raw_text: str  # plain text only, never HTML
    posted_at: datetime | None  # timezone-aware UTC, or None if the source gave no date


@dataclass(frozen=True)
class FetchResult:
    # Valid posts, already deduped within the fetch.
    jobs: list[RawJob] = field(default_factory=list)
    fetched: int = 0  # how many posts the source returned (before dedupe and validation)
    not_modified: bool = False  # the source said nothing changed since last time (HTTP 304)
    etag: str | None = None
    last_modified: str | None = None


class SourceFetcher(Protocol):
    """What every kind of source (RSS now, Telegram later) implements.

    `fetch` returns the source's posts or raises FetchError. It must not touch the database:
    storing the posts is the ingestion service's job, the same for every source type.
    """

    async def fetch(self, source: Source) -> FetchResult: ...
