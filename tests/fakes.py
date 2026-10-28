"""Test doubles that keep tests off the network."""

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from tests.network_guard import UNREACHABLE_PORT

FEEDS_DIR = Path(__file__).parent / "fixtures" / "feeds"

# Where tests point the app's Redis/broker setting. Nothing listens here, on purpose: tests must
# never depend on, or touch, a real Redis (this machine, and CI, may have one on 6379).
UNREACHABLE_REDIS_URL = f"redis://127.0.0.1:{UNREACHABLE_PORT}/0"


def feed_bytes(name: str) -> bytes:
    """A recorded feed from tests/fixtures/feeds."""
    return (FEEDS_DIR / name).read_bytes()


Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


class FakeFeedServer:
    """An in-memory web server for httpx: URLs map to responses, and every request is recorded.

    Use `.transport` as the transport of an httpx client; no socket is ever opened.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: dict[str, Handler] = {}
        self.transport = httpx.MockTransport(self._handle)

    def add(
        self,
        url: str,
        content: bytes | str = b"",
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.add_handler(
            url, lambda request: httpx.Response(status, content=content, headers=headers)
        )

    def redirect(self, url: str, to: str, status: int = 302) -> None:
        self.add(url, status=status, headers={"Location": to})

    def add_handler(self, url: str, handler: Handler) -> None:
        self._routes[str(httpx.URL(url))] = handler  # httpx normalizes ("https://a.com" -> ".../")

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self._routes.get(str(request.url))
        if handler is None:
            return httpx.Response(404)
        response = handler(request)
        return await response if isinstance(response, Awaitable) else response

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


class FakeTaskQueue:
    """Stands in for the Celery queue in API tests: nothing is published anywhere."""

    def __init__(self) -> None:
        self.enqueued: list[uuid.UUID] = []
        self.syncing: set[uuid.UUID] = set()  # sources that "have a sync in progress"
        self.is_syncing_calls: list[uuid.UUID] = []
        self.fail_with: Exception | None = None  # set to simulate Redis being down

    async def is_syncing(self, source_id: uuid.UUID) -> bool:
        self.is_syncing_calls.append(source_id)
        if self.fail_with:
            raise self.fail_with
        return source_id in self.syncing

    async def enqueue_sync(self, source_id: uuid.UUID) -> None:
        if self.fail_with:
            raise self.fail_with
        self.enqueued.append(source_id)


class FakeSourceLocks:
    """In-memory SourceLocks that records every acquire and release."""

    def __init__(self) -> None:
        self.held: set[uuid.UUID] = set()
        self.events: list[tuple[str, uuid.UUID]] = []

    def acquire(self, source_id: uuid.UUID) -> bool:
        self.events.append(("acquire", source_id))
        if source_id in self.held:
            return False
        self.held.add(source_id)
        return True

    def release(self, source_id: uuid.UUID) -> None:
        self.events.append(("release", source_id))
        self.held.discard(source_id)

    def is_locked(self, source_id: uuid.UUID) -> bool:
        return source_id in self.held


# Pretend DNS zone. The addresses mirror what the real names would resolve to.
DEFAULT_DNS_RECORDS: dict[str, list[str]] = {
    "example.com": ["93.184.216.34"],
    "feeds.example.org": ["93.184.216.35", "2606:2800:220:1:248:1893:25c8:1946"],
    "internal.example.com": ["10.0.0.5"],  # a public-looking name pointing at a private IP
    "metadata.example.com": ["169.254.169.254"],
    "loopback.example.com": ["127.0.0.1"],
    "rebind.example.com": ["93.184.216.34", "10.0.0.5"],  # one good record, one bad
    "ipv6-loopback.example.com": ["::1"],
    "mapped.example.com": ["::ffff:10.0.0.5"],
    "nat64.example.com": ["64:ff9b::7f00:1"],  # NAT64 wrapping 127.0.0.1
    "sixtofour.example.com": ["2002:7f00:1::"],  # 6to4 wrapping 127.0.0.1
    "ula.example.com": ["fc00::1"],
    "linklocal.example.com": ["fe80::1"],
    "multicast.example.com": ["224.0.0.1"],
    "cgnat.example.com": ["100.64.0.1"],
    "empty.example.com": [],
    # What Docker's DNS would answer for the Compose service names:
    "db": ["172.21.0.2"],
    "redis": ["172.21.0.3"],
}


class FakeResolver:
    """A DNS resolver that answers from a dict and records which names were asked for."""

    def __init__(self, records: dict[str, list[str]] | None = None) -> None:
        self.records = DEFAULT_DNS_RECORDS | (records or {})
        self.calls: list[str] = []

    async def __call__(self, host: str) -> list[str]:
        self.calls.append(host)
        if host not in self.records:
            raise OSError(f"cannot resolve {host}")  # what a failed DNS lookup raises
        return self.records[host]
