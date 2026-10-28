"""Make "tests never touch a real network service" an enforced rule instead of a habit.

While the test session runs, opening a socket connection is only allowed to:
  * the test database (the host and port in TEST_DATABASE_URL), and
  * loopback addresses on high ports (>= 32768), which is where a test's own throwaway local
    server lives (a socket bound to port 0 gets an ephemeral port), and
  * loopback port 6399, where nothing listens (connecting there can only be refused).
Anything else (a real Redis on 6379, a broker, the internet, another project's database) raises
NetworkAccessBlockedError immediately, naming the test and the address.

The error deliberately derives from AssertionError, NOT OSError: Redis, kombu and httpx catch
OSError and retry, so an OSError would be swallowed and turn into a silent hang, which is exactly
what this guard exists to prevent.

Limit: it patches `socket.socket.connect`. On Linux that covers every client library, asyncpg
included. On Windows the default asyncio loop connects without it, so only the synchronous
clients (redis-py, kombu, sockets made by hand) are covered there; CI runs on Linux.
"""

import os
import socket
from collections.abc import Callable
from typing import Any

from sqlalchemy.engine import make_url

from tests.db_utils import DEFAULT_TEST_DATABASE_URL

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
EPHEMERAL_PORT_START = 32768
# A loopback port where nothing listens. Tests point the app's "Redis" here to prove that an
# unreachable Redis fails fast; connecting there can only ever be refused, so it is allowed.
UNREACHABLE_PORT = 6399
# A fail-fast test connects there once (measured; three would still be fine). More than that
# means something is retrying in a loop, which is how the beat tests once hung CI for hours
# (Celery retries a broker connection up to 100 times, pausing longer each time, and the pauses
# are why the limit must be this low to trip within seconds). So that is turned into an error.
MAX_ATTEMPTS_PER_TEST_ON_UNREACHABLE_PORT = 3


class NetworkAccessBlockedError(AssertionError):
    """A test tried to open a network connection it is not allowed to."""


def _is_allowed(host: object, port: object, database: set[tuple[str, int]]) -> bool:
    if not isinstance(host, str) or not isinstance(port, int):
        return True  # not an IP socket address (for example a Unix socket path)
    if host in LOOPBACK and (port >= EPHEMERAL_PORT_START or port == UNREACHABLE_PORT):
        return True  # a test's own local server, or the port where nothing listens
    return (host, port) in database


def _database_addresses(url: str) -> set[tuple[str, int]]:
    """Every (address, port) the test database can be reached at, names resolved to IPs."""
    parsed = make_url(url)
    host, port = parsed.host or "localhost", parsed.port or 5432
    addresses = {(host, port)}
    try:  # resolving a name opens no connection
        addresses |= {(str(info[4][0]), port) for info in socket.getaddrinfo(host, port)}
    except OSError:
        pass  # unresolvable: then only the literal host can match, and the DB tests will fail
    return addresses


def install() -> Callable[[], None]:
    """Start blocking; returns a function that stops it again."""
    database = _database_addresses(os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL))
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    unreachable_attempts: dict[str, int] = {}

    def check(address: Any) -> None:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            test = os.environ.get("PYTEST_CURRENT_TEST", "(outside a test)")
            if not _is_allowed(host, port, database):
                raise NetworkAccessBlockedError(
                    f"Tests must not open network connections, but {test} tried {host}:{port}. "
                    "Use a fake (see tests/fakes.py)."
                )
            if port == UNREACHABLE_PORT and host in LOOPBACK:
                unreachable_attempts[test] = unreachable_attempts.get(test, 0) + 1
                if unreachable_attempts[test] > MAX_ATTEMPTS_PER_TEST_ON_UNREACHABLE_PORT:
                    raise NetworkAccessBlockedError(
                        f"{test} keeps retrying a connection to the dead port {UNREACHABLE_PORT} "
                        f"({unreachable_attempts[test]} attempts): a retry loop that would hang "
                        "forever with no server. Fake the broker or lock instead."
                    )

    def guarded_connect(self: socket.socket, address: Any) -> None:
        check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        check(address)
        return real_connect_ex(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]

    def uninstall() -> None:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]

    return uninstall
