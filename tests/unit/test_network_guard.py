import socket

import pytest
import redis

from tests.network_guard import (
    MAX_ATTEMPTS_PER_TEST_ON_UNREACHABLE_PORT,
    UNREACHABLE_PORT,
    NetworkAccessBlockedError,
)


@pytest.mark.parametrize("port", [6379, 6380, 5432, 27017, 8000])
def test_connecting_to_well_known_service_ports_on_loopback_is_blocked(port: int) -> None:
    with pytest.raises(NetworkAccessBlockedError, match=str(port)):
        socket.create_connection(("127.0.0.1", port), timeout=1)


@pytest.mark.parametrize("host", ["93.184.216.34", "10.0.0.5", "169.254.169.254"])
def test_connecting_to_anything_that_is_not_loopback_is_blocked(host: str) -> None:
    with pytest.raises(NetworkAccessBlockedError):
        socket.create_connection((host, 443), timeout=1)


def test_the_error_names_the_test_that_tried_it() -> None:
    with pytest.raises(NetworkAccessBlockedError, match="test_the_error_names_the_test"):
        socket.create_connection(("127.0.0.1", 6379), timeout=1)


def test_a_real_redis_client_is_stopped_at_once_instead_of_retrying() -> None:
    # redis-py retries OSError; the guard's error is not one, so it surfaces immediately. (This
    # is the machine's real Redis port: the connection never leaves the guard.)
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")

    with pytest.raises(NetworkAccessBlockedError):
        client.ping()


def test_the_unreachable_port_is_allowed_and_simply_refuses() -> None:
    # Nothing listens there, so the connection is refused: an ordinary OSError, not the guard.
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", UNREACHABLE_PORT), timeout=3)


def test_a_retry_loop_against_the_dead_port_is_turned_into_an_error() -> None:
    # Celery's beat scheduler once did this (up to 100 retries with growing pauses) and hung CI.
    with pytest.raises(NetworkAccessBlockedError, match="retrying"):
        for _ in range(MAX_ATTEMPTS_PER_TEST_ON_UNREACHABLE_PORT + 5):
            try:
                socket.create_connection(("127.0.0.1", UNREACHABLE_PORT), timeout=1)
            except OSError:
                continue  # refused, as expected; a retry loop would simply try again


def test_a_few_attempts_against_the_dead_port_are_fine() -> None:
    for _ in range(3):  # what a bounded publish retry does
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", UNREACHABLE_PORT), timeout=1)


def test_a_throwaway_local_server_on_an_ephemeral_port_is_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass  # connected: no error


def test_the_guard_is_an_assertion_error_not_an_oserror() -> None:
    assert issubclass(NetworkAccessBlockedError, AssertionError)
    assert not issubclass(NetworkAccessBlockedError, OSError)  # or retry loops would swallow it
