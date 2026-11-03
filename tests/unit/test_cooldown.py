import time
from typing import Any

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.services.cooldown import (
    RATE_LIMIT_DEFAULT_SECONDS,
    RATE_LIMIT_MAX_SECONDS,
    RATE_LIMIT_MIN_SECONDS,
    CooldownUnavailable,
    rate_limit_pause,
)
from app.tasks.cooldown import KEY_PREFIX, RedisCooldown, redis_cooldown_for
from tests.fakes import UNREACHABLE_REDIS_URL, FakeCooldown


@pytest.mark.parametrize(
    ("retry_after", "pause"),
    [
        (None, RATE_LIMIT_DEFAULT_SECONDS),  # no header: the default
        (0.0, RATE_LIMIT_MIN_SECONDS),  # a tiny value is not trusted
        (5.0, RATE_LIMIT_MIN_SECONDS),
        (30.0, 30.0),
        (120.0, 120.0),
        (900.0, 900.0),
        (3600.0, RATE_LIMIT_MAX_SECONDS),  # a huge value is capped
    ],
)
def test_the_pause_after_a_429_follows_retry_after_within_limits(
    retry_after: float | None, pause: float
) -> None:
    assert rate_limit_pause(retry_after) == pause


class StubRedis:
    """The two Redis calls the cooldown makes, with a stored expiry in milliseconds."""

    def __init__(self, ttl_ms: int = -2, error: Exception | None = None, delay: float = 0) -> None:
        self.ttl_ms = ttl_ms
        self.error = error
        self.delay = delay
        self.sets: list[tuple[str, str, int]] = []

    def pttl(self, key: str) -> int:
        self._maybe_fail()
        return self.ttl_ms

    def set(self, key: str, value: str, px: int) -> None:
        self._maybe_fail()
        self.sets.append((key, value, px))
        self.ttl_ms = px

    def _maybe_fail(self) -> None:
        time.sleep(self.delay)
        if self.error:
            raise self.error


def cooldown_on(stub: StubRedis, call_timeout: float = 2.0) -> RedisCooldown:
    client: Any = stub
    return RedisCooldown(client, "gemini", call_timeout=call_timeout)


async def test_no_key_means_no_pause() -> None:
    assert await cooldown_on(StubRedis(ttl_ms=-2)).remaining() is None  # Redis: key missing
    assert await cooldown_on(StubRedis(ttl_ms=-1)).remaining() is None  # Redis: no expiry set


async def test_a_running_pause_reports_the_seconds_left() -> None:
    assert await cooldown_on(StubRedis(ttl_ms=42_500)).remaining() == 42.5


async def test_starting_a_pause_sets_one_key_per_provider_with_an_expiry() -> None:
    stub = StubRedis()

    await cooldown_on(stub).start(30.0)

    assert stub.sets == [(f"{KEY_PREFIX}gemini", "1", 30_000)]
    assert KEY_PREFIX == "llm:cooldown:"


async def test_a_shorter_pause_never_replaces_a_longer_one_but_a_longer_one_does() -> None:
    stub = StubRedis(ttl_ms=60_000)
    cooldown = cooldown_on(stub)

    await cooldown.start(30.0)
    assert stub.sets == []

    await cooldown.start(90.0)
    assert stub.sets == [(f"{KEY_PREFIX}gemini", "1", 90_000)]


async def test_a_pause_is_at_least_one_millisecond() -> None:
    stub = StubRedis()

    await cooldown_on(stub).start(0.0)

    assert stub.sets[0][2] == 1


async def test_a_redis_error_means_unavailable_never_no_pause() -> None:
    broken = StubRedis(error=RedisConnectionError("down"))

    with pytest.raises(CooldownUnavailable):
        await cooldown_on(broken).remaining()
    with pytest.raises(CooldownUnavailable):
        await cooldown_on(broken).start(30.0)


async def test_a_redis_that_does_not_answer_in_time_means_unavailable() -> None:
    slow = StubRedis(delay=0.3)

    with pytest.raises(CooldownUnavailable):
        await cooldown_on(slow, call_timeout=0.05).remaining()


async def test_a_real_client_on_a_dead_port_fails_fast_as_unavailable() -> None:
    client = Redis.from_url(UNREACHABLE_REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
    started = time.perf_counter()

    with pytest.raises(CooldownUnavailable):
        await RedisCooldown(client, "gemini").remaining()

    assert time.perf_counter() - started < 15


def test_one_cooldown_client_is_shared_per_url_and_provider() -> None:
    redis_cooldown_for.cache_clear()

    first = redis_cooldown_for(UNREACHABLE_REDIS_URL, "gemini")

    assert redis_cooldown_for(UNREACHABLE_REDIS_URL, "gemini") is first
    assert redis_cooldown_for(UNREACHABLE_REDIS_URL, "ollama") is not first
    redis_cooldown_for.cache_clear()


async def test_the_fake_follows_the_same_rules() -> None:
    fake = FakeCooldown(seconds_left=10.0)

    await fake.start(5.0)
    assert await fake.remaining() == 10.0  # a longer pause is kept
    await fake.start(20.0)
    assert (await fake.remaining(), fake.started) == (20.0, [5.0, 20.0])

    for broken in (FakeCooldown(unavailable=True), FakeCooldown(fail_start=True)):
        with pytest.raises(CooldownUnavailable):
            await broken.start(1.0)
    with pytest.raises(CooldownUnavailable):
        await FakeCooldown(unavailable=True).remaining()
