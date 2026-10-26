import pytest

from app.core.exceptions import RateLimitError
from app.core.rate_limit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_the_limit_then_blocks() -> None:
    limiter = RateLimiter(max_attempts=3, window_seconds=60, clock=FakeClock())

    for _ in range(3):
        limiter.check("login:1.2.3.4")

    with pytest.raises(RateLimitError):
        limiter.check("login:1.2.3.4")


def test_blocked_error_says_when_to_retry() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_attempts=1, window_seconds=60, clock=clock)
    limiter.check("k")
    clock.now += 20

    with pytest.raises(RateLimitError) as error:
        limiter.check("k")

    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "40"}


def test_attempts_are_allowed_again_after_the_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_attempts=1, window_seconds=60, clock=clock)
    limiter.check("k")
    clock.now += 61

    limiter.check("k")  # no error


def test_keys_are_independent() -> None:
    limiter = RateLimiter(max_attempts=1, window_seconds=60, clock=FakeClock())
    limiter.check("login:1.1.1.1")

    limiter.check("login:2.2.2.2")  # different client
    limiter.check("register:1.1.1.1")  # different endpoint


def test_idle_keys_are_swept_when_many_are_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.rate_limit._SWEEP_THRESHOLD", 5)
    clock = FakeClock()
    limiter = RateLimiter(max_attempts=1, window_seconds=60, clock=clock)
    for i in range(5):
        limiter.check(f"key-{i}")
    clock.now += 61

    limiter.check("fresh")

    assert list(limiter._hits) == ["fresh"]
