import math
import time
from collections import deque
from collections.abc import Callable

from app.core.exceptions import RateLimitError

_SWEEP_THRESHOLD = 10_000  # when this many keys are tracked, drop the idle ones


class RateLimiter:
    """Sliding-window limiter: at most `max_attempts` calls per key within `window_seconds`.

    State lives in this process's memory, so it is only correct for a single API instance and
    resets on restart. Moving it to Redis is planned for Phase 7 (hardening).
    """

    def __init__(
        self,
        max_attempts: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = max_attempts
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        """Record one attempt for `key`, or raise RateLimitError if it is over the limit."""
        now = self._clock()
        if len(self._hits) >= _SWEEP_THRESHOLD:
            self._sweep(now)
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self._window:
            hits.popleft()  # forget attempts that have left the window
        if len(hits) >= self._max_attempts:
            retry_after = math.ceil(hits[0] + self._window - now)
            raise RateLimitError(max(retry_after, 1))
        hits.append(now)

    def _sweep(self, now: float) -> None:
        for key in [
            k for k, hits in self._hits.items() if not hits or hits[-1] <= now - self._window
        ]:
            del self._hits[key]
