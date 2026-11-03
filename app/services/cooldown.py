"""A shared pause after the provider says "not now".

After a 429 (or an outage) every worker should stop asking for a while, not just the one that
saw it. The pause lives in Redis so all workers share it. This module is the interface and the
durations; `app/tasks/cooldown.py` is the Redis implementation and `tests/fakes.py` has a fake.

If the pause cannot be read (Redis is down) the answer is "don't ask": see `CooldownUnavailable`.
"""

from typing import Protocol

# How long to pause, in seconds.
RATE_LIMIT_DEFAULT_SECONDS = 60.0  # a 429 without a Retry-After header
RATE_LIMIT_MIN_SECONDS = 30.0  # a tiny Retry-After is not trusted: it would invite a hammering
RATE_LIMIT_MAX_SECONDS = 900.0  # a huge one is capped, so one bad header cannot stall for a day
TRANSIENT_SECONDS = 30.0  # a 5xx, a timeout or a dropped connection
# A configuration failure (bad key, no access, unknown model): every request would fail alike, so
# pause for the longest time we allow.
CONFIGURATION_FAILURE_SECONDS = 900.0
# Told to callers when Redis is down. A floor, so nothing that honours it can retry in a tight loop.
UNAVAILABLE_RETRY_SECONDS = 60.0


class CooldownUnavailable(Exception):
    """The shared pause could not be read or written (Redis is down or too slow).

    Callers treat this as "do not ask the model now": the rules still work without Redis, so
    jobs are still extracted, they just are not enriched until the store is back.
    """


class Cooldown(Protocol):
    async def remaining(self) -> float | None:
        """Seconds left of the pause, or None if there is none. Raises CooldownUnavailable."""
        ...

    async def start(self, seconds: float) -> None:
        """Pause for at least `seconds` (a longer pause is kept). Raises CooldownUnavailable."""
        ...


def rate_limit_pause(retry_after_seconds: float | None) -> float:
    """The pause for a 429: the provider's Retry-After if given, kept between the min and max."""
    wanted = RATE_LIMIT_DEFAULT_SECONDS if retry_after_seconds is None else retry_after_seconds
    return min(max(wanted, RATE_LIMIT_MIN_SECONDS), RATE_LIMIT_MAX_SECONDS)
