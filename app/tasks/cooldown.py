import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from redis import Redis
from redis.exceptions import RedisError

from app.services.cooldown import CooldownUnavailable
from app.tasks.celery_app import (
    QUEUE_CALL_TIMEOUT_SECONDS,
    REDIS_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_TIMEOUT_SECONDS,
)

KEY_PREFIX = "llm:cooldown:"


class RedisCooldown:
    """The shared provider pause, stored as one Redis key with an expiry (so it ends by itself).

    redis-py is a blocking library and the service is async, so each call runs in a worker thread
    (`asyncio.to_thread`) under an overall deadline, the same way `CeleryTaskQueue` does. Any
    Redis error or timeout becomes `CooldownUnavailable`; nothing here ever lets a model request
    through when Redis cannot answer.
    """

    def __init__(
        self, client: Redis, provider: str, call_timeout: float = QUEUE_CALL_TIMEOUT_SECONDS
    ) -> None:
        self._client = client
        self._key = f"{KEY_PREFIX}{provider}"
        self._call_timeout = call_timeout

    async def remaining(self) -> float | None:
        milliseconds = await self._call(self._pttl)
        return milliseconds / 1000 if milliseconds > 0 else None  # -2: no key, -1: no expiry

    async def start(self, seconds: float) -> None:
        await self._call(self._extend, max(int(seconds * 1000), 1))

    def _pttl(self) -> int:
        return cast(int, self._client.pttl(self._key))

    def _extend(self, milliseconds: int) -> None:
        if self._pttl() >= milliseconds:
            return  # a longer pause is already running: never shorten it
        self._client.set(self._key, "1", px=milliseconds)

    async def _call(self, function: Callable[..., Any], *args: Any) -> Any:
        try:
            return await asyncio.wait_for(asyncio.to_thread(function, *args), self._call_timeout)
        except (RedisError, TimeoutError) as error:
            raise CooldownUnavailable("the cooldown store is unavailable") from error


@lru_cache
def redis_cooldown_for(redis_url: str, provider: str) -> RedisCooldown:
    """One client (and connection pool) per process and provider, like `redis_locks_for`."""
    client = Redis.from_url(
        redis_url,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
    )
    return RedisCooldown(client, provider)
