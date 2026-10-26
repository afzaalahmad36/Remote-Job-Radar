from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """The async engine: owns the connection pool. One per process, created on first use.

    "Async" means queries `await` instead of blocking the event loop; it needs an async driver,
    hence the `postgresql+asyncpg://` URL.
    """
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False keeps loaded attributes readable after commit; with an async session
    # a lazy refresh would need an `await`, which plain attribute access cannot do.
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session (one unit of work) per request.

    Services receive the session and decide when to `commit()`. Leaving the `async with`
    closes the session and returns its connection to the pool.
    """
    async with get_sessionmaker()() as session:
        yield session
