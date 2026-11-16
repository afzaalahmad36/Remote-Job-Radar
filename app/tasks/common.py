"""What the task modules share: database sessions for a task, and which errors to retry.

Kept apart from `jobs.py` so that `embedding.py` can use it without importing the extraction
tasks (and, through them, the LLM client): the embedding path must not reach a model or an HTTP
client (a test checks it).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from kombu.exceptions import KombuError
from redis.exceptions import RedisError
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings

# What publishing a task can raise when Redis is unreachable. A failed publish is logged, never
# raised into the caller: ingestion must not fail because extraction could not be queued, since the
# sweep queues whatever was missed.
ENQUEUE_ERRORS = (KombuError, RedisError, OSError)

# Failures worth retrying: Redis or the database being briefly unreachable. Anything else (a bug)
# is not retried, because running it again would just fail the same way.
RETRYABLE_ERRORS = (RedisError, OperationalError, InterfaceError)

SessionFactory = async_sessionmaker[AsyncSession]


@asynccontextmanager
async def session_factory(settings: Settings) -> AsyncIterator[SessionFactory]:
    """A session factory on a brand-new, throwaway engine.

    A Celery task is a normal function, so each run starts its own event loop with
    `asyncio.run`, and a loop is gone when `run` returns. The API's engine keeps open
    connections that belong to *its* loop, so a task must not use it. NullPool means no
    connection outlives the task, and `dispose()` closes the engine before the loop ends.
    """
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
