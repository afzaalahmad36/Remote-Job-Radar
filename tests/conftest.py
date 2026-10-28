from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api_deps import get_resolver, get_task_queue
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.ingestion.base import SourceFetcher
from app.ingestion.rss import RssFetcher
from app.main import create_app
from app.models import SourceType
from tests import network_guard
from tests.db_utils import UnsafeDatabaseError, alembic_config, get_test_database_url
from tests.fakes import UNREACHABLE_REDIS_URL, FakeFeedServer, FakeResolver, FakeTaskQueue
from tests.task_helpers import celery_env, seed  # noqa: F401  (fixtures, registered for all tests)


@pytest.fixture(scope="session", autouse=True)
def _block_real_network() -> Iterator[None]:
    """Every test runs with outbound connections blocked (except the test DB): see the module."""
    uninstall = network_guard.install()
    yield
    uninstall()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The only database tests may use. Aborts the whole run if it isn't `jobradar_test`."""
    try:
        return get_test_database_url()
    except UnsafeDatabaseError as error:
        pytest.exit(str(error), returncode=2)


@pytest.fixture
def settings(test_database_url: str) -> Settings:
    """Explicit test settings: no env vars or .env file needed."""
    return Settings(
        _env_file=None,
        env="test",
        database_url=test_database_url,
        redis_url=UNREACHABLE_REDIS_URL,
        secret_key=SecretStr("test-secret-key-at-least-32-characters-long"),
        cors_origins="http://localhost:3000",
    )


@pytest.fixture
def resolver() -> FakeResolver:
    """DNS for tests: a fixed zone, never the network. Tests can inspect `resolver.calls`."""
    return FakeResolver()


@pytest.fixture
def feed_server() -> FakeFeedServer:
    """The "internet" feeds are fetched from: tests register URLs on it and inspect requests."""
    return FakeFeedServer()


@pytest.fixture
def fetchers(
    resolver: FakeResolver, feed_server: FakeFeedServer
) -> dict[SourceType, SourceFetcher]:
    return {SourceType.RSS: RssFetcher(resolver=resolver, transport=feed_server.transport)}


@pytest.fixture
def task_queue() -> FakeTaskQueue:
    """The background-job queue for API tests: records what was queued, publishes nothing."""
    return FakeTaskQueue()


@pytest.fixture
def app(settings: Settings, resolver: FakeResolver, task_queue: FakeTaskQueue) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_resolver] = lambda: resolver
    app.dependency_overrides[get_task_queue] = lambda: task_queue
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An HTTP client that calls the app in-process (no server, no network)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def db_client(app: FastAPI, db_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    """Like `client`, but the API uses the rollback-per-test session, so its data disappears."""

    async def override_get_session() -> AsyncSession:
        return db_session

    app.dependency_overrides[get_session] = override_get_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
def migrated_db(test_database_url: str) -> None:
    """Bring the test database to the latest schema once per run, using the real migrations."""
    command.upgrade(alembic_config(test_database_url), "head")


@pytest.fixture
async def db_session(test_database_url: str, migrated_db: None) -> AsyncIterator[AsyncSession]:
    """A session whose work is rolled back after each test, so tests never leave data behind.

    The session joins an outer transaction that is never committed. With
    `create_savepoint`, a `session.commit()` inside a test only releases a savepoint, and the
    outer transaction is rolled back at the end.
    """
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        outer = await connection.begin()
        session = AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await outer.rollback()
    await engine.dispose()
