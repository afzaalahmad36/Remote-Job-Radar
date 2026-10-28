from collections.abc import Iterator

import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine, get_session, get_sessionmaker
from tests.fakes import UNREACHABLE_REDIS_URL


@pytest.fixture
def db_env(monkeypatch: pytest.MonkeyPatch, test_database_url: str) -> Iterator[None]:
    """Point the app's own settings at the test database, and reset the cached engine after."""
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("REDIS_URL", UNREACHABLE_REDIS_URL)
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()
    yield
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.mark.usefixtures("db_env")
async def test_get_session_yields_a_working_session() -> None:
    async for session in get_session():
        assert await session.scalar(text("SELECT 1")) == 1

    await get_engine().dispose()
