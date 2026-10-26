from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Explicit test settings: no env vars or .env file needed."""
    return Settings(
        _env_file=None,
        env="test",
        database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/jobradar_test",
        redis_url="redis://localhost:6379/1",
        secret_key=SecretStr("test-secret-key"),
        cors_origins="http://localhost:3000",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An HTTP client that calls the app in-process (no server, no network)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
