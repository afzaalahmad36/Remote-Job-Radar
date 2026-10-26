import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def make_settings(**overrides: str) -> Settings:
    # _env_file=None ignores any local .env so the test only sees what it passes in.
    values = {
        "database_url": "postgresql+asyncpg://u:p@db:5432/x",
        "redis_url": "redis://redis:6379/0",
        "secret_key": "s3cret",
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[arg-type]


def test_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/jobradar")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("SECRET_KEY", "s3cret")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.env == "production"
    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/jobradar"
    assert settings.secret_key.get_secret_value() == "s3cret"


def test_get_settings_is_built_once_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    get_settings.cache_clear()  # drop any value cached by an earlier test

    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_missing_required_setting_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "REDIS_URL", "SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_secret_key_is_hidden_in_repr() -> None:
    assert "s3cret" not in repr(make_settings())


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make_settings(cors_origins="http://a.com, http://b.com ,")

    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]


def test_cors_origins_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)  # conftest sets it for the app tests

    assert make_settings().cors_origins_list == []
