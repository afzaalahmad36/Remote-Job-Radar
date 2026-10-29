from app.core.config import Settings


def settings_with(**overrides: object) -> Settings:
    """Settings that ignore any local .env, so a test only sees what it passes in."""
    values: dict[str, object] = {
        "database_url": "postgresql+asyncpg://u:p@db:5432/x",
        "redis_url": "redis://redis:6379/0",
        "secret_key": "s3cret",
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[arg-type]
