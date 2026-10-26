from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables (and a local `.env` file).

    pydantic-settings matches each field to the env var of the same name, case-insensitively
    (`database_url` <- `DATABASE_URL`), converts it to the declared type and raises a
    `ValidationError` at startup if a required one is missing. Fields without a default are
    required, so no secret or URL is ever hard-coded here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"
    database_url: str
    redis_url: str
    # SecretStr hides the value in repr() and logs; call .get_secret_value() to read it.
    secret_key: SecretStr
    # Comma-separated list, e.g. "http://localhost:3000,https://app.example.com".
    cors_origins: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once per process (lru_cache memoizes the first call)."""
    return Settings()  # type: ignore[call-arg]  # required fields are filled from the environment
