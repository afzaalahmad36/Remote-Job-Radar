from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_PRODUCTION_SECRET_LENGTH = 32


class Settings(BaseSettings):
    """Application settings, read from environment variables (and a local `.env` file).

    pydantic-settings matches each field to the env var of the same name, case-insensitively
    (`database_url` <- `DATABASE_URL`), converts it to the declared type and raises a
    `ValidationError` at startup if a required one is missing. Fields without a default are
    required, so no secret or URL is ever hard-coded here.
    """

    # hide_input_in_errors: a failed validation must not print the submitted values, which here
    # include SECRET_KEY and the database URL (with its password) in startup logs.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    env: Literal["development", "test", "production"] = "development"
    database_url: str
    redis_url: str
    # SecretStr hides the value in repr() and logs; call .get_secret_value() to read it.
    secret_key: SecretStr
    # Comma-separated list, e.g. "http://localhost:3000,https://app.example.com".
    cors_origins: str = ""

    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_days: int = Field(default=7, gt=0)
    # Max requests per client IP and auth endpoint within the window (in-process limiter).
    auth_rate_limit_attempts: int = Field(default=10, gt=0)
    auth_rate_limit_window_seconds: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _reject_weak_secret_key_in_production(self) -> Self:
        """A model validator runs after all fields are parsed, so it can compare several."""
        if self.env == "production":
            key = self.secret_key.get_secret_value()
            if key.lower() == "change-me":
                raise ValueError("SECRET_KEY must not be the placeholder 'change-me' in production")
            if len(key) < MIN_PRODUCTION_SECRET_LENGTH:
                raise ValueError(
                    f"SECRET_KEY must be at least {MIN_PRODUCTION_SECRET_LENGTH} characters "
                    "in production"
                )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once per process (lru_cache memoizes the first call)."""
    return Settings()  # type: ignore[call-arg]  # required fields are filled from the environment
