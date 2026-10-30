from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_PRODUCTION_SECRET_LENGTH = 32
MAX_LLM_REQUEST_TIMEOUT_SECONDS = 25.0  # see app/extraction/limits.py for why it is capped


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
    # How often Celery beat syncs every active source. At least 5 minutes: shorter than that
    # hammers the feeds' servers (and risks getting us blocked) for no real benefit.
    ingest_interval_minutes: int = Field(default=30, ge=5)
    # How many tasks one Celery worker runs at once (its prefork processes). docker-compose passes
    # this to `--concurrency`; a worker started by hand picks it up from the Celery config.
    celery_worker_concurrency: int = Field(default=2, ge=1, le=16)
    # Max manual source syncs per user within the window (each one makes outbound requests).
    sync_rate_limit_attempts: int = Field(default=10, gt=0)
    sync_rate_limit_window_seconds: int = Field(default=60, gt=0)

    # --- LLM extraction (all optional: with the defaults nothing ever calls a provider) ---------
    # "none" = rules only. "fake" is for local development and is refused in production.
    # "ollama" is a reserved name: its adapter is not written yet.
    llm_provider: Literal["none", "gemini", "ollama", "fake"] = "none"
    # The kill switch. Off by default: turning a provider on is always a deliberate act.
    llm_extraction_enabled: bool = False
    # No default on purpose: model names change, so they live in the environment, not in code.
    llm_model_extraction: str = ""
    gemini_api_key: SecretStr = SecretStr("")
    # The public endpoint; a setting only so the host is not hard-coded in the adapter.
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    # "" leaves the field out of the request. Not every model accepts every level.
    gemini_thinking_level: Literal["", "minimal", "low", "medium", "high"] = ""
    # Timeout of ONE provider request. The cap keeps a whole extraction (at most two requests)
    # inside the Celery time limits: see app/extraction/limits.py.
    llm_request_timeout_seconds: float = Field(
        default=20.0, gt=0, le=MAX_LLM_REQUEST_TIMEOUT_SECONDS
    )
    llm_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    # Characters of a post sent to a provider (longer posts keep their start and end).
    llm_max_input_chars: int = Field(default=12_000, ge=1_000, le=20_000)
    # A post shorter than this is not sent to a provider: it is a headline or a stub, and a model
    # cannot invent what the text does not say. 0 turns the gate off. 500 is based on 55 real
    # posts: every stub was 172-268 characters, every real post at least 1,945.
    llm_min_post_chars: int = Field(default=500, ge=0, le=5_000)
    # Budgets counted in provider REQUESTS, sized for a free tier. Check your own limits in the
    # provider's console and raise these deliberately; the defaults are small on purpose.
    llm_daily_request_cap_per_user: int = Field(default=25, ge=0)
    llm_daily_request_cap_global: int = Field(default=50, ge=0)
    llm_requests_per_minute: int = Field(default=5, ge=1)
    # After this many days a job that still waits for the LLM keeps its rules data and is skipped.
    llm_enrichment_give_up_days: int = Field(default=7, ge=1)

    @model_validator(mode="after")
    def _check_llm_settings(self) -> Self:
        if self.llm_provider == "fake" and self.env == "production":
            raise ValueError("LLM_PROVIDER=fake is not allowed in production")
        if self.llm_provider == "gemini" and not self.gemini_base_url.startswith("https://"):
            raise ValueError("GEMINI_BASE_URL must start with https://")
        if self.llm_min_post_chars >= self.llm_max_input_chars:
            raise ValueError("LLM_MIN_POST_CHARS must be smaller than LLM_MAX_INPUT_CHARS")
        if self.llm_extraction_enabled:
            if self.llm_provider == "none":
                raise ValueError("LLM_EXTRACTION_ENABLED needs LLM_PROVIDER to be set")
            if not self.llm_model_extraction.strip():
                raise ValueError("LLM_EXTRACTION_ENABLED needs LLM_MODEL_EXTRACTION")
            if self.llm_provider == "gemini" and not self.gemini_api_key.get_secret_value():
                raise ValueError("LLM_PROVIDER=gemini needs GEMINI_API_KEY")
        return self

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
