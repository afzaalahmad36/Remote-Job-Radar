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
    monkeypatch.setenv("SECRET_KEY", "s3cret-" * 6)  # 42 chars: long enough for production

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.env == "production"
    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/jobradar"
    assert settings.secret_key.get_secret_value() == "s3cret-" * 6


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


@pytest.mark.parametrize("key", ["change-me", "Change-Me", "too-short-for-production", "x" * 31])
def test_production_rejects_placeholder_and_short_secret_keys(key: str) -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        make_settings(env="production", secret_key=key)


def test_validation_errors_do_not_print_secrets_or_the_database_password() -> None:
    with pytest.raises(ValidationError) as error:
        make_settings(
            env="production",
            secret_key="my-real-but-short-key",
            database_url="postgresql+asyncpg://u:db-password-123@db:5432/x",
        )

    message = str(error.value)
    assert "my-real-but-short-key" not in message
    assert "db-password-123" not in message


def test_production_accepts_a_long_secret_key() -> None:
    settings = make_settings(env="production", secret_key="x" * 32)

    assert settings.env == "production"


@pytest.mark.parametrize("env", ["development", "test"])
def test_other_environments_allow_a_short_secret_key(env: str) -> None:
    make_settings(env=env, secret_key="change-me")  # no error


def test_token_and_rate_limit_defaults() -> None:
    settings = make_settings()

    assert settings.access_token_expire_minutes == 30
    assert settings.refresh_token_expire_days == 7
    assert (settings.auth_rate_limit_attempts, settings.auth_rate_limit_window_seconds) == (10, 60)


def test_secret_key_is_hidden_in_repr() -> None:
    assert "s3cret" not in repr(make_settings())


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make_settings(cors_origins="http://a.com, http://b.com ,")

    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]


def test_cors_origins_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)  # conftest sets it for the app tests

    assert make_settings().cors_origins_list == []


# --- LLM settings -----------------------------------------------------------------------------


def test_llm_defaults_make_no_provider_call_possible() -> None:
    settings = make_settings()

    assert settings.llm_provider == "none"
    assert settings.llm_extraction_enabled is False
    assert settings.llm_model_extraction == ""  # no model name is baked into the code
    assert settings.gemini_api_key.get_secret_value() == ""


def test_llm_budget_defaults_are_small_and_counted_in_requests() -> None:
    settings = make_settings()

    assert settings.llm_daily_request_cap_per_user == 25
    assert settings.llm_daily_request_cap_global == 50
    assert settings.llm_requests_per_minute == 5
    assert settings.llm_enrichment_give_up_days == 7
    assert settings.llm_min_post_chars == 500
    assert settings.llm_request_timeout_seconds == 20.0


def test_llm_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "some-model")
    monkeypatch.setenv("GEMINI_API_KEY", "a-key")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert (settings.llm_provider, settings.llm_extraction_enabled) == ("gemini", True)
    assert settings.llm_model_extraction == "some-model"
    assert settings.gemini_api_key.get_secret_value() == "a-key"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"llm_extraction_enabled": True}, "LLM_PROVIDER"),
        ({"llm_extraction_enabled": True, "llm_provider": "gemini"}, "LLM_MODEL_EXTRACTION"),
        (
            {"llm_extraction_enabled": True, "llm_provider": "gemini", "llm_model_extraction": "m"},
            "GEMINI_API_KEY",
        ),
    ],
)
def test_enabling_extraction_requires_a_complete_provider_setup(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(**overrides)  # type: ignore[arg-type]


def test_a_provider_may_be_named_without_being_enabled() -> None:
    make_settings(llm_provider="gemini")  # no error: the kill switch is still off


def test_the_fake_provider_is_refused_in_production_only() -> None:
    make_settings(llm_provider="fake")  # development
    with pytest.raises(ValidationError, match="fake"):
        make_settings(llm_provider="fake", env="production", secret_key="x" * 32)


def test_the_gemini_endpoint_must_be_https() -> None:
    with pytest.raises(ValidationError, match="https"):
        make_settings(llm_provider="gemini", gemini_base_url="http://generativelanguage.example")


@pytest.mark.parametrize("name", ["llm_request_timeout_seconds", "llm_requests_per_minute"])
def test_llm_numbers_must_be_positive(name: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: 0})  # type: ignore[arg-type]


def test_the_gemini_key_is_hidden_in_repr_and_in_validation_errors() -> None:
    key = "AIza-very-secret-key"

    assert key not in repr(make_settings(gemini_api_key=key))
    with pytest.raises(ValidationError) as error:
        make_settings(gemini_api_key=key, llm_provider="gemini", llm_extraction_enabled=True)
    assert key not in str(error.value)


@pytest.mark.parametrize("value", ["-1", "5001"])
def test_llm_min_post_chars_must_be_in_range(value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_min_post_chars=value)


def test_llm_min_post_chars_must_be_below_the_input_limit() -> None:
    with pytest.raises(ValidationError, match="LLM_MIN_POST_CHARS"):
        make_settings(llm_min_post_chars="2000", llm_max_input_chars="1500")

    assert make_settings(llm_min_post_chars="0").llm_min_post_chars == 0
