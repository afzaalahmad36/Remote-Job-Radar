"""The embedding settings: off by default, and a pinned revision that is never a branch name."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings

ROOT = Path(__file__).resolve().parent.parent.parent
REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"


def make(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "database_url": "postgresql+asyncpg://u:p@localhost/x",
        "redis_url": "redis://localhost:1/0",
        "secret_key": SecretStr("test-secret-key-at-least-32-characters-long"),
    }
    return Settings(**{**base, **overrides})


def test_embeddings_are_off_by_default() -> None:
    assert make().embedding_provider == "none"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_provider_from_an_older_env_file_means_off(blank: str) -> None:
    assert make(embedding_provider=blank).embedding_provider == "none"


def test_the_model_is_named_by_its_name_and_its_pinned_revision() -> None:
    settings = make()

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_model_revision == REVISION
    assert settings.embedding_model_id == f"BAAI/bge-small-en-v1.5@{REVISION}"
    assert len(settings.embedding_model_id) <= 100  # the column is VARCHAR(100)


@pytest.mark.parametrize("revision", ["main", "", "abc123", REVISION.upper(), REVISION + "0"])
def test_the_real_model_needs_a_full_commit_hash_not_a_branch(revision: str) -> None:
    with pytest.raises(ValidationError, match="40-character commit hash"):
        make(embedding_provider="fastembed", embedding_model_revision=revision)


def test_a_branch_name_is_fine_while_the_real_model_is_off() -> None:
    assert make(embedding_model_revision="main").embedding_provider == "none"


def test_the_fake_embedder_is_refused_in_production() -> None:
    with pytest.raises(ValidationError, match="not allowed in production"):
        make(env="production", embedding_provider="fake")

    assert make(env="development", embedding_provider="fake").embedding_provider == "fake"


def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(ValidationError):
        make(embedding_provider="openai")


def test_the_example_env_file_documents_every_embedding_setting_with_the_same_defaults() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for line in (
        "EMBEDDING_PROVIDER=none",
        "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
        f"EMBEDDING_MODEL_REVISION={REVISION}",
        "EMBEDDING_CACHE_DIR=/models",
    ):
        assert line in example.splitlines()
