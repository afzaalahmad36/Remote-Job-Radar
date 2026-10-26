import os
import subprocess
import sys

import pytest

from tests.db_utils import (
    DEFAULT_TEST_DATABASE_URL,
    ROOT,
    UnsafeDatabaseError,
    get_test_database_url,
    require_test_database,
)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://postgres:postgres@localhost:5433/jobradar",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/jobradar_test_backup",
        "postgresql+asyncpg://postgres:postgres@localhost:5433",
    ],
)
def test_guard_refuses_databases_not_named_jobradar_test(url: str) -> None:
    with pytest.raises(UnsafeDatabaseError):
        require_test_database(url)


def test_guard_accepts_the_test_database() -> None:
    assert require_test_database(DEFAULT_TEST_DATABASE_URL) == DEFAULT_TEST_DATABASE_URL


def test_env_override_is_checked_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/jobradar")

    with pytest.raises(UnsafeDatabaseError):
        get_test_database_url()


def test_a_test_run_aborts_before_touching_a_wrongly_named_database() -> None:
    """End to end: the DB fixtures stop the whole run, they don't just fail one test."""
    env = os.environ | {"TEST_DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5433/jobradar"}

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/integration"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2  # pytest.exit(returncode=2)
    assert "Refusing to run tests against database 'jobradar'" in result.stdout + result.stderr


def test_database_url_env_var_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/jobradar")

    assert get_test_database_url() == DEFAULT_TEST_DATABASE_URL
