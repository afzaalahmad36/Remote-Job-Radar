"""Helpers for tests that touch Postgres, including the guard that protects real databases."""

import os
from pathlib import Path

from alembic.config import Config
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent
TEST_DB_NAME = "jobradar_test"
# Local default: the jobradar-pg container on host port 5433. CI overrides it via the env var.
DEFAULT_TEST_DATABASE_URL = f"postgresql+asyncpg://postgres:postgres@localhost:5433/{TEST_DB_NAME}"


class UnsafeDatabaseError(RuntimeError):
    """Raised when a test would run against a database that is not the dedicated test one."""


def require_test_database(url: str) -> str:
    """Return `url` only if it points at `jobradar_test`; refuse anything else."""
    database = make_url(url).database
    if database != TEST_DB_NAME:
        raise UnsafeDatabaseError(
            f"Refusing to run tests against database {database!r}; "
            f"only {TEST_DB_NAME!r} is allowed."
        )
    return url


def get_test_database_url() -> str:
    """TEST_DATABASE_URL (never DATABASE_URL, which may point at a real database) or the default."""
    return require_test_database(os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL))


def alembic_config(database_url: str) -> Config:
    """An Alembic config aimed at `database_url` instead of the app's DATABASE_URL."""
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config
