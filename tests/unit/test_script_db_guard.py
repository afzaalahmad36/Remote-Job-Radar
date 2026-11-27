"""The database guard of the manual scripts: pure checks, no connection."""

import pytest

from scripts import recompute_extractions, seed_demo
from scripts.db_guard import ScriptError, database_target, local_database_target

LOCAL = "postgresql+asyncpg://u:p@localhost:5434/jobradar"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_a_database_on_this_machine_is_allowed(host: str) -> None:
    assert local_database_target(LOCAL.replace("localhost", host)).endswith(":5434/jobradar")


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        (LOCAL.replace("/jobradar", "/jobradar_test"), "named 'jobradar'"),
        (LOCAL.replace("/jobradar", "/other"), "named 'jobradar'"),
        (LOCAL.replace("localhost", "db.example.com"), "this machine"),
        (LOCAL.replace("localhost", "localhost,db.example.com"), "this machine|valid"),
        ("postgresql+asyncpg:///jobradar", "without a host"),  # PGHOST would decide
        ("not a url", "valid database URL"),
    ],
)
def test_a_wrong_name_or_a_host_that_may_be_remote_is_refused(url: str, reason: str) -> None:
    with pytest.raises(ScriptError, match=reason):
        local_database_target(url)


@pytest.mark.parametrize(
    "url",
    [
        f"{LOCAL}?host=db.example.com",
        "postgresql+asyncpg://u:p@/jobradar?host=db.example.com",
        f"{LOCAL}?database=other",
        f"{LOCAL}?dsn=postgresql://db.example.com/jobradar",
        f"{LOCAL}?password=secret-value",
    ],
)
def test_a_query_that_could_replace_the_checked_values_is_refused(url: str) -> None:
    for check in (database_target, local_database_target):
        with pytest.raises(ScriptError, match="must not have a query") as caught:
            check(url)
        assert "secret-value" not in str(caught.value)


def test_both_writing_scripts_use_the_local_guard() -> None:
    assert vars(seed_demo)["local_database_target"] is local_database_target
    assert vars(recompute_extractions)["local_database_target"] is local_database_target
