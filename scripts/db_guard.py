"""Reading the development database safely from the manual scripts.

Shared by `try_extraction` and the evaluation scripts, so the guard exists once: only a database
named `jobradar`, only one that has a `jobs` table, and a connection that Postgres itself puts in
read-only mode (any write is refused by the server, not by our code). The scripts that write use
`local_database_target`, which also requires the database to be on this machine.
"""

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

EXPECTED_DATABASE = "jobradar"  # the development database; never the test one or another project's
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ScriptError(Exception):
    """A problem the user can fix; main() prints it without a traceback."""


def database_target(database_url: str) -> str:
    """ "host:port/name" of the database, after checking it is the one a script may read.

    The password is never part of the result. The name is checked before any connection is made.
    A URL with a query is refused: the driver lets `?host=`, `?database=` or `?dsn=` replace what
    the rest of the URL says, so these checks would look at values that are not used.
    """
    try:
        url = make_url(database_url)
    except ArgumentError:
        raise ScriptError("DATABASE_URL is not a valid database URL") from None
    if url.query:  # the message never repeats the query: it can hold a password
        raise ScriptError("DATABASE_URL must not have a query (the part after '?') for a script")
    if url.database != EXPECTED_DATABASE:
        raise ScriptError(
            f"refusing to read database {url.database!r} on {url.host}:{url.port}: "
            f"this script only reads a database named {EXPECTED_DATABASE!r}"
        )
    return f"{url.host or '(local socket)'}:{url.port or 5432}/{url.database}"


def local_database_target(database_url: str) -> str:
    """`database_target`, for a script that writes: the host must also be this machine.

    A URL without a host is refused too, because the driver would then take it from `PGHOST`.
    """
    target = database_target(database_url)
    host = make_url(database_url).host
    if host not in LOCAL_HOSTS:
        raise ScriptError(
            f"refusing to write to {host or 'a URL without a host'}: "
            "this script only writes to this machine (use localhost)"
        )
    return target


async def ensure_jobs_table(connection: AsyncConnection) -> None:
    """Refuse a database that has no `jobs` table (a wrong or unmigrated database)."""
    if await connection.scalar(text("select to_regclass('jobs')::text")) is None:
        raise ScriptError("the database has no 'jobs' table: wrong database, or not migrated")


def read_only_engine(database_url: str) -> AsyncEngine:
    """An engine whose connections the server itself keeps read-only."""
    return create_async_engine(
        database_url, connect_args={"server_settings": {"default_transaction_read_only": "on"}}
    )
