"""Load a demo account with about 20 invented job posts, to try the API without a real feed.

Run by hand, never by CI or app code. It makes no provider request and no network request.

What it does:

1. creates the demo user (`demo@example.com`) and ONE inactive RSS source whose URL is on
   example.com, so nothing is ever fetched from it;
2. stores the posts of `tests/fixtures/demo/demo_jobs.json` (invented: fictional companies,
   example.com addresses only), once each: running it again adds nothing;
3. runs the real rules phase on each post (`extract_job` with no model client), so title, stack,
   salary, seniority and remote policy are filled by the same code the pipeline uses.

It does NOT embed or score. Those happen in the background: with the full stack running
(`worker`, `worker-embed`, `beat` and `EMBEDDING_PROVIDER=fastembed`) the 5-minute sweep embeds the
jobs, and the demo user's CV upload then produces scores (docs/API_EXAMPLES.md).

The password: by default a random one, printed ONCE (it is not stored anywhere else). A fixed,
documented one exists only behind `--demo-password`, for a throwaway local database. Both are
refused when ENV=production, and the script only writes to a database named `jobradar` on this
machine.

    uv run python -m scripts.seed_demo                    # creates the user, prints the password
    uv run python -m scripts.seed_demo --demo-password    # the fixed password (local only)
    uv run python -m scripts.seed_demo --reset            # delete the demo user and start again
"""

import argparse
import asyncio
import json
import secrets
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.extraction.text_hash import raw_text_hash
from app.models import ExtractionStatus, Job, Source, SourceType, User
from app.services.extraction_service import extract_job
from scripts.db_guard import ScriptError, database_target

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "demo" / "demo_jobs.json"

DEMO_EMAIL = "demo@example.com"
# Only for `--demo-password`: it is public (it is in the docs), so it must never guard real data.
DEMO_FIXED_PASSWORD = "demo-password-local-only"  # noqa: S105
SOURCE_NAME = "Demo feed (invented posts)"
SOURCE_URL = "https://example.com/demo-feed.xml"  # never fetched: the source is inactive
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class DemoJob(BaseModel):
    """One invented post of the fixture."""

    model_config = ConfigDict(extra="forbid")

    external_id: str = Field(min_length=1, max_length=200)
    days_before: int = Field(ge=0, le=365)  # how long before "now" it was posted
    text: str = Field(min_length=20, max_length=20_000)


class DemoFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str
    jobs: list[DemoJob]


def load_fixture(path: Path = FIXTURE) -> list[DemoJob]:
    """The posts of the fixture, validated (unknown fields, duplicate ids and texts refused)."""
    jobs = DemoFixture.model_validate(json.loads(path.read_text(encoding="utf-8"))).jobs
    if len({job.external_id for job in jobs}) != len(jobs):
        raise ScriptError("the fixture has a repeated external_id")
    if len({raw_text_hash(job.text) for job in jobs}) != len(jobs):
        raise ScriptError("the fixture has two posts with the same text")
    return jobs


@dataclass(frozen=True)
class SeedResult:
    user_created: bool
    password: str | None  # only when the user was created by this run
    inserted: int  # jobs stored by this run
    extracted: int  # jobs of the demo source that have a title after the rules phase
    total: int  # jobs of the demo source


def check_target(settings: Settings) -> str:
    """ "host:port/name" if this is the local `jobradar` database and not production."""
    if settings.env == "production":
        raise ScriptError("refusing to run with ENV=production: the demo account is for local use")
    target = database_target(settings.database_url)
    host = make_url(settings.database_url).host or "localhost"
    if host not in LOCAL_HOSTS:
        raise ScriptError(f"refusing to write to {host}: this script only writes to this machine")
    return target


async def delete_demo_user(factory: async_sessionmaker[AsyncSession]) -> int:
    """Delete the demo user only; the cascade removes its source, jobs, CV and matches."""
    async with factory() as session, session.begin():
        result = await session.execute(delete(User).where(User.email == DEMO_EMAIL))
        return int(result.rowcount or 0)  # type: ignore[attr-defined]


async def _get_or_create_user(
    factory: async_sessionmaker[AsyncSession], password: str
) -> tuple[uuid.UUID, bool]:
    async with factory() as session, session.begin():
        user_id = await session.scalar(select(User.id).where(User.email == DEMO_EMAIL))
        if user_id is not None:
            return user_id, False
        user = User(email=DEMO_EMAIL, hashed_password=await hash_password(password))
        session.add(user)
        await session.flush()
        return user.id, True


async def _get_or_create_source(session: AsyncSession, user_id: uuid.UUID) -> uuid.UUID:
    source_id = await session.scalar(
        select(Source.id).where(Source.user_id == user_id, Source.name == SOURCE_NAME)
    )
    if source_id is not None:
        return source_id
    source = Source(
        user_id=user_id,
        type=SourceType.RSS,
        name=SOURCE_NAME,
        config={"url": SOURCE_URL},
        is_active=False,  # beat only syncs active sources: nothing is ever fetched
    )
    session.add(source)
    await session.flush()
    return source.id


async def _insert_jobs(
    session: AsyncSession, source_id: uuid.UUID, jobs: Sequence[DemoJob], now: datetime
) -> tuple[uuid.UUID, ...]:
    rows = [
        {
            "id": uuid.uuid4(),
            "source_id": source_id,
            "external_id": job.external_id,
            "raw_text": job.text,
            "posted_at": now - timedelta(days=job.days_before),
            "extraction_status": ExtractionStatus.PENDING,
        }
        for job in sorted(jobs, key=lambda j: j.external_id)
    ]
    statement = (
        insert(Job)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["source_id", "external_id"])  # idempotent
        .returning(Job.id)
    )
    return tuple((await session.execute(statement)).scalars().all())


async def seed(
    factory: async_sessionmaker[AsyncSession],
    jobs: Sequence[DemoJob],
    *,
    settings: Settings,
    password: str,
    now: datetime | None = None,
) -> SeedResult:
    """Create what is missing and run the rules phase on the new posts. Safe to run again."""
    now = now or datetime.now(UTC)
    user_id, user_created = await _get_or_create_user(factory, password)
    async with factory() as session, session.begin():
        source_id = await _get_or_create_source(session, user_id)
        new_ids = await _insert_jobs(session, source_id, jobs, now)
    for job_id in new_ids:  # rules only: no client, so no model and no network
        await extract_job(factory, job_id, settings=settings, now=now)
    async with factory() as session:
        stored = (await session.execute(select(Job.title).where(Job.source_id == source_id))).all()
    return SeedResult(
        user_created=user_created,
        password=password if user_created else None,
        inserted=len(new_ids),
        extracted=sum(1 for (title,) in stored if title),
        total=len(stored),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed_demo", description="Load a demo account with invented job posts."
    )
    parser.add_argument(
        "--demo-password",
        action="store_true",
        help="use the fixed, public password instead of a random one (throwaway local DB only)",
    )
    parser.add_argument(
        "--reset", action="store_true", help="delete the demo user (and its data) first"
    )
    return parser


async def _run(settings: Settings, args: argparse.Namespace) -> int:
    print(f"Database: {check_target(settings)}")
    jobs = load_fixture()
    if args.demo_password:
        print(
            "WARNING: the fixed demo password is public. Use it only on a throwaway local "
            "database, never on one that holds real data."
        )
        password = DEMO_FIXED_PASSWORD
    else:
        password = secrets.token_urlsafe(12)
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        if args.reset:
            print(f"Deleted {await delete_demo_user(factory)} demo user(s).")
        result = await seed(factory, jobs, settings=settings, password=password)
    finally:
        await engine.dispose()
    print(f"Posts stored: {result.inserted} new, {result.total} in total.")
    print(f"Posts with a title after the rules phase: {result.extracted} of {result.total}.")
    if result.password is not None:
        print(f"Demo login: {DEMO_EMAIL} / {result.password}")
        if not args.demo_password:
            print("This password is shown once and not stored anywhere else.")
    else:
        print(f"The demo user already exists ({DEMO_EMAIL}); its password was not changed.")
        print("Use --reset to start again with a new password.")
    print("Embeddings and scores come from the running stack (see docs/API_EXAMPLES.md).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(get_settings(), args))
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
