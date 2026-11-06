"""One-off: send old rules-only extractions back to "rules phase not run", so the sweep redoes them.

Run by hand, never by CI and never imported by app code. It makes no provider request.

Why: `RULES_VERSION` only decides whether an old result may be reused for identical text; the sweep
only picks up jobs whose `extraction_method` is NULL. After the rules change (a new version), jobs
made by the old rules keep their old values until something resets them. This script resets them
the way `extraction_repository.reset_for_retry` does (status `pending`, enrichment `not_needed`,
error NULL, method NULL) without touching `extraction_retries` or `extraction_attempts`. The
extraction sweep (beat + a worker) then runs the rules again for each of them, 100 per run.

Which rows: `extraction_method = 'rules'` and a version older than the current one. Never a row the
model touched: `rules_llm` rows, rows with `llm_enrichment = 'done'` and rows holding a model or a
prompt version are left alone. The reset is done in batches, each in its own transaction, and the
conditions are checked again inside the UPDATE, so a worker that changed a job in between is not
overwritten.

Safety: it refuses to run without --confirm-backup-taken (a dry run writes nothing, so it does not
need it), and it only touches a database named `jobradar` on this machine.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Integer, case, cast, func, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.elements import ColumnElement

from app.core.config import get_settings
from app.extraction.rules import RULES_VERSION
from app.models import ExtractionMethod, ExtractionStatus, Job, LlmEnrichment
from scripts.db_guard import ScriptError, database_target

logger = logging.getLogger("recompute_extractions")

CONFIRM_FLAG = "--confirm-backup-taken"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_BATCH_SIZE = 100  # what the sweep queues per run
MAX_BATCH_SIZE = 1_000
VERSION_PATTERN = r"^rules_v[0-9]{1,6}$"  # "rules_v2"; anything else is never reset
VERSION_PREFIX_LENGTH = len("rules_v")

# Two commands, no shell redirect: in PowerShell 5.1 `>` re-encodes a binary dump and corrupts it.
BACKUP_HELP = (
    "Take a backup first, then pass --confirm-backup-taken:\n"
    "  docker compose exec -T db pg_dump -U postgres -Fc -f /tmp/jobradar.dump jobradar\n"
    "  docker compose cp db:/tmp/jobradar.dump ./jobradar-backup.dump\n"
    "Check the file size before you go on: it must be about as large as your data (hundreds of KB\n"
    "or more). A 0-byte or tiny file means the dump failed; do not continue without a real one."
)


def version_number(version: str) -> int:
    """3 for "rules_v3". Raises ValueError for any other spelling."""
    if not version.startswith("rules_v") or not version[VERSION_PREFIX_LENGTH:].isdigit():
        raise ValueError(f"not a rules version: {version!r}")
    return int(version[VERSION_PREFIX_LENGTH:])


def candidate_conditions(current: int) -> list[ColumnElement[bool]]:
    """The one definition of "an old rules-only row", used by the count and by the UPDATE.

    The version number is read inside a CASE, which Postgres evaluates in order, so a row whose
    version is not `rules_v<digits>` gives NULL (and is not a candidate) instead of a cast error.
    """
    number = case(
        (
            Job.extraction_rules_version.op("~")(VERSION_PATTERN),
            cast(func.substr(Job.extraction_rules_version, VERSION_PREFIX_LENGTH + 1), Integer),
        ),
        else_=None,
    )
    return [
        Job.extraction_method == ExtractionMethod.RULES,
        number < current,
        Job.llm_enrichment != LlmEnrichment.DONE,
        Job.extraction_model.is_(None),
        Job.extraction_prompt_version.is_(None),
    ]


@dataclass(frozen=True)
class Candidates:
    total: int
    breakdown: list[tuple[str, str, str, int]]  # version, status, enrichment, rows


async def count_candidates(session: AsyncSession, current: int) -> Candidates:
    conditions = candidate_conditions(current)
    rows = (
        await session.execute(
            select(
                Job.extraction_rules_version,
                Job.extraction_status,
                Job.llm_enrichment,
                func.count(),
            )
            .where(*conditions)
            .group_by(Job.extraction_rules_version, Job.extraction_status, Job.llm_enrichment)
            .order_by(Job.extraction_rules_version, Job.extraction_status, Job.llm_enrichment)
        )
    ).all()
    breakdown = [(str(v), s.value, e.value, int(n)) for v, s, e, n in rows]
    return Candidates(total=sum(row[3] for row in breakdown), breakdown=breakdown)


async def reset_batch(session: AsyncSession, current: int, batch_size: int) -> int:
    """Reset up to `batch_size` candidates, oldest first. Returns how many rows were reset.

    Two steps in one transaction: pick the ids with `FOR UPDATE SKIP LOCKED` (rows a worker is
    writing right now are left for a later run, and the picked rows stay locked until commit),
    then update exactly those ids. The ids are picked first and not in a subquery of the UPDATE
    because Postgres may run a `LIMIT ... FOR UPDATE SKIP LOCKED` subquery more than once, which
    updated more rows than the limit (a test caught it). The conditions are repeated in the
    UPDATE. `extraction_retries` is deliberately not set.
    """
    conditions = candidate_conditions(current)
    ids = list(
        await session.scalars(
            select(Job.id)
            .where(*conditions)
            .order_by(Job.created_at, Job.id)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
    )
    if not ids:
        return 0
    statement = (
        update(Job)
        .where(Job.id.in_(ids), *conditions)
        .values(
            extraction_status=ExtractionStatus.PENDING,
            llm_enrichment=LlmEnrichment.NOT_NEEDED,
            extraction_error=None,
            extraction_method=None,
        )
        .returning(Job.id)
    )
    return len((await session.execute(statement)).all())


@dataclass(frozen=True)
class RecomputeResult:
    candidates: Candidates
    batches: list[int]  # rows reset in each batch, in order (empty for a dry run)

    @property
    def reset(self) -> int:
        return sum(self.batches)


async def recompute(
    factory: async_sessionmaker[AsyncSession],
    *,
    current_version: str = RULES_VERSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> RecomputeResult:
    """Count the old rules-only rows and, unless `dry_run`, reset them batch by batch."""
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ScriptError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    current = version_number(current_version)
    async with factory() as session:
        candidates = await count_candidates(session, current)
    batches: list[int] = []
    while not dry_run and candidates.total:
        async with factory() as session, session.begin():  # one short transaction per batch
            reset = await reset_batch(session, current, batch_size)
        if reset == 0:
            break  # nothing left, or every remaining row is locked by a worker: run again later
        batches.append(reset)
        logger.info("batch %d: reset %d row(s)", len(batches), reset)
    return RecomputeResult(candidates, batches)


def check_local_database(database_url: str) -> str:
    """ "host:port/name" if this is the local `jobradar` database; ScriptError otherwise."""
    target = database_target(database_url)  # the name must be jobradar
    host = make_url(database_url).host or "localhost"
    if host not in LOCAL_HOSTS:
        raise ScriptError(f"refusing to write to {host}: this script only writes to this machine")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recompute_extractions",
        description="Reset old rules-only extractions so the sweep recomputes them.",
        epilog=BACKUP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(CONFIRM_FLAG, action="store_true", help="required for anything that writes")
    parser.add_argument("--dry-run", action="store_true", help="print counts only; write nothing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def _print_counts(candidates: Candidates, current_version: str) -> None:
    print(
        f"Rows made by an older rules version than {current_version} (method rules): "
        f"{candidates.total}"
    )
    for version, status, enrichment, rows in candidates.breakdown:
        print(f"  {version}  {status}  {enrichment}: {rows}")


async def _run(database_url: str, args: argparse.Namespace) -> int:
    target = check_local_database(database_url)
    print(f"Database: {target}  (current rules version: {RULES_VERSION})")
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        result = await recompute(factory, batch_size=args.batch_size, dry_run=args.dry_run)
    finally:
        await engine.dispose()
    _print_counts(result.candidates, RULES_VERSION)
    if args.dry_run:
        print("Dry run: nothing was changed.")
    else:
        print(f"Reset {result.reset} row(s) in {len(result.batches)} batch(es).")
        print("The extraction sweep (beat + a worker) now recomputes them, 100 per 5 minutes.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        if not args.dry_run and not args.confirm_backup_taken:
            raise ScriptError(f"{CONFIRM_FLAG} is required.\n{BACKUP_HELP}")
        return asyncio.run(_run(get_settings().database_url, args))
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
