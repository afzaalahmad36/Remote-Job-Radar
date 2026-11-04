"""Label job posts by hand, blind, to measure extraction accuracy later.

    uv run python -m scripts.eval_label --source-name "My Feed" --n 30 --seed 1
    uv run python -m scripts.eval_label --source-id <uuid>        # same, by id

It picks N posts of ONE source with distinct text (a fixed --seed gives the same pick again), shows
each in full, and asks for what the post says. It never reads or shows what the system extracted:
the only columns it selects are the job id and the text. Blank means "the post does not say".
Commands at any prompt: /skip (leave this post out) and /quit (stop; run again to resume).

Conventions for the salary, so labels match how the rules read a post: a single amount ("$90k")
is min = max; "up to X" is max only; "from X" or "X+" is min only. Stack: list every technology the
post names, required or optional. Labels are saved after every post to .local/eval/labels.json
(git-ignored); the file holds ids, a hash of the text and your labels, never the post text.

Reads the database like try_extraction does: only a database named `jobradar`, read-only.
Scoring is `uv run python -m scripts.try_extraction --eval .local/eval/labels.json`.
"""

import argparse
import asyncio
import random
import sys
import uuid
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import asyncpg  # type: ignore[import-untyped]
from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.extraction.text_hash import raw_text_hash
from app.models import Job, RemotePolicy, SalaryPeriod, Seniority, Source
from scripts.db_guard import ScriptError, database_target, ensure_jobs_table, read_only_engine
from scripts.eval_labelfile import (
    DEFAULT_LABELS_PATH,
    LabelEntry,
    LabelFile,
    Labels,
    load_label_file,
    save_label_file,
)

DEFAULT_N = 30
MAX_N = 200
MAX_CANDIDATES = 5_000  # posts read from the source before picking
PAGE_LINES = 40


class StopLabelling(Exception):
    """The labeller typed /quit (or the input ended)."""


class SkipPost(Exception):
    """The labeller typed /skip."""


class RedoPost(Exception):
    """The labeller asked to enter this post again."""


class Prompter(Protocol):
    def ask(self, prompt: str) -> str: ...

    def show(self, text: str) -> None: ...


class ConsolePrompter:
    def ask(self, prompt: str) -> str:
        try:
            return input(prompt).strip()
        except EOFError:
            raise StopLabelling from None

    def show(self, text: str) -> None:
        print(text)


# --- choosing the posts ------------------------------------------------------------------------


def selection_query(source_id: uuid.UUID) -> Select[tuple[uuid.UUID, str]]:
    """The only thing read about the posts: id and text. Nothing the system extracted."""
    return (
        select(Job.id, Job.raw_text)
        .where(Job.source_id == source_id)
        .order_by(Job.id)
        .limit(MAX_CANDIDATES)
    )


def pick_posts(rows: list[tuple[uuid.UUID, str]], n: int, seed: int) -> list[uuid.UUID]:
    """N posts with distinct text, the same ones for the same rows and seed.

    Posts are put in id order first (so database order cannot change the result), the first post of
    each distinct text is kept, and `random.Random(seed)` draws from those.
    """
    unique: dict[str, uuid.UUID] = {}
    for job_id, text in sorted(rows, key=lambda row: row[0]):
        unique.setdefault(raw_text_hash(text), job_id)
    candidates = sorted(unique.values())
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


async def _fetch_rows(
    database_url: str, query: Select[tuple[uuid.UUID, str]]
) -> list[tuple[uuid.UUID, str]]:
    """(id, text) rows. The connection itself is read-only (Postgres refuses any write)."""
    engine = read_only_engine(database_url)
    try:
        async with engine.connect() as connection:
            await ensure_jobs_table(connection)
            return [(row.id, row.raw_text) for row in await connection.execute(query)]
    except (OSError, SQLAlchemyError, asyncpg.PostgresError) as error:
        # Only the error's type: its message could contain the database URL and password.
        raise ScriptError(f"could not read the database ({type(error).__name__})") from None
    finally:
        await engine.dispose()


async def read_source_posts(database_url: str, source_id: uuid.UUID) -> list[tuple[uuid.UUID, str]]:
    return await _fetch_rows(database_url, selection_query(source_id))


async def read_posts_by_id(
    database_url: str, job_ids: list[uuid.UUID]
) -> list[tuple[uuid.UUID, str]]:
    return await _fetch_rows(database_url, select(Job.id, Job.raw_text).where(Job.id.in_(job_ids)))


async def find_source(database_url: str, name: str) -> uuid.UUID:
    engine = read_only_engine(database_url)
    try:
        async with engine.connect() as connection:
            await ensure_jobs_table(connection)
            ids = list(await connection.scalars(select(Source.id).where(Source.name == name)))
    except (OSError, SQLAlchemyError, asyncpg.PostgresError) as error:
        raise ScriptError(f"could not read the database ({type(error).__name__})") from None
    finally:
        await engine.dispose()
    if not ids:
        raise ScriptError(f"no source is named {name!r}; use --source-id")
    if len(ids) > 1:
        listed = ", ".join(str(i) for i in ids)
        raise ScriptError(
            f"{len(ids)} sources are named {name!r}; use --source-id with one of: {listed}"
        )
    return ids[0]


# --- reading what the labeller types -----------------------------------------------------------


def parse_stack(answer: str) -> list[str]:
    """Comma-separated technologies, trimmed, without empty entries or repeats."""
    seen: dict[str, None] = {}
    for part in answer.split(","):
        entry = part.strip()
        if entry:
            seen.setdefault(entry, None)
    return list(seen)


def parse_amount(answer: str) -> int | None:
    """ "90000", "90,000", "90 000", "90k" and "1.5k" are amounts; blank is none."""
    text = answer.strip().lower().replace(",", "").replace(" ", "").replace("_", "")
    if not text:
        return None
    factor = 1
    if text.endswith("k"):
        text, factor = text[:-1], 1000
    try:
        value = float(text) * factor
    except ValueError:
        raise ValueError("enter a number such as 90000, 90,000 or 90k") from None
    if value < 0 or value != value or value == float("inf"):
        raise ValueError("enter a positive number")
    return round(value)


def parse_currency(answer: str) -> str | None:
    text = answer.strip().upper()
    if text and not (len(text) == 3 and text.isalpha()):
        raise ValueError("enter a three-letter code such as USD or EUR, or leave blank")
    return text or None


def parse_choice[E: StrEnum](answer: str, choices: type[E]) -> E | None:
    text = answer.strip().lower().replace("-", "").replace(" ", "")
    if not text:
        return None
    for choice in choices:
        if text == choice.value:
            return choice
    raise ValueError("choose one of: " + " / ".join(c.value for c in choices) + " (or blank)")


# --- asking ------------------------------------------------------------------------------------


def ask_until_valid[T](prompter: Prompter, prompt: str, parse: Callable[[str], T]) -> T:
    """Ask until the answer parses. /skip and /quit work at every prompt."""
    while True:
        answer = prompter.ask(prompt)
        if answer == "/quit":
            raise StopLabelling
        if answer == "/skip":
            raise SkipPost
        try:
            return parse(answer)
        except ValueError as error:
            prompter.show(f"  ! {error}")


def ask_labels(prompter: Prompter) -> Labels:
    """The fields of one post, in order. Blank means the post does not say."""

    def text(answer: str) -> str | None:
        return answer.strip() or None

    return Labels(
        title=ask_until_valid(prompter, "title            : ", text),
        company=ask_until_valid(prompter, "company          : ", text),
        stack=ask_until_valid(prompter, "stack (a, b, c)  : ", parse_stack),
        remote_policy=ask_until_valid(
            prompter,
            "remote policy [remote/hybrid/onsite] : ",
            lambda answer: parse_choice(answer, RemotePolicy),
        ),
        salary_min=ask_until_valid(prompter, "salary min       : ", parse_amount),
        salary_max=ask_until_valid(prompter, "salary max       : ", parse_amount),
        salary_currency=ask_until_valid(prompter, "salary currency  : ", parse_currency),
        salary_period=ask_until_valid(
            prompter,
            "salary period [year/month/week/day/hour] : ",
            lambda answer: parse_choice(answer, SalaryPeriod),
        ),
        seniority=ask_until_valid(
            prompter,
            "seniority [intern/junior/mid/senior/lead/principal] : ",
            lambda answer: parse_choice(answer, Seniority),
        ),
    )


def show_post(prompter: Prompter, number: int, total: int, text: str) -> None:
    """The full text, a screenful at a time."""
    prompter.show(f"\n{'=' * 78}\nPost {number} of {total}\n{'=' * 78}")
    lines = text.splitlines() or [""]
    for start in range(0, len(lines), PAGE_LINES):
        prompter.show("\n".join(lines[start : start + PAGE_LINES]))
        if start + PAGE_LINES < len(lines):
            prompter.ask(
                f"-- more ({min(start + PAGE_LINES, len(lines))}/{len(lines)} lines), Enter --"
            )
    prompter.show("-" * 78)


def confirm_labels(prompter: Prompter, labels: Labels) -> None:
    """Show what was entered; Enter saves, 'r' enters the post again."""
    prompter.show("  " + str(labels.model_dump(mode="json", exclude_none=True)))
    answer = prompter.ask("Enter to save, r to redo this post: ")
    if answer == "/quit":
        raise StopLabelling
    if answer == "/skip":
        raise SkipPost
    if answer.lower() == "r":
        raise RedoPost


def label_post(prompter: Prompter, job_id: uuid.UUID, text: str) -> LabelEntry:
    """One post, labelled or skipped. May raise StopLabelling."""
    digest = raw_text_hash(text)
    while True:
        try:
            labels = ask_labels(prompter)
            confirm_labels(prompter, labels)
        except SkipPost:
            return LabelEntry(job_id=job_id, raw_text_hash=digest, status="skipped")
        except RedoPost:
            continue
        return LabelEntry(job_id=job_id, raw_text_hash=digest, status="labelled", labels=labels)


def run_labelling(
    prompter: Prompter, labels: LabelFile, texts: dict[uuid.UUID, str], path: Path
) -> int:
    """Label every pending post, saving after each. Returns how many were handled this run."""
    pending = labels.pending()
    handled = 0
    total = len(labels.selected)
    for job_id in pending:
        text = texts.get(job_id)
        if text is None:
            prompter.show(f"  (job {job_id} no longer exists; left for the evaluation to report)")
            continue
        number = total - len(pending) + handled + 1
        show_post(prompter, number, total, text)
        try:
            entry = label_post(prompter, job_id, text)
        except StopLabelling:
            break
        labels.posts.append(entry)
        save_label_file(path, labels)
        handled += 1
    return handled


# --- the command -------------------------------------------------------------------------------


def _count(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= number <= MAX_N:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_N}")
    return number


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a UUID") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_label",
        description="Label job posts by hand, blind, for the extraction accuracy evaluation.",
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--source-name", help="the source to take posts from")
    which.add_argument("--source-id", type=_uuid, help="the source to take posts from")
    parser.add_argument("--n", type=_count, default=DEFAULT_N, help=f"posts to label ({DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=1, help="makes the pick repeatable (1)")
    parser.add_argument("--out", type=Path, default=DEFAULT_LABELS_PATH, help="the labels file")
    return parser


def start_or_resume(
    path: Path, source_id: uuid.UUID, seed: int, rows: list[tuple[uuid.UUID, str]], n: int
) -> LabelFile:
    """The existing file (checked against the arguments) or a new selection."""
    if path.exists():
        labels = load_label_file(path)
        if labels.source_id != source_id or labels.seed != seed:
            raise ScriptError(
                f"{path} was made for source {labels.source_id} with seed {labels.seed}; "
                "use the same --source-id and --seed to resume, or another --out"
            )
        return labels
    selected = pick_posts(rows, n, seed)
    if not selected:
        raise ScriptError("the source has no posts to label")
    return LabelFile(source_id=source_id, seed=seed, selected=selected)


def run(args: argparse.Namespace, prompter: Prompter) -> int:
    settings = get_settings()
    target = database_target(settings.database_url)  # refuses the wrong database up front
    print(f"Reading (read-only) from database {target}")
    source_id = args.source_id or asyncio.run(find_source(settings.database_url, args.source_name))
    if args.out.exists():
        rows: list[tuple[uuid.UUID, str]] = []  # a resumed run keeps its selection
    else:
        rows = asyncio.run(read_source_posts(settings.database_url, source_id))
    labels = start_or_resume(args.out, source_id, args.seed, rows, args.n)
    save_label_file(args.out, labels)
    pending = labels.pending()
    print(
        f"{len(labels.posts)} of {len(labels.selected)} done, {len(pending)} to go. "
        "Commands: /skip, /quit."
    )
    texts = dict(asyncio.run(read_posts_by_id(settings.database_url, pending)))
    handled = run_labelling(prompter, labels, texts, args.out)
    left = len(labels.pending())
    print(f"\n{handled} post(s) handled this run, {left} left. Saved to {args.out}")
    return 0


def main(argv: list[str] | None = None, prompter: Prompter | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, prompter or ConsolePrompter())
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
