"""One-off, deterministic fixer for the hand labels in `.local/eval/labels.json`.

It reads the label file and, for each labelled post, only the post's id and `raw_text` from the
database (read-only, the same database-name guard as the other eval scripts). It never reads a
stored extraction column, never imports the extractor or the merge, and makes no provider request.
Run by hand; CI never runs it.

Rules, applied in this order to every labelled post:

  1. Title and company. The first non-empty line of the text is the headline, usually
     "Company: Role". With a ": " in it, company = the text before the first ": " and title = the
     text after it, both exactly as written (whitespace trimmed). Without one, neither field is
     changed and the post is listed as NEEDS_HUMAN.
  2. Seniority. The one of intern / junior / mid / senior / lead / principal that appears as a
     whole word in the (new) title; none, or more than one, gives no seniority.
  3. Salary. An amount that is not in the text as a number (commas, dots or spaces as thousands
     separators, "k" for thousands) is removed and the post is listed. The period is the period
     word (hour, year, month, week, day and their usual forms) within 60 characters of a kept
     amount, else none. With no amount left, currency and period are removed too.
  4. Stack. An entry is removed only when neither it nor any alias of its canonical name (the stack
     dictionary's spellings and the [aliases] table of eval_clean_policy.toml) appears in the text
     as a whole word. An entry is never added.
  5. remote_policy is not touched. Posts with identical labels are listed, not changed.

The original is first copied to `<name>.clean-v1.json` (an existing copy is never overwritten).
The table of changes may show post text: it goes only to your terminal. After you confirm, the
file is written and the same checks as `eval_clean_labels.py --check-labels` are run again; that
report holds post numbers and rule codes only.
"""

import argparse
import asyncio
import re
import shutil
import sys
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.extraction.stack import spellings
from app.extraction.text_hash import raw_text_hash
from scripts.db_guard import ScriptError, database_target
from scripts.eval_check_labels import amount_spans, appears, check_labels, fold, format_report
from scripts.eval_check_labels import identity as label_identity
from scripts.eval_clean_policy import DEFAULT_POLICY_PATH, load_policy
from scripts.eval_label import read_posts_by_id
from scripts.eval_labelfile import (
    DEFAULT_LABELS_PATH,
    LabelEntry,
    LabelFile,
    Labels,
    load_label_file,
    save_label_file,
)

BACKUP_SUFFIX = ".clean-v1"
PERIOD_WINDOW = 60
SENIORITY_WORDS = ("intern", "junior", "mid", "senior", "lead", "principal")
# Fields the fixer may change. Everything else in the file, remote_policy included, must stay equal.
CHANGEABLE = frozenset(
    {"title", "company", "stack", "seniority", "salary_min", "salary_max"}
    | {"salary_currency", "salary_period"}
)

RULE_HEADLINE = "1 headline"
RULE_SENIORITY = "2 seniority word"
RULE_AMOUNT = "3 amount not in text"
RULE_PERIOD = "3 period word"
RULE_NO_AMOUNTS = "3 no amount left"
RULE_STACK = "4 not in text"
RULES = (RULE_HEADLINE, RULE_SENIORITY, RULE_AMOUNT, RULE_PERIOD, RULE_NO_AMOUNTS, RULE_STACK)

NEEDS_HUMAN = "NEEDS_HUMAN"
SALARY_REMOVED = "SALARY_AMOUNT_REMOVED"
INFO = "INFO"
IDENTICAL = "IDENTICAL"

_SENIORITY = re.compile(r"\b(" + "|".join(SENIORITY_WORDS) + r")\b", re.IGNORECASE)
_PERIOD_OF_WORD = {
    **dict.fromkeys(("hour", "hourly", "hr"), "hour"),
    **dict.fromkeys(("year", "yearly", "annual", "annually", "yr"), "year"),
    **dict.fromkeys(("month", "monthly"), "month"),
    **dict.fromkeys(("week", "weekly"), "week"),
    **dict.fromkeys(("day", "daily"), "day"),
}
# "hr" and "yr" count only when written like a unit ("/hr", "per hr", "50hr"), so "HR" (human
# resources) next to an amount is not read as "hourly".
_PERIOD_WORD = re.compile(
    r"\b(hourly|hour|yearly|year|annually|annual|monthly|month|weekly|week|daily|day)\b"
    r"|(?:(?<=/)|(?<=per )|(?<=\d )|(?<=\d))(hr|yr)\b"
)


@dataclass(frozen=True)
class Change:
    post: int
    field: str
    before: str
    after: str
    rule: str


@dataclass
class PostFix:
    labels: Labels
    changes: list[Change] = field(default_factory=list)
    flags: list[tuple[str, str]] = field(default_factory=list)  # (list name, short reason)


def _show(value: object) -> str:
    return "(none)" if value is None or value == "" else str(value)


def _replace(labels: Labels, **updates: object) -> Labels:
    """A copy with some fields replaced, validated again (the enums come back from their text)."""
    return Labels.model_validate({**labels.model_dump(mode="json"), **updates})


# --- rule 1: title and company -------------------------------------------------------------------


def headline(text: str) -> str | None:
    """The first non-empty line of the post."""
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


def split_headline(text: str) -> tuple[str, str] | None:
    """(company, title) from a "Company: Role" headline; None when it is not of that form."""
    line = headline(text)
    if line is None or ": " not in line:
        return None
    company, _, title = line.partition(": ")
    company, title = company.strip(), title.strip()
    return (company, title) if company and title else None


def _headline_step(labels: Labels, text: str, flags: list[tuple[str, str]]) -> Labels:
    parts = split_headline(text)
    if parts is None:
        flags.append((NEEDS_HUMAN, "headline is not 'Company: Role'; title and company unchanged"))
        return labels
    company, title = parts
    return _replace(labels, company=company, title=title)


# --- rule 2: seniority ---------------------------------------------------------------------------


def seniority_from_title(title: str | None) -> str | None:
    """The one seniority word in the title, else None (Staff, Director, Manager do not map)."""
    found = {match.group(1).lower() for match in _SENIORITY.finditer(title or "")}
    return found.pop() if len(found) == 1 else None


# --- rule 3: salary ------------------------------------------------------------------------------


def periods_near(amounts: set[int], folded: str) -> set[str]:
    """Periods named within 60 characters of any occurrence of the given amounts."""
    found: set[str] = set()
    for value, start, end in amount_spans(folded):
        if value not in amounts:
            continue
        window = folded[max(0, start - PERIOD_WINDOW) : end + PERIOD_WINDOW]
        for match in _PERIOD_WORD.finditer(window):
            found.add(_PERIOD_OF_WORD[match.group(1) or match.group(2)])
    return found


def _salary_step(labels: Labels, folded: str, flags: list[tuple[str, str]]) -> Labels:
    present = {value for value, _, _ in amount_spans(folded)}
    minimum, maximum = labels.salary_min, labels.salary_max
    if minimum is not None and minimum not in present:
        minimum = None
        flags.append((SALARY_REMOVED, "salary_min is not in the text"))
    if maximum is not None and maximum not in present:
        maximum = None
        flags.append((SALARY_REMOVED, "salary_max is not in the text"))
    kept = {amount for amount in (minimum, maximum) if amount is not None}
    if not kept:
        return _replace(
            labels, salary_min=None, salary_max=None, salary_currency=None, salary_period=None
        )
    periods = periods_near(kept, folded)
    if len(periods) > 1:
        flags.append(
            (NEEDS_HUMAN, "more than one salary period near the amounts; period unchanged")
        )
        period = labels.salary_period.value if labels.salary_period else None
    else:
        period = next(iter(periods), None)
    return _replace(labels, salary_min=minimum, salary_max=maximum, salary_period=period)


# --- rule 4: stack -------------------------------------------------------------------------------


def entry_names(entry: str, aliases: Mapping[str, tuple[str, ...]]) -> set[str]:
    """Every folded spelling that counts as this stack entry being in the text."""
    names = {fold(entry)}
    while True:
        grown = set(names)
        for name in names:
            grown.update(fold(spelling) for spelling in spellings(name))
        for alias, targets in aliases.items():
            folded_targets = {fold(target) for target in targets}
            if len(targets) == 1 and (alias in grown or folded_targets & grown):
                grown.update({fold(alias), *folded_targets})
            elif len(targets) > 1 and alias in grown:  # "React Vite": either product will do
                grown.update(folded_targets)
        if grown == names:
            return names
        names = grown


def _stack_step(labels: Labels, folded: str, aliases: Mapping[str, tuple[str, ...]]) -> Labels:
    kept = [
        entry
        for entry in labels.stack
        if any(appears(name, folded) for name in entry_names(entry, aliases))
    ]
    return _replace(labels, stack=kept)


# --- one post ------------------------------------------------------------------------------------


def _record(
    out: list[Change], number: int, before: Labels, after: Labels, rule: str, fields: Sequence[str]
) -> None:
    old, new = before.values(), after.values()
    for name in fields:
        if name == "stack":
            out.extend(
                Change(number, name, entry, "(removed)", rule)
                for entry in old.stack
                if entry not in new.stack
            )
        elif getattr(old, name) != getattr(new, name):
            out.append(
                Change(number, name, _show(getattr(old, name)), _show(getattr(new, name)), rule)
            )


def fix_post(
    number: int, labels: Labels, text: str, aliases: Mapping[str, tuple[str, ...]]
) -> PostFix:
    """Apply rules 1 to 4 to one post, in order. Reads only the labels and the post text."""
    result = PostFix(labels)
    folded = fold(text)

    def step(rule: str, fields: Sequence[str], new: Labels) -> None:
        _record(result.changes, number, result.labels, new, rule, fields)
        result.labels = new

    step(RULE_HEADLINE, ("title", "company"), _headline_step(labels, text, result.flags))
    seniority = seniority_from_title(result.labels.title)
    step(RULE_SENIORITY, ("seniority",), _replace(result.labels, seniority=seniority))

    salary = _salary_step(result.labels, folded, result.flags)
    rest = (
        RULE_PERIOD if (salary.salary_min, salary.salary_max) != (None, None) else RULE_NO_AMOUNTS
    )
    _record(
        result.changes, number, result.labels, salary, RULE_AMOUNT, ("salary_min", "salary_max")
    )
    _record(
        result.changes, number, result.labels, salary, rest, ("salary_currency", "salary_period")
    )
    result.labels = salary

    step(RULE_STACK, ("stack",), _stack_step(result.labels, folded, aliases))
    return result


# --- the whole file ------------------------------------------------------------------------------


@dataclass
class FixPlan:
    fixed: LabelFile
    changes: list[Change]
    lists: dict[str, list[tuple[int, str]]]  # list name -> (post number, reason)


def plan_fixes(
    label_file: LabelFile, texts: Mapping[uuid.UUID, str], aliases: Mapping[str, tuple[str, ...]]
) -> FixPlan:
    posts: list[LabelEntry] = []
    changes: list[Change] = []
    lists: dict[str, list[tuple[int, str]]] = defaultdict(list)
    number = 0
    for entry in label_file.posts:
        if entry.labels is None:
            posts.append(entry)
            continue
        number += 1
        text = texts.get(entry.job_id)
        if text is None or raw_text_hash(text) != entry.raw_text_hash:
            why = "post not found" if text is None else "post text changed since it was labelled"
            lists[NEEDS_HUMAN].append((number, f"{why}; not touched"))
            posts.append(entry)
            continue
        fixed = fix_post(number, entry.labels, text, aliases)
        changes.extend(fixed.changes)
        for name, reason in fixed.flags:
            lists[name].append((number, reason))
        if not fixed.labels.stack:
            lists[INFO].append((number, "stack is empty"))
        posts.append(entry.model_copy(update={"labels": fixed.labels}))
    fixed_file = label_file.model_copy(update={"posts": posts})
    _list_identical(fixed_file, lists)
    return FixPlan(fixed_file, changes, dict(lists))


def _list_identical(label_file: LabelFile, lists: dict[str, list[tuple[int, str]]]) -> None:
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    labelled = [entry.labels for entry in label_file.posts if entry.labels is not None]
    for number, labels in enumerate(labelled, start=1):
        groups[label_identity(labels)].append(number)
    for numbers in groups.values():
        if len(numbers) > 1:
            lists[IDENTICAL].append(
                (numbers[0], "same labels as posts " + ", ".join(str(n) for n in numbers[1:]))
            )


def verify_only_allowed_changes(before: LabelFile, after: LabelFile) -> None:
    """Refuse to write unless only the fixable label fields differ and no stack entry was added."""
    if before.model_dump(exclude={"posts"}) != after.model_dump(exclude={"posts"}):
        raise ScriptError("refusing to write: the file header changed")
    if len(before.posts) != len(after.posts):
        raise ScriptError("refusing to write: the number of posts changed")
    for old, new in zip(before.posts, after.posts, strict=True):
        if (old.job_id, old.raw_text_hash, old.status) != (
            new.job_id,
            new.raw_text_hash,
            new.status,
        ):
            raise ScriptError("refusing to write: an id, hash or status changed")
        if old.labels is None or new.labels is None:
            if old.labels is not new.labels:
                raise ScriptError("refusing to write: a skipped post gained labels")
            continue
        fixed_part = old.labels.model_dump(exclude=set(CHANGEABLE))
        if fixed_part != new.labels.model_dump(exclude=set(CHANGEABLE)):
            raise ScriptError("refusing to write: a field outside the fixable ones changed")
        remaining = iter(old.labels.stack)  # new stack must be old stack with entries removed
        if not all(entry in remaining for entry in new.labels.stack):
            raise ScriptError("refusing to write: a stack entry was added or reordered")


# --- output --------------------------------------------------------------------------------------


def format_changes(changes: Sequence[Change]) -> list[str]:
    header = ("post", "field", "before", "after", "rule")
    rows = [(str(c.post), c.field, c.before, c.after, c.rule) for c in changes]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [" | ".join(cell.ljust(width) for cell, width in zip(header, widths, strict=True))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    )
    return lines


def format_counts(changes: Sequence[Change]) -> list[str]:
    lines = []
    for rule in RULES:
        mine = [c for c in changes if c.rule == rule]
        posts = len({c.post for c in mine})
        lines.append(f"  rule {rule}: {len(mine)} change(s) in {posts} post(s)")
    return lines


def format_lists(lists: Mapping[str, Sequence[tuple[int, str]]]) -> list[str]:
    lines = []
    for name in (NEEDS_HUMAN, SALARY_REMOVED, INFO, IDENTICAL):
        for number, reason in sorted(lists.get(name, [])):
            lines.append(f"  {name}: post {number}: {reason}")
    return lines


# --- files and the command -----------------------------------------------------------------------


def back_up(path: Path) -> tuple[Path, bool]:
    """Copy the labels to `<name>.clean-v1.json` first. An existing copy is never overwritten."""
    backup = path.with_name(f"{path.stem}{BACKUP_SUFFIX}{path.suffix}")
    if backup.exists():
        return backup, False
    shutil.copyfile(path, backup)
    return backup, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_fix_labels",
        description="Correct the labels from the post text (title, seniority, salary, stack).",
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    return parser


@dataclass
class Session:
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = field(default=print)


def run_fix(labels_path: Path, settings: Settings, session: Session) -> int:
    """Plan, show, ask, write, re-check. Reads the database read-only."""
    if not labels_path.is_file():
        raise ScriptError(f"{labels_path} does not exist")
    label_file = load_label_file(labels_path)
    target = database_target(settings.database_url)  # refuses the wrong database up front
    backup, created = back_up(labels_path)
    session.out(f"Backup: {backup} ({'written' if created else 'already there, kept as it is'})")
    session.out(f"Reading post text (read-only) from database {target}")
    ids = [entry.job_id for entry in label_file.posts if entry.labels is not None]
    texts = dict(asyncio.run(read_posts_by_id(settings.database_url, ids)))
    aliases = load_policy(DEFAULT_POLICY_PATH).aliases

    plan = plan_fixes(label_file, texts, aliases)
    verify_only_allowed_changes(label_file, plan.fixed)
    if plan.changes:
        session.out("\n".join(format_changes(plan.changes)))
    session.out(f"\n{len(plan.changes)} change(s):")
    for line in format_counts(plan.changes):
        session.out(line)
    for line in format_lists(plan.lists):
        session.out(line)
    if not plan.changes:
        session.out("Nothing to change.")
        return 0

    try:
        answer = session.ask("Write these changes? [y/N] ")
    except EOFError:
        session.out("\nInput ended. Nothing was changed.")
        return 1
    if answer.strip().lower() not in ("y", "yes"):
        session.out("Nothing was changed.")
        return 1

    save_label_file(labels_path, plan.fixed)
    session.out("Written. Checking the labels against the post text again (codes only):")
    report = check_labels(load_label_file(labels_path), texts)
    for line in format_report(report):
        session.out(line)
    return 0 if not report.problems else 1


def main(argv: Sequence[str] | None = None, session: Session | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_fix(args.labels, get_settings(), session or Session())
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
