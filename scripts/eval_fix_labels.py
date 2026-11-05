"""One-off, deterministic fixer for the hand labels in `.local/eval/labels.json`.

It reads the label file and, for each labelled post, only the post's id and `raw_text` from the
database (read-only, the same database-name guard as the other eval scripts). It never reads a
stored extraction column, never imports the extractor or the merge, and makes no provider request.
Run by hand; CI never runs it.

Rules, applied in this order to every labelled post. Whatever a rule is not sure about is not
changed: it goes to the NEEDS_HUMAN table, which is only printed and never written.

  1. Title and company, from the headline (the first non-empty line). Used only when the headline
     has exactly one ": ", the part before it is 1 to 4 words without digits, and the part after
     it has no "$", no "(", no "full time" / "part time" / "hiring" and is under 70 characters.
     Then company = the text before the ": " and title = the text after it, exactly as written
     (whitespace trimmed, no character replaced). A company the label spells longer, and the post
     spells longer too, is not shortened. Any other headline changes neither field.
  2. Seniority, from the FINAL title: the one of intern / junior / mid / senior / lead / principal
     that is a whole word in it (none, or more than one, gives no seniority). If the title edit
     removed a level word (those six, or "staff") that the original title had, the original
     seniority label is kept and the post goes to NEEDS_HUMAN instead.
  3. Salary. An amount that is not in the text as a number (commas, dots or spaces as thousands
     separators, "k" for thousands) is removed and the post is listed. The period is the period
     word within 60 characters of a kept amount; if there is none, one within 200 characters or in
     the sentence that holds the amount; only if there is still none is the period removed. Two
     different periods at the same distance leave the period as it is (NEEDS_HUMAN). With no
     amount left, currency and period are removed too.
  4. Stack. An entry is tested against the text case-insensitively, ignoring dots, hyphens and
     spaces, under the stack dictionary's spellings, the [aliases] table of
     eval_clean_policy.toml and plain variants ("NextJS", "Node", "Postgres"). Entries found
     nowhere are removed, each with a note on whether any partial match exists. If more than 3
     entries of one post would go, none is removed and the post goes to NEEDS_HUMAN. An entry is
     never added.
  5. remote_policy is not touched. Posts with identical labels are listed, not changed.

The original is first copied to `<name>.clean-v1.json` (an existing copy is never overwritten).
The tables may show post text: they go only to your terminal. After you confirm, the file is
written and the same checks as `eval_clean_labels.py --check-labels` are run again; that report
holds post numbers and rule codes only.
"""

import argparse
import asyncio
import re
import shutil
import sys
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from app.extraction.stack import spellings
from app.extraction.text_hash import raw_text_hash
from scripts.db_guard import ScriptError, database_target
from scripts.eval_check_labels import (
    amount_spans,
    appears,
    check_labels,
    fold,
    format_report,
    stack_variants,
)
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
NEAR_CHARS = 60  # a period word this close to an amount is taken first
WIDE_CHARS = 200  # ... and only if there is none, this close or in the amount's sentence
MAX_COMPANY_WORDS = 4
MAX_TITLE_CHARS = 69  # "under 70 characters"
MAX_STACK_REMOVALS = 3  # more than this in one post: remove none, ask a human
SHOWN_LINE_CHARS = 300
SENIORITY_WORDS = ("intern", "junior", "mid", "senior", "lead", "principal")
GUARDED_LEVEL_WORDS = (*SENIORITY_WORDS, "staff")  # a title edit may not silently drop these
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
_GUARDED_LEVEL = re.compile(r"\b(" + "|".join(GUARDED_LEVEL_WORDS) + r")\b", re.IGNORECASE)
_NOT_A_TITLE = re.compile(r"full[ -]?time|part[ -]?time|hiring", re.IGNORECASE)
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
_SENTENCE_END = re.compile(r"[.!?]\s|\n")
_SEPARATORS = r"[\s.\-_]*"  # what may sit between the letters of a name: "Next.js" = "NextJS"
_VENDOR_PREFIXES = ("apache ", "amazon ", "aws ", "google ", "microsoft ", "azure ", "gcp ")


@dataclass(frozen=True)
class Change:
    post: int
    field: str
    before: str
    after: str
    rule: str


@dataclass(frozen=True)
class Review:
    """Something the fixer will not decide. Printed for the user, never written."""

    post: int
    field: str
    current: str  # the current label
    shows: str  # what the post shows: the reason and the exact lines involved


@dataclass
class PostFix:
    labels: Labels
    changes: list[Change] = field(default_factory=list)
    flags: list[tuple[str, str]] = field(default_factory=list)  # (list name, short reason)
    reviews: list[Review] = field(default_factory=list)


def _show(value: object) -> str:
    return "(none)" if value is None or value == "" else str(value)


def _replace(labels: Labels, **updates: object) -> Labels:
    """A copy with some fields replaced, validated again (the enums come back from their text)."""
    return Labels.model_validate({**labels.model_dump(mode="json"), **updates})


def fold_lines(text: str) -> str:
    """Like `fold`, but line breaks stay, so a sentence or a line can still be told apart."""
    lines = unicodedata.normalize("NFKC", text).casefold().splitlines()
    return "\n".join(" ".join(line.split()) for line in lines)


def _cut(line: str) -> str:
    line = line.strip()
    return line if len(line) <= SHOWN_LINE_CHARS else line[:SHOWN_LINE_CHARS] + "…"


# --- rule 1: title and company -------------------------------------------------------------------


def headline(text: str) -> str | None:
    """The first non-empty line of the post."""
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


def headline_problem(line: str | None) -> str | None:
    """Why a headline cannot be split into company and title; None when it can."""
    if line is None:
        return "the post has no headline"
    if line.count(": ") != 1:
        return "the headline does not have exactly one ': '"
    company, _, title = (part.strip() for part in line.partition(": "))
    if not 1 <= len(company.split()) <= MAX_COMPANY_WORDS:
        return f"the part before ': ' is not 1 to {MAX_COMPANY_WORDS} words"
    if any(char.isdigit() for char in company):
        return "the part before ': ' has a digit"
    if not title:
        return "the part after ': ' is empty"
    if "$" in title or "(" in title:
        return "the part after ': ' has a '$' or a '('"
    if _NOT_A_TITLE.search(title):
        return "the part after ': ' mentions full time, part time or hiring"
    if len(title) > MAX_TITLE_CHARS:
        return "the part after ': ' is 70 characters or longer"
    return None


def split_headline(text: str) -> tuple[str, str] | None:
    """(company, title) from a "Company: Role" headline; None when the checks do not pass."""
    line = headline(text)
    if headline_problem(line) is not None or line is None:
        return None
    company, _, title = line.partition(": ")
    return company.strip(), title.strip()


def _longer_spelling(current: str | None, company: str, folded: str) -> bool:
    """True when the label spells the company longer than the headline and the post does too."""
    if not current or len(current) <= len(company):
        return False
    long, short = fold(current), fold(company)
    return long.startswith(short) and not long[len(short)].isalnum() and long in folded


def _headline_step(
    number: int, labels: Labels, text: str, folded: str, reviews: list[Review]
) -> Labels:
    line = headline(text)
    problem = headline_problem(line)
    parts = split_headline(text)
    if parts is None:
        current = f"title: {_show(labels.title)}; company: {_show(labels.company)}"
        shows = f"{problem}. headline: {_show(line)}"
        reviews.append(Review(number, "title, company", current, shows))
        return labels
    company, title = parts
    if _longer_spelling(labels.company, company, folded):
        company = labels.company or company
    return _replace(labels, company=company, title=title)


# --- rule 2: seniority ---------------------------------------------------------------------------


def seniority_from_title(title: str | None) -> str | None:
    """The one seniority word in the title, else None (Staff, Director, Manager do not map)."""
    found = {match.group(1).lower() for match in _SENIORITY.finditer(title or "")}
    return found.pop() if len(found) == 1 else None


def level_words(title: str | None) -> set[str]:
    return {match.group(1).lower() for match in _GUARDED_LEVEL.finditer(title or "")}


def _seniority_step(
    number: int, original: Labels, labels: Labels, line: str | None, reviews: list[Review]
) -> Labels:
    """Seniority from the final title, unless the title edit dropped a level word."""
    removed = level_words(original.title) - level_words(labels.title)
    if removed:
        current = original.seniority.value if original.seniority else None
        shows = (
            f"the title edit drops {', '.join(sorted(removed))}: label title "
            f"{_show(original.title)!r} -> {_show(labels.title)!r}. headline: {_show(line)}"
        )
        reviews.append(Review(number, "seniority", _show(current), shows))
        return labels
    return _replace(labels, seniority=seniority_from_title(labels.title))


# --- rule 3: salary ------------------------------------------------------------------------------


def _periods_in(windows: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for window in windows:
        for match in _PERIOD_WORD.finditer(window):
            found.add(_PERIOD_OF_WORD[match.group(1) or match.group(2)])
    return found


def _sentence(text: str, start: int, end: int) -> str:
    left = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        left = match.end()
    after = _SENTENCE_END.search(text, end)
    return text[left : after.start() if after else len(text)]


def periods_near(amounts: set[int], text: str) -> set[str]:
    """Periods named next to the amounts: within 60 characters, else within 200 or in the sentence.

    `text` is folded (lower case); line breaks may be kept so a sentence ends at one.
    """
    spans = [(start, end) for value, start, end in amount_spans(text) if value in amounts]
    near = _periods_in([text[max(0, s - NEAR_CHARS) : e + NEAR_CHARS] for s, e in spans])
    if near:
        return near
    wide = [text[max(0, s - WIDE_CHARS) : e + WIDE_CHARS] for s, e in spans]
    return _periods_in([*wide, *(_sentence(text, s, e) for s, e in spans)])


def lines_with_amounts(text: str, amounts: set[int]) -> list[str]:
    """The post's own lines (not folded) that hold one of the amounts."""
    return [
        _cut(line)
        for line in text.splitlines()
        if any(value in amounts for value, _, _ in amount_spans(fold(line)))
    ]


def _salary_step(
    number: int,
    labels: Labels,
    text: str,
    lined: str,
    flags: list[tuple[str, str]],
    reviews: list[Review],
) -> Labels:
    present = {value for value, _, _ in amount_spans(lined)}
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
    periods = periods_near(kept, lined)
    if len(periods) > 1:
        current = labels.salary_period.value if labels.salary_period else None
        shows = (
            f"periods {', '.join(sorted(periods))} are equally close to the amounts. lines: "
            + (" // ".join(lines_with_amounts(text, kept)) or "(none)")
        )
        reviews.append(Review(number, "salary_period", _show(current), shows))
        period = current
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


def plain_variants(name: str) -> set[str]:
    """Shorter everyday spellings: "Node.js" -> "node", "Apache Kafka" -> "kafka"."""
    out = {name}
    for prefix in _VENDOR_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            out.add(name[len(prefix) :])
    for suffix in (".js", " js", "-js"):
        if name.endswith(suffix) and len(name) > len(suffix):
            out.add(name[: -len(suffix)])
    return out


@lru_cache(maxsize=512)
def _flexible(variant: str) -> re.Pattern[str] | None:
    """The name as a whole word, letters in order, with dots, hyphens and spaces optional."""
    chars = [char for char in variant if char.isalnum() or char in "+#"]
    if not chars:
        return None
    body = _SEPARATORS.join(re.escape(char) for char in chars)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def entry_in_text(entry: str, folded: str, aliases: Mapping[str, tuple[str, ...]]) -> bool:
    names = entry_names(entry, aliases) | stack_variants(entry)
    names |= {variant for name in set(names) for variant in plain_variants(name)}
    for name in names:
        pattern = _flexible(name)
        if pattern is not None and pattern.search(folded):
            return True
    return False


def partial_match(entry: str, folded: str) -> str | None:
    """A word of the entry that is in the text, or the entry inside a longer word."""
    name = fold(entry)
    for word in re.split(r"[^a-z0-9+#]+", name):
        if len(word) >= 3 and appears(word, folded):
            return f"word {word!r}"
    squashed = re.sub(r"[^a-z0-9+#]", "", name)
    if len(squashed) >= 4 and squashed in re.sub(r"[^a-z0-9+#]", "", folded):
        return f"{squashed!r} inside a longer word"
    return None


def _partial_note(entry: str, folded: str) -> str:
    found = partial_match(entry, folded)
    return f"partial match: {found}" if found else "no partial match"


def _stack_step(
    number: int,
    labels: Labels,
    folded: str,
    aliases: Mapping[str, tuple[str, ...]],
    out: PostFix,
) -> Labels:
    missing = [e for e in labels.stack if not entry_in_text(e, folded, aliases)]
    if not missing:
        return labels
    if len(missing) > MAX_STACK_REMOVALS:
        current = "; ".join(missing)
        shows = f"{len(missing)} entries are not found; none removed. " + "; ".join(
            f"{entry}: {_partial_note(entry, folded)}" for entry in missing
        )
        out.reviews.append(Review(number, "stack", current, shows))
        return labels
    for entry in missing:
        after = f"(removed; {_partial_note(entry, folded)})"
        out.changes.append(Change(number, "stack", entry, after, RULE_STACK))
    return _replace(labels, stack=[e for e in labels.stack if e not in missing])


# --- one post ------------------------------------------------------------------------------------


def _record(
    out: list[Change], number: int, before: Labels, after: Labels, rule: str, fields: Sequence[str]
) -> None:
    old, new = before.values(), after.values()
    for name in fields:
        if getattr(old, name) != getattr(new, name):
            out.append(
                Change(number, name, _show(getattr(old, name)), _show(getattr(new, name)), rule)
            )


def fix_post(
    number: int, labels: Labels, text: str, aliases: Mapping[str, tuple[str, ...]]
) -> PostFix:
    """Apply rules 1 to 4 to one post, in order. Reads only the labels and the post text."""
    result = PostFix(labels)
    folded, lined = fold(text), fold_lines(text)

    def step(rule: str, fields: Sequence[str], new: Labels) -> None:
        _record(result.changes, number, result.labels, new, rule, fields)
        result.labels = new

    new = _headline_step(number, labels, text, folded, result.reviews)
    step(RULE_HEADLINE, ("title", "company"), new)
    new = _seniority_step(number, labels, result.labels, headline(text), result.reviews)
    step(RULE_SENIORITY, ("seniority",), new)

    salary = _salary_step(number, result.labels, text, lined, result.flags, result.reviews)
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

    result.labels = _stack_step(number, result.labels, folded, aliases, result)
    return result


# --- the whole file ------------------------------------------------------------------------------


@dataclass
class FixPlan:
    fixed: LabelFile
    changes: list[Change]
    lists: dict[str, list[tuple[int, str]]]  # list name -> (post number, reason)
    reviews: list[Review]


def plan_fixes(
    label_file: LabelFile, texts: Mapping[uuid.UUID, str], aliases: Mapping[str, tuple[str, ...]]
) -> FixPlan:
    posts: list[LabelEntry] = []
    changes: list[Change] = []
    reviews: list[Review] = []
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
            reviews.append(Review(number, "(whole post)", "(not touched)", why))
            posts.append(entry)
            continue
        fixed = fix_post(number, entry.labels, text, aliases)
        changes.extend(fixed.changes)
        reviews.extend(fixed.reviews)
        for name, reason in fixed.flags:
            lists[name].append((number, reason))
        if not fixed.labels.stack:
            lists[INFO].append((number, "stack is empty"))
        posts.append(entry.model_copy(update={"labels": fixed.labels}))
    fixed_file = label_file.model_copy(update={"posts": posts})
    _list_identical(fixed_file, lists)
    return FixPlan(fixed_file, changes, dict(lists), reviews)


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


def _identity_of(entry: LabelEntry) -> tuple[uuid.UUID, str, str]:
    return entry.job_id, entry.raw_text_hash, entry.status


def verify_only_allowed_changes(before: LabelFile, after: LabelFile) -> None:
    """Refuse to write unless only the fixable label fields differ and no stack entry was added."""
    if before.model_dump(exclude={"posts"}) != after.model_dump(exclude={"posts"}):
        raise ScriptError("refusing to write: the file header changed")
    if len(before.posts) != len(after.posts):
        raise ScriptError("refusing to write: the number of posts changed")
    for old, new in zip(before.posts, after.posts, strict=True):
        if _identity_of(old) != _identity_of(new):
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


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [" | ".join(cell.ljust(width) for cell, width in zip(header, widths, strict=True))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    )
    return lines


def format_changes(changes: Sequence[Change]) -> list[str]:
    rows = [(str(c.post), c.field, c.before, c.after, c.rule) for c in changes]
    return _table(("post", "field", "before", "after", "rule"), rows)


def format_reviews(reviews: Sequence[Review]) -> list[str]:
    """The NEEDS_HUMAN table. Nothing in it is written to the labels file."""
    rows = [(str(r.post), r.field, r.current, r.shows) for r in sorted(reviews, key=_review_key)]
    return _table(("post", "field", "current label", "what the post shows"), rows)


def _review_key(review: Review) -> tuple[int, str]:
    return review.post, review.field


def format_counts(changes: Sequence[Change]) -> list[str]:
    lines = []
    for rule in RULES:
        mine = [c for c in changes if c.rule == rule]
        posts = len({c.post for c in mine})
        lines.append(f"  rule {rule}: {len(mine)} change(s) in {posts} post(s)")
    return lines


def format_lists(lists: Mapping[str, Sequence[tuple[int, str]]]) -> list[str]:
    lines = []
    for name in (SALARY_REMOVED, INFO, IDENTICAL):
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


def _show_plan(plan: FixPlan, session: Session) -> None:
    if plan.changes:
        session.out("\n".join(format_changes(plan.changes)))
    session.out(f"\n{len(plan.changes)} change(s):")
    for line in format_counts(plan.changes):
        session.out(line)
    for line in format_lists(plan.lists):
        session.out(line)
    session.out(f"  {NEEDS_HUMAN}: {len(plan.reviews)} item(s); nothing in that table is written")
    if plan.reviews:
        session.out("\n" + NEEDS_HUMAN + "\n" + "\n".join(format_reviews(plan.reviews)))


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
    _show_plan(plan, session)
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
