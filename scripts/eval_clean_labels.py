"""One-off helper: tidy the stack labels in `.local/eval/labels.json` before scoring.

Run by hand, never by CI or app code. It reads and writes only the labels file; it does not touch
the database, any stored extraction or any provider. It changes the `stack` list of each labelled
post and nothing else (ids, hashes, titles, companies, salary, remote policy and seniority stay
byte-for-byte as they were).

Order of events: copy `labels.json` to `labels.raw.json` (never overwritten), show every proposed
rewrite as a table (before, after, rule), ask keep/drop for each entry that is not clearly a named
product, then ask for a final yes before writing anything.

    uv run python -m scripts.eval_clean_labels
"""

import argparse
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.extraction.stack import canonical_name
from scripts.db_guard import ScriptError
from scripts.eval_labelfile import (
    DEFAULT_LABELS_PATH,
    LabelEntry,
    LabelFile,
    load_label_file,
    save_label_file,
)

MAX_ENTRY_CHARS = 30
MAX_ENTRY_WORDS = 4  # "Amazon Web Services", "Google Cloud Platform"; more reads as a sentence

# Words that name an idea, not a product. An entry holding one is asked about unless the stack
# dictionary knows the whole entry. This is a prompt for the human, never a decision.
GENERIC_WORDS = frozenset(
    """
    experience knowledge skills skill development design architecture engineering management
    testing concepts principles practices systems solutions tools technologies platforms
    pipelines automation security networking scalability performance optimization optimisation
    communication leadership agile scrum devops microservices algorithms structures
    fundamentals proficiency familiarity understanding background
    """.split()
)

_BRACKET_GROUP = re.compile(r"\s*[(\[][^()\[\]]*[)\]]")
_SPACED_SLASH = re.compile(r"\s+/\s+")
_SENTENCE_PUNCTUATION = re.compile(r"[.!?;:](\s|$)")
_BRACKETS = "()[]{}"
_EDGE_JUNK = " \t.,;:-"


@dataclass
class Item:
    """One proposed stack entry. `flag` is why the human is asked; `keep` is their answer."""

    name: str
    flag: str | None = None
    keep: bool = True


@dataclass
class Proposal:
    """What one original entry (or several, rejoined) becomes."""

    post: int  # 1-based position among the labelled posts
    before: str
    items: list[Item]
    rules: list[str]

    @property
    def changed(self) -> bool:
        return [i.name for i in self.items] != [self.before] or any(
            rule in ("duplicate", "empty") for rule in self.rules
        )

    @property
    def flagged(self) -> bool:
        return any(item.flag for item in self.items)


# --- lint: the one definition of a clean stack entry --------------------------------------------


def lint_entry(name: str) -> list[str]:
    """Why `name` is not a clean stack entry (empty list = clean)."""
    problems = []
    if not name.strip():
        problems.append("empty")
    if any(char in name for char in _BRACKETS):
        problems.append("brackets")
    if len(name) > MAX_ENTRY_CHARS:
        problems.append("too_long")
    if len(name.split()) > MAX_ENTRY_WORDS or _SENTENCE_PUNCTUATION.search(name):
        problems.append("sentence")
    return problems


@dataclass(frozen=True)
class LintIssue:
    post: int
    entry: str
    problems: tuple[str, ...]


def lint_file(label_file: LabelFile) -> list[LintIssue]:
    issues = []
    for number, entry in enumerate(_labelled(label_file), start=1):
        assert entry.labels is not None
        for name in entry.labels.stack:
            problems = lint_entry(name)
            if problems:
                issues.append(LintIssue(number, name, tuple(problems)))
    return issues


def _labelled(label_file: LabelFile) -> list[LabelEntry]:
    return [entry for entry in label_file.posts if entry.labels is not None]


# --- the rewrite rules (pure) -------------------------------------------------------------------


def _depth(text: str) -> int:
    return text.count("(") + text.count("[") - text.count(")") - text.count("]")


def rejoin_split_brackets(entries: Sequence[str]) -> list[list[str]]:
    """Group entries so a bracket that a comma split ("AWS (SageMaker", "EKS)") is whole again."""
    groups: list[list[str]] = []
    pending: list[str] = []
    for entry in entries:
        pending.append(entry)
        if _depth(", ".join(pending)) <= 0:
            groups.append(pending)
            pending = []
    if pending:  # a bracket that never closes: keep what there is, the next rule cuts it
        groups.append(pending)
    return groups


def strip_brackets(text: str) -> str:
    """Remove bracketed annotations, then any bracket left unbalanced ("AWS (SageMaker")."""
    previous = None
    while previous != text:
        previous = text
        text = _BRACKET_GROUP.sub("", text)
    for opener in "([":
        if opener in text:
            text = text[: text.index(opener)]
    return text.replace(")", "").replace("]", "").strip()


def split_slash_products(text: str) -> list[str]:
    """Split "Terraform / Pulumi". Only a slash with spaces on both sides: "CI/CD" is one name."""
    return _SPACED_SLASH.split(text)


def tidy(text: str) -> str:
    return " ".join(text.split()).strip(_EDGE_JUNK)


def flag_reason(name: str) -> str | None:
    """Why this entry may not be a named product. None for a dictionary name or a plausible one."""
    problems = [p for p in lint_entry(name) if p != "empty"]
    if problems:
        return ", ".join(problems)
    if canonical_name(name) is not None:
        return None
    words = name.split()
    if any(word.casefold().strip(".,") in GENERIC_WORDS for word in words):
        return "looks like a concept"
    if len(words) > 1 and name == name.casefold():
        return "looks like a description"
    return None


def propose_for_post(post: int, stack: Sequence[str]) -> list[Proposal]:
    proposals = []
    seen: set[str] = set()
    for group in rejoin_split_brackets(stack):
        before = ", ".join(group)
        rules = ["rejoin-brackets"] if len(group) > 1 else []
        without = strip_brackets(before)
        if without != before.strip():
            rules.append("strip-brackets")
        parts = [tidy(part) for part in split_slash_products(without)]
        if len(parts) > 1:
            rules.append("split-slash")
        items = []
        for part in parts:
            if not part:
                rules.append("empty")
            elif part.casefold() in seen:
                rules.append("duplicate")
            else:
                seen.add(part.casefold())
                items.append(Item(part, flag_reason(part)))
        proposals.append(Proposal(post, before, items, rules))
    return proposals


def build_proposals(label_file: LabelFile) -> list[Proposal]:
    proposals = []
    for number, entry in enumerate(_labelled(label_file), start=1):
        assert entry.labels is not None
        proposals.extend(propose_for_post(number, entry.labels.stack))
    return proposals


def apply_proposals(label_file: LabelFile, proposals: Sequence[Proposal]) -> LabelFile:
    """A copy of the file whose only difference is each labelled post's `stack`."""
    kept: dict[int, list[str]] = {}
    for proposal in proposals:
        kept.setdefault(proposal.post, []).extend(i.name for i in proposal.items if i.keep)
    number = 0
    posts = []
    for entry in label_file.posts:
        if entry.labels is None:
            posts.append(entry)
            continue
        number += 1
        labels = entry.labels.model_copy(update={"stack": kept.get(number, [])})
        posts.append(entry.model_copy(update={"labels": labels}))
    cleaned = label_file.model_copy(update={"posts": posts})
    _require_only_stacks_changed(label_file, cleaned)
    return cleaned


def _require_only_stacks_changed(before: LabelFile, after: LabelFile) -> None:
    def without_stacks(label_file: LabelFile) -> dict[str, object]:
        data = label_file.model_dump(mode="json")
        for post in data["posts"]:
            if post["labels"] is not None:
                post["labels"].pop("stack")
        return dict(data)

    if without_stacks(before) != without_stacks(after):
        raise ScriptError("internal error: a field other than the stack would change; not writing")


# --- the table and the questions ----------------------------------------------------------------


def format_table(proposals: Sequence[Proposal]) -> str:
    rows = [("post", "before", "after", "rule")]
    for p in proposals:
        if not (p.changed or p.flagged):
            continue
        after = " ; ".join(i.name for i in p.items) or "(dropped)"
        rule = ", ".join(p.rules)
        flags = "; ".join(f"ASK: {i.flag}" for i in p.items if i.flag)
        rows.append((str(p.post), p.before, after, ", ".join(x for x in (rule, flags) if x)))
    widths = [max(len(row[col]) for row in rows) for col in range(4)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(line.rstrip() for line in lines)


def ask_keep_or_drop(proposals: Sequence[Proposal], ask: Callable[[str], str]) -> bool:
    """Ask about each flagged entry. False when the user quits (nothing is written then)."""
    flagged = [(p, i) for p in proposals for i in p.items if i.flag]
    for position, (proposal, item) in enumerate(flagged, start=1):
        question = (
            f"[{position}/{len(flagged)}] post {proposal.post}: {item.name!r} ({item.flag}). "
            "Keep or drop? [k/d/q] "
        )
        while True:
            answer = ask(question).strip().lower()
            if answer in ("k", "keep"):
                break
            if answer in ("d", "drop"):
                item.keep = False
                break
            if answer in ("q", "quit"):
                return False
    return True


# --- files and the command ----------------------------------------------------------------------


def back_up(path: Path) -> tuple[Path, bool]:
    """Copy the labels to `<name>.raw.json` before anything else. An existing backup is kept."""
    backup = path.with_name(f"{path.stem}.raw{path.suffix}")
    if backup.exists():
        return backup, False
    shutil.copyfile(path, backup)
    return backup, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_clean_labels", description="Tidy the stack labels before scoring."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    return parser


@dataclass
class Session:
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = field(default=print)


def run(labels_path: Path, session: Session) -> int:
    if not labels_path.is_file():
        raise ScriptError(f"{labels_path} does not exist")
    label_file = load_label_file(labels_path)
    backup, created = back_up(labels_path)
    session.out(f"Backup: {backup} ({'written' if created else 'already there, kept as it is'})")

    proposals = build_proposals(label_file)
    issues_before = lint_file(label_file)
    if not any(p.changed or p.flagged for p in proposals):
        session.out(f"Nothing to clean ({len(issues_before)} lint problems).")
        return 0 if not issues_before else 1
    session.out(format_table(proposals))
    unchanged = sum(not (p.changed or p.flagged) for p in proposals)
    session.out(f"\n{unchanged} entries already clean. Lint problems now: {len(issues_before)}.")

    try:
        if not ask_keep_or_drop(proposals, session.ask):
            session.out("Stopped. Nothing was changed.")
            return 1
        dropped = sum(not i.keep for p in proposals for i in p.items)
        answer = session.ask(f"Write these changes ({dropped} flagged entries dropped)? [y/N] ")
    except EOFError:
        session.out("\nInput ended. Nothing was changed.")
        return 1
    if answer.strip().lower() not in ("y", "yes"):
        session.out("Nothing was changed.")
        return 1

    save_label_file(labels_path, apply_proposals(label_file, proposals))
    issues_after = lint_file(load_label_file(labels_path))
    for issue in issues_after:
        why = ", ".join(issue.problems)
        session.out(f"  still unclean: post {issue.post} {issue.entry!r}: {why}")
    session.out(f"Written. Lint problems after: {len(issues_after)}.")
    return 0 if not issues_after else 1


def main(argv: Sequence[str] | None = None, session: Session | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args.labels, session or Session())
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
