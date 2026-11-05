"""One-off helper: tidy the stack labels in `.local/eval/labels.json` before scoring.

Run by hand, never by CI or app code. It reads and writes only the labels file; it does not touch
the database, any stored extraction or any provider. It changes the `stack` list of each labelled
post and nothing else (ids, hashes, titles, companies, salary, remote policy and seniority stay
byte-for-byte as they were).

Order of events: copy `labels.json` to `labels.raw.json` (never overwritten), show every proposed
rewrite as a table (before, after, rule), ask keep/drop for each entry that is not clearly a named
product, then ask for a final yes before writing anything.

    uv run python -m scripts.eval_clean_labels

With `--policy` nothing is asked per entry: every entry that would be asked about gets a proposed
keep or drop from the rule in `eval_clean_policy.py` (lists in `eval_clean_policy.toml`), the reason
is shown in the table, and the one final "Write these changes?" question remains.

    uv run python -m scripts.eval_clean_labels --policy
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
from scripts.eval_clean_policy import DEFAULT_POLICY_PATH, Policy, decide, load_policy
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

# Names of a kind of thing, written in front of the products ("Vector Databases (Milvus, Faiss)").
# The category is proposed for dropping; the products in the brackets are kept as entries.
CATEGORY_NAMES = frozenset(
    {
        "database",
        "databases",
        "vector database",
        "vector databases",
        "cloud platform",
        "cloud platforms",
        "cloud service",
        "cloud services",
        "cloud provider",
        "cloud providers",
        "framework",
        "frameworks",
        "language",
        "languages",
        "programming language",
        "programming languages",
        "library",
        "libraries",
    }
)

# Words written in brackets about the skill level, not about a technology.
ANNOTATION_WORDS = frozenset(
    "primary secondary bonus preferred optional desired required mandatory plus advantage".split()
)
_ANNOTATION_PREFIXES = ("nice to have", "a plus", "transitioning")
_VERSION = re.compile(r"v?\d[\d.x+]*")
_SPACED_SLASH = re.compile(r"\s+/\s+")
_ANY_SLASH = re.compile(r"\s*/\s*")
_SENTENCE_PUNCTUATION = re.compile(r"[.!?;:](\s|$)")
_BRACKETS = "()[]{}"
_EDGE_JUNK = " \t.,;:-"


@dataclass
class Item:
    """One candidate entry. `rules` say what happened to it.

    `drop_reason` is set when the script proposes dropping it (category name, annotation,
    duplicate, policy); `note` is why the policy keeps it; `flag` is why the human is asked;
    `keep` is the proposal or the human's answer.
    """

    name: str
    flag: str | None = None
    keep: bool = True
    rules: list[str] = field(default_factory=list)
    drop_reason: str | None = None
    note: str | None = None

    def rule_text(self) -> str:
        parts = list(self.rules)
        if self.drop_reason:
            parts.append(f"DROP ({self.drop_reason})")
        if self.note:
            parts.append(f"KEEP ({self.note})")
        if self.flag:
            parts.append(f"ASK: {self.flag}")
        return ", ".join(parts) or "kept"


@dataclass
class Proposal:
    """What one original entry (or several, rejoined) becomes: one item per candidate."""

    post: int  # 1-based position among the labelled posts
    before: str
    items: list[Item]

    @property
    def changed(self) -> bool:
        if len(self.items) != 1:
            return True
        item = self.items[0]
        return (
            item.name != self.before
            or bool(item.rules)
            or item.drop_reason is not None
            or item.note is not None
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


def scan_brackets(text: str) -> tuple[str, list[str]]:
    """Split `text` into what is outside any bracket and the content of each outermost group.

    A bracket that never closes keeps its content (everything after the opener); a closer with
    no opener is ignored.
    """
    outside: list[str] = []
    inside: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char in ")]":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    inside.append(text[start:index])
        elif depth == 0:
            outside.append(char)
    if depth > 0:
        inside.append(text[start:])
    return "".join(outside), inside


def _split_bracket_piece(piece: str) -> list[str]:
    """Split on a slash with no spaces too ("NestJS/Express"), unless a side is under 3 letters.

    The size check keeps "CI/CD" and "TCP/IP" whole. It is a heuristic: anything it splits wrongly
    shows up in the table.
    """
    if _SPACED_SLASH.search(piece):
        return _SPACED_SLASH.split(piece)
    sides = _ANY_SLASH.split(piece)
    return sides if len(sides) > 1 and all(len(side) >= 3 for side in sides) else [piece]


def bracket_candidates(content: str) -> list[str]:
    """Separate candidate entries from the content of a bracket: split on commas and slashes."""
    outer, nested = scan_brackets(content)
    pieces = [part for piece in outer.split(",") for part in _split_bracket_piece(piece)]
    for group in nested:
        pieces.extend(bracket_candidates(group))
    return pieces


def annotation_reason(name: str) -> str | None:
    """Why bracket text is a note about the skill, not a technology (None = it may be one)."""
    folded = name.casefold()
    if folded in ANNOTATION_WORDS or folded.startswith(_ANNOTATION_PREFIXES):
        return "annotation"
    if _VERSION.fullmatch(folded):
        return "version"
    if len(name.split()) > MAX_ENTRY_WORDS:
        return "clause"
    return None


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
    if name == name.casefold():  # a lowercase name the dictionary does not know
        return "looks like a description" if len(words) > 1 else "unknown lowercase word"
    return None


def _key(name: str) -> str:
    """Two spellings of one technology ("Go", "Golang") share a key, so only one is kept."""
    return (canonical_name(name) or name).casefold()


def _judge(name: str, bracketed: bool, rules: list[str], seen: set[str]) -> Item:
    item = Item(name, rules=rules)
    if name.casefold() in CATEGORY_NAMES:
        item.keep, item.drop_reason = False, "category name"
    elif bracketed and (reason := annotation_reason(name)):
        item.keep, item.drop_reason = False, reason
    elif _key(name) in seen:
        item.keep, item.drop_reason = False, "duplicate"
    else:
        seen.add(_key(name))
        item.flag = flag_reason(name)
    return item


def propose_for_post(post: int, stack: Sequence[str]) -> list[Proposal]:
    proposals = []
    seen: set[str] = set()
    for group in rejoin_split_brackets(stack):
        before = ", ".join(group)
        lead = ["rejoin-brackets"] if len(group) > 1 else []
        outside, groups = scan_brackets(before)
        top = [tidy(part) for part in split_slash_products(outside)]
        inner = [tidy(p) for content in groups for p in bracket_candidates(content)]
        top_rules = lead + (["brackets-removed"] if groups else [])
        top_rules += ["split-slash"] if len([t for t in top if t]) > 1 else []
        items = [_judge(n, False, list(top_rules), seen) for n in top if n]
        items += [_judge(n, True, [*lead, "bracket-content"], seen) for n in inner if n]
        proposals.append(Proposal(post, before, items))
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
    """One row per candidate entry, with the original text on the first row of its group."""
    rows = [("post", "before", "after", "rule")]
    for p in proposals:
        if not (p.changed or p.flagged):
            continue
        if not p.items:
            rows.append((str(p.post), p.before, "(nothing)", "DROP (empty)"))
        for position, item in enumerate(p.items):
            first = position == 0
            lead = (str(p.post), p.before) if first else ("", "")
            rows.append((*lead, item.name, item.rule_text()))
    widths = [max(len(row[col]) for row in rows) for col in range(4)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(line.rstrip() for line in lines)


def apply_policy(proposals: Sequence[Proposal], policy: Policy) -> tuple[int, int]:
    """Propose keep or drop for entries the policy decides; none of them is asked about after.

    Returns (kept, dropped) counts of the entries it decided. Entries already dropped by a rule
    (category name, annotation, duplicate) are left as they are.
    """
    kept = dropped = 0
    for proposal in proposals:
        for item in proposal.items:
            if not item.keep:
                continue
            decision = decide(item.name, item.flag, policy)
            if decision is None:
                continue
            item.flag = None
            if decision.keep:
                item.note = decision.reason
                kept += 1
            else:
                item.keep, item.drop_reason = False, decision.reason
                dropped += 1
    return kept, dropped


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
    parser.add_argument(
        "--policy",
        action="store_true",
        help="propose keep or drop for every entry that would be asked about, and ask nothing",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help=f"allow, deny and fragment lists for --policy (default: {DEFAULT_POLICY_PATH.name})",
    )
    return parser


@dataclass
class Session:
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = field(default=print)


def run(labels_path: Path, session: Session, policy: Policy | None = None) -> int:
    if not labels_path.is_file():
        raise ScriptError(f"{labels_path} does not exist")
    label_file = load_label_file(labels_path)
    backup, created = back_up(labels_path)
    session.out(f"Backup: {backup} ({'written' if created else 'already there, kept as it is'})")

    proposals = build_proposals(label_file)
    if policy is not None:
        kept, dropped_by_policy = apply_policy(proposals, policy)
        session.out(f"Policy: {kept} entries kept and {dropped_by_policy} dropped by the lists.")
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
        answer = session.ask(f"Write these changes ({dropped} entries dropped)? [y/N] ")
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.policy_file is not None and not args.policy:
        parser.error("--policy-file only makes sense together with --policy")
    try:
        policy = load_policy(args.policy_file or DEFAULT_POLICY_PATH) if args.policy else None
        return run(args.labels, session or Session(), policy)
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
