"""Scoring extraction accuracy against hand labels. Pure: no database, no network, no clock.

The pass rule, the 90% target and the invented-values gate are written in CLAUDE.md (section 6,
"Extraction accuracy evaluation"). This module is that text as code, and nothing else: change the
rule there first, then here.

One post gives a `FieldValues` from the label and one from the system (the stored extraction, or the
rules recomputed offline). `score_post` compares them field by field; `summarise` adds the posts up.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.extraction.stack import canonical_name

TARGET_PERCENT = 90  # share of posts that must parse correctly
INVENTED_GATE_PERCENT = 10  # more posts than this with any invented value is a FAIL
STACK_RECALL_PERCENT = 70  # share of the label's stack that a post must find
SCALAR_FIELDS = ("title", "company", "remote_policy", "seniority", "salary")
SALARY_PARTS = ("min", "max", "currency", "period")


class Outcome(StrEnum):
    """What happened to one field of one post."""

    CORRECT = "correct"  # the same value
    ABSENT = "absent"  # the label has none and the system found none: correct too
    WRONG = "wrong"  # both have a value and they differ
    MISSING = "missing"  # the label has a value and the system found none
    INVENTED = "invented"  # the label has none and the system gave a value


@dataclass(frozen=True)
class FieldValues:
    """The values of one post, from a label or from the system. None / empty means "none"."""

    title: str | None = None
    company: str | None = None
    stack: tuple[str, ...] = ()
    remote_policy: str | None = None
    seniority: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    salary_period: str | None = None


# --- normalisation -----------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
# "(m/f/d)", "(f/m/x)", "[w/m/d]": gender markers some European posts append to a title.
_GENDER_MARKER = re.compile(r"[(\[]\s*[mfwdx](?:\s*[/|*]\s*[mfwdx])+\s*[)\]]", re.IGNORECASE)
_TITLE_SEPARATORS = re.compile(r"[-–—/|,]")
_LEGAL_SUFFIXES = frozenset({"inc", "llc", "ltd", "gmbh", "ag", "bv", "sa", "plc", "corp", "co"})
_ENUM_ALIASES = {
    "on-site": "onsite",
    "on_site": "onsite",
    "on site": "onsite",
    "in-office": "onsite",
}


def _fold(text: str) -> str:
    """NFKC, casefolded, single-spaced."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _strip_edges(text: str) -> str:
    """Drop leading and trailing characters that are not letters or digits (a trailing + or # stays,
    because "C++" and "C#" end with them)."""
    start, end = 0, len(text)
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not (text[end - 1].isalnum() or text[end - 1] in "+#"):
        end -= 1
    return text[start:end]


def normalise_title(value: str | None) -> str | None:
    if value is None:
        return None
    text = _GENDER_MARKER.sub(" ", _fold(value))
    text = _TITLE_SEPARATORS.sub(" ", text)
    return _strip_edges(_WHITESPACE.sub(" ", text).strip()) or None


def normalise_company(value: str | None) -> str | None:
    if value is None:
        return None
    words = _strip_edges(_fold(value)).split(" ")
    while len(words) > 1 and words[-1].strip(".,") in _LEGAL_SUFFIXES:
        words.pop()
    return _strip_edges(" ".join(words)) or None


def normalise_enum(value: str | None) -> str | None:
    if value is None:
        return None
    text = _fold(value)
    return _ENUM_ALIASES.get(text, text) or None


def normalise_currency(value: str | None) -> str | None:
    return (value.strip().upper() or None) if value is not None else None


def stack_key(entry: str) -> str:
    """The dictionary's spelling when it knows the technology, else the folded text."""
    return (canonical_name(entry) or _fold(entry)).casefold()


def stack_keys(entries: Iterable[str]) -> dict[str, str]:
    """key -> first entry that produced it, without duplicates and without empty entries."""
    keys: dict[str, str] = {}
    for entry in entries:
        key = stack_key(entry)
        if key:
            keys.setdefault(key, entry)
    return keys


# --- one field ---------------------------------------------------------------------------------


def compare(label: object | None, system: object | None) -> Outcome:
    """The five-way outcome for two already normalised values (None = no value)."""
    if label is None:
        return Outcome.ABSENT if system is None else Outcome.INVENTED
    if system is None:
        return Outcome.MISSING
    return Outcome.CORRECT if label == system else Outcome.WRONG


def combine(outcomes: Iterable[Outcome]) -> Outcome:
    """One outcome for a group (the salary parts): the worst one wins."""
    seen = set(outcomes)
    for worst in (Outcome.WRONG, Outcome.INVENTED, Outcome.MISSING):
        if worst in seen:
            return worst
    return Outcome.CORRECT if Outcome.CORRECT in seen else Outcome.ABSENT


def salary_parts(label: FieldValues, system: FieldValues) -> dict[str, Outcome]:
    """Outcome of min, max, currency and period, each on its own."""
    return {
        "min": compare(label.salary_min, system.salary_min),
        "max": compare(label.salary_max, system.salary_max),
        "currency": compare(
            normalise_currency(label.salary_currency), normalise_currency(system.salary_currency)
        ),
        "period": compare(
            normalise_enum(label.salary_period), normalise_enum(system.salary_period)
        ),
    }


@dataclass(frozen=True)
class StackScore:
    """The stack of one post. Counts only: no text is kept."""

    true_positives: int
    false_positives: int
    false_negatives: int
    label_size: int
    dictionary_gaps: int  # label entries the technology dictionary does not know

    @property
    def ok(self) -> bool:
        """No entry outside the label, and at least 70% of the label found."""
        found_enough = self.true_positives * 100 >= self.label_size * STACK_RECALL_PERCENT
        return self.false_positives == 0 and found_enough


def score_stack(label: Iterable[str], system: Iterable[str]) -> StackScore:
    wanted, given = stack_keys(label), stack_keys(system)
    return StackScore(
        true_positives=len(wanted.keys() & given.keys()),
        false_positives=len(given.keys() - wanted.keys()),
        false_negatives=len(wanted.keys() - given.keys()),
        label_size=len(wanted),
        dictionary_gaps=sum(1 for entry in wanted.values() if canonical_name(entry) is None),
    )


# --- one post ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PostScore:
    outcomes: Mapping[str, Outcome]  # one per scalar field; salary is the worst of its parts
    stack: StackScore
    invented: bool  # any invented company, salary part, seniority or remote policy
    failures: tuple[str, ...]  # reason codes; empty means the post parses correctly

    @property
    def passed(self) -> bool:
        return not self.failures


def score_post(label: FieldValues, system: FieldValues) -> PostScore:
    """Compare one post. The label must have a title (a post without one is excluded earlier)."""
    title = normalise_title(label.title)
    if title is None:
        raise ValueError("a post with a blank title label cannot be scored")
    company = normalise_company(label.company)
    parts = salary_parts(label, system)
    outcomes = {
        "title": compare(title, normalise_title(system.title)),
        "company": compare(company, normalise_company(system.company)),
        "remote_policy": compare(
            normalise_enum(label.remote_policy), normalise_enum(system.remote_policy)
        ),
        "seniority": compare(normalise_enum(label.seniority), normalise_enum(system.seniority)),
        "salary": combine(parts.values()),
    }
    stack = score_stack(label.stack, system.stack)
    invented = Outcome.INVENTED in (
        outcomes["company"],
        outcomes["remote_policy"],
        outcomes["seniority"],
        *parts.values(),
    )
    return PostScore(outcomes, stack, invented, _failures(outcomes, company, stack, parts))


def _failures(
    outcomes: Mapping[str, Outcome],
    company: str | None,
    stack: StackScore,
    parts: Mapping[str, Outcome],
) -> tuple[str, ...]:
    failures: list[str] = []
    if outcomes["title"] is not Outcome.CORRECT:
        failures.append(f"title_{outcomes['title']}")
    if company is not None and outcomes["company"] is not Outcome.CORRECT:
        failures.append(f"company_{outcomes['company']}")  # a blank label accepts any company
    if stack.false_positives:
        failures.append("stack_extra_entries")
    if stack.true_positives * 100 < stack.label_size * STACK_RECALL_PERCENT:
        failures.append("stack_low_recall")
    for name in ("remote_policy", "seniority"):
        if outcomes[name] in (Outcome.WRONG, Outcome.INVENTED):
            failures.append(f"{name}_{outcomes[name]}")
    for kind in (Outcome.WRONG, Outcome.INVENTED):
        if kind in parts.values():
            failures.append(f"salary_{kind}")
            break
    return tuple(failures)


# --- many posts --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSummary:
    counts: Mapping[Outcome, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accuracy(self) -> float | None:
        """Correct or correctly absent, over all scored posts."""
        right = self.counts.get(Outcome.CORRECT, 0) + self.counts.get(Outcome.ABSENT, 0)
        return right / self.total if self.total else None

    @property
    def recall_on_labelled(self) -> float | None:
        """How often a non-blank label value was found correctly."""
        labelled = sum(
            self.counts.get(o, 0) for o in (Outcome.CORRECT, Outcome.WRONG, Outcome.MISSING)
        )
        return self.counts.get(Outcome.CORRECT, 0) / labelled if labelled else None


@dataclass(frozen=True)
class StackSummary:
    posts_ok: int
    posts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    dictionary_gaps: int

    @property
    def precision(self) -> float | None:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else None

    @property
    def recall(self) -> float | None:
        wanted = self.true_positives + self.false_negatives
        return self.true_positives / wanted if wanted else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p is not None and r is not None and p + r else None


@dataclass(frozen=True)
class Summary:
    scored: int
    passed: int
    invented_posts: int
    fields: Mapping[str, FieldSummary]
    stack: StackSummary
    failure_counts: Mapping[str, int]

    @property
    def pass_share(self) -> float | None:
        return self.passed / self.scored if self.scored else None

    @property
    def invented_share(self) -> float | None:
        return self.invented_posts / self.scored if self.scored else None

    @property
    def target_met(self) -> bool:
        return self.scored > 0 and self.passed * 100 >= self.scored * TARGET_PERCENT

    @property
    def gate_failed(self) -> bool:
        """More than 10% of the posts contain an invented value."""
        return self.invented_posts * 100 > self.scored * INVENTED_GATE_PERCENT

    @property
    def verdict(self) -> str:
        if self.scored == 0:
            return "NO DATA"
        return "PASS" if self.target_met and not self.gate_failed else "FAIL"

    @property
    def fail_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.scored and not self.target_met:
            reasons.append(f"fewer than {TARGET_PERCENT}% of the posts parse correctly")
        if self.gate_failed:
            reasons.append(
                f"more than {INVENTED_GATE_PERCENT}% of the posts contain an invented value"
            )
        return tuple(reasons)

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.passed, self.scored)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval (95% by default) for a share. None when there are no posts."""
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarise(scores: Iterable[PostScore]) -> Summary:
    posts = list(scores)
    fields = {
        name: FieldSummary(Counter(post.outcomes[name] for post in posts)) for name in SCALAR_FIELDS
    }
    stacks = [post.stack for post in posts]
    return Summary(
        scored=len(posts),
        passed=sum(post.passed for post in posts),
        invented_posts=sum(post.invented for post in posts),
        fields=fields,
        stack=StackSummary(
            posts_ok=sum(stack.ok for stack in stacks),
            posts=len(stacks),
            true_positives=sum(stack.true_positives for stack in stacks),
            false_positives=sum(stack.false_positives for stack in stacks),
            false_negatives=sum(stack.false_negatives for stack in stacks),
            dictionary_gaps=sum(stack.dictionary_gaps for stack in stacks),
        ),
        failure_counts=Counter(code for post in posts for code in post.failures),
    )


# --- text output -------------------------------------------------------------------------------


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def headline(name: str, summary: Summary) -> str:
    """One block: the verdict, the parse-correct share with its interval, the invented gate and the
    stack dictionary gaps, side by side so none of them is read alone."""
    if summary.scored == 0:
        return f"{name}: NO DATA (no post could be scored)"
    low_high = summary.interval
    interval = (
        f", 95% interval {percent(low_high[0])} to {percent(low_high[1])}" if low_high else ""
    )
    lines = [
        f"{name}: {summary.verdict} (on this sample)",
        f"  parse correctly : {summary.passed}/{summary.scored} = "
        f"{percent(summary.pass_share)}{interval}  (target {TARGET_PERCENT}%)",
        f"  invented gate   : {summary.invented_posts}/{summary.scored} posts with an invented "
        f"value = {percent(summary.invented_share)}  (FAIL above {INVENTED_GATE_PERCENT}%)",
        f"  stack dictionary: {summary.stack.dictionary_gaps} label entries unknown to the "
        "dictionary (lost recall that is not a rule miss)",
    ]
    lines.extend(f"  FAIL because    : {reason}" for reason in summary.fail_reasons)
    return "\n".join(lines)


def field_table(summary: Summary) -> str:
    """Per-field accuracy, recall on labelled values, and the wrong / missing / invented counts."""
    rows = [
        f"  {'field':<15}{'accuracy':>9}{'recall*':>9}{'wrong':>7}{'missing':>9}{'invented':>10}"
    ]
    for name in SCALAR_FIELDS:
        field = summary.fields[name]
        count = field.counts.get
        rows.append(
            f"  {name:<15}{percent(field.accuracy):>9}{percent(field.recall_on_labelled):>9}"
            f"{count(Outcome.WRONG, 0):>7}{count(Outcome.MISSING, 0):>9}"
            f"{count(Outcome.INVENTED, 0):>10}"
        )
    stack = summary.stack
    rows.append(
        f"  {'stack':<15}{percent(stack.posts_ok / stack.posts if stack.posts else None):>9}"
        f"{percent(stack.recall):>9}{'':>7}{stack.false_negatives:>9}{stack.false_positives:>10}"
    )
    rows.append("  * recall = how often a non-blank label value was found correctly.")
    rows.append(
        f"  stack entries: precision {percent(stack.precision)}, recall {percent(stack.recall)}, "
        f"F1 {percent(stack.f1)} (micro-averaged); for stack, missing/invented count entries"
    )
    return "\n".join(rows)
