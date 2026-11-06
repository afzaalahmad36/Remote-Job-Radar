"""`eval_clean_labels.py --check-labels`: find problems in hand labels before scoring.

Read-only. For each labelled post it reads the post text from the database (the same database
guard as the other scripts: only `jobradar`, a server-side read-only connection) and checks the
labels against that text. It never reads a stored or recomputed extraction, makes no provider
request and writes nothing (no backup, no report file).

It prints post numbers and rule codes only, never label text or post text. A post number is the
position among the labelled posts, the same number the cleaner's table uses.

Rules (the code is what is printed):

  title_not_in_text                  the title label is not a substring of the post text
  company_too_long / company_comma   the company label is over 40 characters / holds a comma
  company_not_in_text                the company label is not in the post text
  stack_not_in_text (xN)             N stack entries are not in the post text
  salary_min_not_in_text / salary_max_not_in_text
                                     the amount is not in the text as a number (commas, dots,
                                     spaces as thousands separators, and "k" for thousands)
  period_no_word_near_amount         a salary period is set but no period word is within 60
                                     characters of an amount
  seniority_no_level_word_in_title   a seniority is set but the title label has no level word
  identical_labels (with post N)     two posts have the same labels

Two housekeeping codes stop a check from judging labels against the wrong text:
`job_not_found` and `stale_text` (the text changed since it was labelled). Such a post gets no
other checks.

Text is compared case-folded, NFKC-normalised, with whitespace collapsed.
"""

import asyncio
import re
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.extraction.stack import canonical_name
from app.extraction.text_hash import raw_text_hash
from scripts.db_guard import database_target
from scripts.eval_label import read_posts_by_id
from scripts.eval_labelfile import LabelFile, Labels, load_label_file

MAX_COMPANY_CHARS = 40
PERIOD_WINDOW = 60
LEVEL_WORDS = (
    "junior",
    "mid",
    "senior",
    "lead",
    "principal",
    "staff",
    "director",
    "manager",
    "intern",
)

# Other spellings of a name, both ways ("k8s" for Kubernetes, "Google Cloud Platform" for GCP).
# Short on purpose: a stack entry is also tried under its dictionary spelling.
STACK_ALIASES: dict[str, tuple[str, ...]] = {
    "kubernetes": ("k8s",),
    "postgresql": ("postgres",),
    "javascript": ("js",),
    "typescript": ("ts",),
    "node.js": ("nodejs", "node"),
    "go": ("golang",),
    "aws": ("amazon web services",),
    "gcp": ("google cloud platform", "google cloud"),
    "mongodb": ("mongo",),
    "elasticsearch": ("elastic",),
    "c#": ("csharp",),
    ".net": ("dotnet",),
    "react": ("react.js", "reactjs"),
    "kafka": ("apache kafka",),
    "parquet": ("apache parquet",),
    "microsoft office": ("microsoft office suite", "ms office"),
}

_NUMBER_GROUPED = re.compile(r"(?<![\w.,])\d{1,3}(?:[,.   ]\d{3})+(?![\d])")
_NUMBER_PLAIN = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)?)(\s?k\b)?", re.IGNORECASE)
_PERIOD_WORD = re.compile(r"year|annual|month|hour|\bweek\b|\bday\b|\bper\b|/hr|/yr|/mo|/wk")
_LEVEL = re.compile(r"\b(" + "|".join(LEVEL_WORDS) + r")\b")


def fold(text: str) -> str:
    """NFKC, case-folded, single spaces: the form labels and post text are compared in."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Problem:
    code: str
    count: int = 1  # how many entries (stack) hit the rule
    detail: str = ""

    def text(self) -> str:
        return (
            self.code
            + (f" x{self.count}" if self.count > 1 else "")
            + (f" ({self.detail})" if self.detail else "")
        )


# --- numbers ------------------------------------------------------------------------------------


def amount_spans(folded: str) -> list[tuple[int, int, int]]:
    """(value, start, end) of every number in the text; "90k" is 90000, "90,000" is 90000."""
    spans = []
    for match in _NUMBER_GROUPED.finditer(folded):
        value = int(re.sub(r"\D", "", match.group()))
        spans.append((value, match.start(), match.end()))
    for match in _NUMBER_PLAIN.finditer(folded):
        number = float(match.group(1))
        value = round(number * 1000) if match.group(2) else int(number)
        spans.append((value, match.start(), match.end()))
    return spans


# --- the per-post checks ------------------------------------------------------------------------


def stack_variants(entry: str) -> set[str]:
    key = fold(entry)
    variants = {key}
    canonical = canonical_name(entry)
    if canonical:
        variants.add(fold(canonical))
    for name in list(variants):
        variants.update(STACK_ALIASES.get(name, ()))
        variants.update(name for name, others in STACK_ALIASES.items() if key in others)
    return variants


def appears(variant: str, folded: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", folded) is not None


def _company_problems(company: str, folded: str) -> list[Problem]:
    problems = []
    if len(company) > MAX_COMPANY_CHARS:
        problems.append(Problem("company_too_long"))
    if "," in company:
        problems.append(Problem("company_comma"))
    if fold(company) not in folded:
        problems.append(Problem("company_not_in_text"))
    return problems


def _salary_problems(labels: Labels, folded: str) -> list[Problem]:
    problems = []
    spans = amount_spans(folded)
    present = {value for value, _, _ in spans}
    for code, amount in (("salary_min", labels.salary_min), ("salary_max", labels.salary_max)):
        if amount is not None and amount not in present:
            problems.append(Problem(f"{code}_not_in_text"))
    if labels.salary_period is not None:
        amounts = {a for a in (labels.salary_min, labels.salary_max) if a is not None}
        windows = [
            folded[max(0, start - PERIOD_WINDOW) : end + PERIOD_WINDOW]
            for value, start, end in spans
            if value in amounts
        ]
        if not any(_PERIOD_WORD.search(window) for window in windows):
            problems.append(Problem("period_no_word_near_amount"))
    return problems


def check_post(labels: Labels, text: str) -> list[Problem]:
    """Problems with one post's labels against its text. Never looks at an extraction."""
    folded = fold(text)
    problems = []
    if labels.title and fold(labels.title) not in folded:
        problems.append(Problem("title_not_in_text"))
    if labels.company:
        problems.extend(_company_problems(labels.company, folded))
    missing = [e for e in labels.stack if not any(appears(v, folded) for v in stack_variants(e))]
    if missing:
        problems.append(Problem("stack_not_in_text", len(missing)))
    problems.extend(_salary_problems(labels, folded))
    if labels.seniority is not None and not _LEVEL.search(fold(labels.title or "")):
        problems.append(Problem("seniority_no_level_word_in_title"))
    return problems


def identity(labels: Labels) -> tuple[object, ...]:
    return (
        fold(labels.title or ""),
        fold(labels.company or ""),
        tuple(sorted(fold(entry) for entry in labels.stack)),
        labels.remote_policy,
        labels.seniority,
        labels.salary_min,
        labels.salary_max,
        labels.salary_currency,
        labels.salary_period,
    )


# --- the whole file -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    checked: int  # labelled posts
    problems: dict[int, list[Problem]]  # post number -> problems, only posts with some


def check_labels(label_file: LabelFile, texts: Mapping[uuid.UUID, str]) -> Report:
    labelled = [(e, e.labels) for e in label_file.posts if e.labels is not None]
    problems: dict[int, list[Problem]] = defaultdict(list)
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for number, (entry, labels) in enumerate(labelled, start=1):
        groups[identity(labels)].append(number)
        text = texts.get(entry.job_id)
        if text is None:
            problems[number].append(Problem("job_not_found"))
        elif raw_text_hash(text) != entry.raw_text_hash:
            problems[number].append(Problem("stale_text"))
        else:
            problems[number].extend(check_post(labels, text))
    for numbers in groups.values():
        for number in numbers if len(numbers) > 1 else []:
            others = ", ".join(str(n) for n in numbers if n != number)
            problems[number].append(Problem("identical_labels", detail=f"with post {others}"))
    with_problems = {n: found for n, found in sorted(problems.items()) if found}
    return Report(len(labelled), with_problems)


def format_report(report: Report) -> list[str]:
    """Post numbers and rule codes only: no label text, no post text."""
    lines = [
        f"post {number}: " + ", ".join(problem.text() for problem in found)
        for number, found in report.problems.items()
    ]
    totals: dict[str, int] = defaultdict(int)
    for found in report.problems.values():
        for problem in found:
            totals[problem.code] += 1
    lines.append(f"Checked {report.checked} labelled posts; {len(report.problems)} have problems.")
    lines.extend(f"  {code}: {count} post(s)" for code, count in sorted(totals.items()))
    return lines


def run_check(labels_path: Path, settings: Settings, out: Callable[[str], None] = print) -> int:
    """Check the labels in `labels_path`. Reads the database read-only; writes nothing."""
    label_file = load_label_file(labels_path)
    target = database_target(settings.database_url)  # refuses the wrong database up front
    out(f"Reading post text (read-only) from database {target}")
    ids = [entry.job_id for entry in label_file.posts if entry.labels is not None]
    texts = dict(asyncio.run(read_posts_by_id(settings.database_url, ids)))
    report = check_labels(label_file, texts)
    for line in format_report(report):
        out(line)
    return 0 if not report.problems else 1
