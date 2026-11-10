"""Second pass over the hand labels: add what the post states and the first pass left out.

Run by hand; CI never runs it. It reads the label file and, for each labelled post, only the
post's id and `raw_text` from the database (read-only, the same database-name guard as the other
eval scripts). It never imports the extractor's rules, the merge or a provider client, reads no
stored extraction, and writes only under `.local/`. The one thing it takes from `app.extraction`
is the technology dictionary (`stack._TECHNOLOGIES`, read directly: it has no public accessor and
the extractor is not changed for this).

Why it exists: the first-pass labels list what the labeller chose to write down, while the label
convention says "list every technology the post names" and "a blank salary means the post does
not say". Under the fixed rules below the labels are completed, and the first-pass labels stay in
`labels.pass1.json`, so both scores can be reported. These rules were written after the first
accuracy run had been diagnosed, and part of them mirrors rules the extractor now has, so a score
on the completed labels is not independent evidence; that is why the first-pass score is kept.

Rules, applied to every labelled post. A component that is already filled is never changed.

  (a) Salary. A candidate is a money amount or range (a currency sign or code next to a number)
      in a sentence that has a pay word (salary, base, compensation, pay, OTE, CTC, wage, earn,
      remuneration) in it or in the line before it, or that holds a value the label already has.
      A sentence about a benefit (401(k), match, stipend, allowance, reimbursement, per diem,
      bonus, equity, stock) without a pay word is not a candidate.
      * Label with no amount and exactly one candidate: min, max and currency are filled
        (a range is min and max, a single amount is min = max, "up to X" is max only, "from X" or
        "X+" is min only). More than one candidate: NEEDS_HUMAN, nothing filled.
      * Label with only a min (or only a max): the other end is filled when exactly one candidate
        is a range whose matching end equals the label.
      * A blank currency is filled from the candidate that matches the label. "$" with an upper
        case CAN or CAD in the sentence is CAD, unless the sentence says USD.
      * The period is not touched.
  (b) Stack. A technology is added when ALL of these hold: it is on the allow-list of
      eval_clean_policy.toml, an alias there, or in the stack dictionary; it is not on the deny or
      fragment lists; it appears as a whole word in the text (any listed spelling, case
      insensitive) in at least one mention that is not in a customer list ("trusted by ...",
      "clients include ...") and not an "X alternative" / "instead of X" phrase; the label does not
      already hold it under another spelling ("Airflow" and "Apache Airflow"); and the post names
      12 or fewer distinct technologies in all (more: nothing is added, NEEDS_HUMAN). The words
      swift, rust, ruby, unreal, dart, flask, rails, celery and jest count only when a mention
      starts with a capital letter; those rows are marked FLAG in the table.
      Revised once, after the first dry-run showed false adds (nothing had been written): a
      mention does not count when
      * a hyphen touches it ("Go-To-Market", "go-to-market", "Python-based");
      * it is the company in the headline, the first line's "Company: Role" (a mention in the body
        still counts);
      * the next word is capital, clinic, catalog, catalogue, ventures or partners;
      * it is capitalised and a neighbouring word is also capitalised ("Maven Health"), unless that
        neighbour is itself a technology ("Python Django") or the pair is a dictionary name
        ("React Native");
      * it is a bare "Go" with no list (a comma or slash and two other technologies in the
        sentence), no "Go (" and no "golang" within 60 characters;
      * its sentence says not, n't, except, excluding or simple.
      A technology whose mentions were all thrown out is listed as INFO with the first reason. A
      post that would gain more than --max-adds-per-post (default 8) entries gets none: NEEDS_HUMAN.

A change table (with the sentence behind every stack entry and its match type: list = commas or
slashes and two other technologies in the sentence, standalone = no other technology in it,
other = the rest) and the NEEDS_HUMAN table go to your
terminal only. Then one question: "Write these changes? [y/N]". Before writing, the labels are
copied to `<name>.pass1.json` (an existing copy is never overwritten); after writing, the same
checks as `eval_clean_labels.py --check-labels` run again and only post numbers and codes print.
"""

import argparse
import asyncio
import re
import shutil
import sys
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from app.extraction import stack as dictionary
from app.extraction.text_hash import raw_text_hash
from scripts.db_guard import ScriptError, database_target
from scripts.eval_check_labels import check_labels, fold, format_report
from scripts.eval_clean_policy import DEFAULT_POLICY_PATH, Policy, load_policy, normal
from scripts.eval_fix_labels import Session
from scripts.eval_label import read_posts_by_id
from scripts.eval_labelfile import (
    DEFAULT_LABELS_PATH,
    LabelEntry,
    LabelFile,
    Labels,
    load_label_file,
    save_label_file,
)
from scripts.eval_scoring import stack_key

BACKUP_SUFFIX = ".pass1"
MAX_TECHNOLOGIES = 12
DEFAULT_MAX_ADDS = 8
GO = "go"
# A word after a technology that makes it part of a company or product name ("Spark Capital").
NAME_FOLLOWERS = frozenset("capital clinic catalog catalogue ventures partners".split())
CAPITALISED_ONLY = frozenset("swift rust ruby unreal dart flask rails celery jest".split())
SHOWN_CHARS = 160
LOOKBACK_CHARS = 120
WINDOW_CHARS = 300

RULE_SALARY = "a salary"
RULE_STACK = "b stack"
NEEDS_HUMAN = "NEEDS_HUMAN"
INFO = "INFO"

# --- salary --------------------------------------------------------------------------------------

_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
_CODES = "USD|EUR|GBP|CAD|AUD|CHF|INR"
_NUMBER = r"\d{1,3}(?:[,.]\d{3})+|\d+(?:\.\d+)?"
_MONEY = re.compile(
    rf"(?:(?P<sym>[$€£₹])|(?P<code>{_CODES})\s?)(?P<a>{_NUMBER})(?P<ak>\s?[kK])?"
    rf"(?:\s?(?:-|–|—|to)\s?(?:[$€£₹]|(?:{_CODES})\s?)?(?P<b>{_NUMBER})(?P<bk>\s?[kK])?)?"
    rf"(?:\s?(?P<suffix>{_CODES}|CAN)\b)?"
)
_PAY_WORDS = re.compile(
    r"salary|\bbase\b|compensation|\bpay\b|\bote\b|\bctc\b|\bwages?\b|\bearn|remuneration",
    re.IGNORECASE,
)
_BENEFIT_WORDS = re.compile(
    r"401\s?\(?k\)?|\bmatch(?:ing)?\b|\ballowance|\bstipend|\breimburs|\bper diem\b|\bbonus"
    r"|\bequity\b|\bstock\b",
    re.IGNORECASE,
)
_UP_TO = re.compile(r"(?:up\s*to|max(?:imum)?)\W*$", re.IGNORECASE)
_FROM = re.compile(r"(?:from|starting(?:\s+at)?|min(?:imum)?|at\s+least)\W*$", re.IGNORECASE)
_CANADIAN = re.compile(r"\b(?:CAD|CAN)\b")
_US_DOLLAR = re.compile(r"\bUSD\b")


@dataclass(frozen=True)
class Candidate:
    low: int | None
    high: int | None
    currency: str | None
    sentence: str


@dataclass(frozen=True)
class Change:
    post: int
    field: str
    before: str
    after: str
    rule: str
    note: str = ""
    match: str = ""  # stack only: list, standalone or other


@dataclass(frozen=True)
class Review:
    post: int
    field: str
    current: str
    shows: str


@dataclass
class PostResult:
    labels: Labels
    changes: list[Change] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)


def _cut(text: str, limit: int = SHOWN_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _sentence(text: str, start: int, end: int) -> tuple[str, str]:
    """The sentence holding text[start:end] (bounded) and the line before it."""
    left = text[max(0, start - WINDOW_CHARS) : start]
    cut = max(left.rfind(". "), left.rfind("! "), left.rfind("? "), left.rfind("\n"))
    right = text[end : end + WINDOW_CHARS]
    stops = [
        i
        for i in (right.find(". "), right.find("! "), right.find("? "), right.find("\n"))
        if i >= 0
    ]
    sentence = left[cut + 1 :] + text[start:end] + right[: min(stops) if stops else len(right)]
    before_line = left[:cut].rsplit("\n", 1)[-1] if cut > 0 else ""
    return sentence, before_line


_GROUPED = re.compile(r"\d{1,3}(?:[,.]\d{3})+")


def _thousands(digits: str, has_k: bool) -> float:
    """ "90,000" and "90.000" are ninety thousand, "12.5" is a decimal, a "k" means thousands."""
    plain = re.sub(r"[,.]", "", digits) if _GROUPED.fullmatch(digits) else digits
    return float(plain) * (1000 if has_k else 1)


def _amounts(match: re.Match[str]) -> tuple[int, int | None]:
    low = _thousands(match["a"], bool(match["ak"]))
    if match["b"] is None:
        return int(low), None
    high = _thousands(match["b"], bool(match["bk"]))
    if match["ak"] or match["bk"]:  # "80-120k" and "80k-120k" both mean thousands
        low = low * 1000 if not match["ak"] and low < 1000 else low
        high = high * 1000 if not match["bk"] and high < 1000 else high
    return int(low), int(high)


def _currency(match: re.Match[str], sentence: str) -> str | None:
    if match["suffix"] == "CAN":
        return "CAD"
    explicit = match["suffix"] or match["code"]
    if explicit:
        return str(explicit)
    base = _SYMBOLS[match["sym"]]
    if base == "USD" and _CANADIAN.search(sentence) and not _US_DOLLAR.search(sentence):
        return "CAD"  # "$140K-$180K CAN": the dollar sign is a Canadian dollar
    return base


def find_candidates(text: str, labels: Labels) -> list[Candidate]:
    """Every distinct salary the post states, under the rules in the module docstring."""
    present = {labels.salary_min, labels.salary_max} - {None}
    found: dict[tuple[int | None, int | None, str | None], Candidate] = {}
    for match in _MONEY.finditer(text):
        low, high = _amounts(match)
        sentence, heading = _sentence(text, match.start(), match.end())
        if _BENEFIT_WORDS.search(sentence) and not _PAY_WORDS.search(sentence):
            continue
        said_pay = _PAY_WORDS.search(sentence) or _PAY_WORDS.search(heading)
        if not said_pay and not ({low, high} & present):
            continue
        if high is not None and low < high:
            shape: tuple[int | None, int | None] = (low, high)
        else:
            before = text[max(0, match.start() - 14) : match.start()]
            if _UP_TO.search(before):
                shape = (None, low)
            elif _FROM.search(before) or text[match.end() : match.end() + 1] == "+":
                shape = (low, None)
            else:
                shape = (low, low)
        currency = _currency(match, sentence)
        candidate = Candidate(shape[0], shape[1], currency, _cut(sentence))
        found.setdefault((candidate.low, candidate.high, currency), candidate)
    return list(found.values())


def _consistent(candidate: Candidate, labels: Labels) -> bool:
    """The candidate repeats the amount(s) the label already has."""
    if labels.salary_min is not None and labels.salary_max is not None:
        return candidate.low == labels.salary_min and candidate.high == labels.salary_max
    if labels.salary_min is not None:
        return candidate.low == labels.salary_min
    return candidate.high == labels.salary_max


def _show_candidate(candidate: Candidate) -> str:
    return f"{candidate.low}-{candidate.high} {candidate.currency}: {candidate.sentence}"


def complete_salary(number: int, labels: Labels, text: str, result: PostResult) -> Labels:
    """Fill the missing salary amounts and currency the post states (rule a)."""
    has_min, has_max = labels.salary_min is not None, labels.salary_max is not None
    if has_min and has_max and labels.salary_currency is not None:
        return labels
    candidates = find_candidates(text, labels)
    if has_min or has_max:
        candidates = [c for c in candidates if _consistent(c, labels)]
    if len(candidates) > 1:
        shows = " // ".join(_show_candidate(c) for c in candidates)
        current = f"{labels.salary_min}-{labels.salary_max} {labels.salary_currency}"
        shows = f"{len(candidates)} candidates: {shows}"
        result.reviews.append(Review(number, "salary", current, shows))
        return labels
    return _fill(number, labels, candidates[0], result) if candidates else labels


def _fill(number: int, labels: Labels, candidate: Candidate, result: PostResult) -> Labels:
    updates: dict[str, int | str | None] = {}
    if labels.salary_min is None and candidate.low is not None:
        if labels.salary_max is None or candidate.low != candidate.high:
            updates["salary_min"] = candidate.low
    if labels.salary_max is None and candidate.high is not None:
        if labels.salary_min is None or candidate.low != candidate.high:
            updates["salary_max"] = candidate.high
    if labels.salary_currency is None and candidate.currency is not None:
        updates["salary_currency"] = candidate.currency
    for name, value in updates.items():
        before = getattr(labels, name)
        result.changes.append(
            Change(
                number,
                name,
                str(before) if before is not None else "(none)",
                str(value),
                RULE_SALARY,
                candidate.sentence,
            )
        )
    return labels.model_copy(update=updates) if updates else labels


# --- stack ---------------------------------------------------------------------------------------

_REFERENCE_BEFORE = re.compile(
    r"\b(?:trusted|used|loved|chosen) by\b"
    r"|\b(?:customers|clients|partners|investors)\s+(?:include|including|such as|like)\b"
    r"|\b(?:alternatives?|competitors?)\s+(?:to|of)\s+$"
    r"|\binstead of\s+$",
    re.IGNORECASE,
)
_REFERENCE_AFTER = re.compile(r"\W{0,3}(?:alternatives?|competitors?|clones?)\b", re.IGNORECASE)


def _is_reference(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - LOOKBACK_CHARS) : start]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
    return bool(
        _REFERENCE_BEFORE.search(window[cut + 1 :]) or _REFERENCE_AFTER.match(text, end, end + 20)
    )


@dataclass(frozen=True)
class Technology:
    name: str  # what is written into the label
    spellings: tuple[str, ...]  # every way a post may write it


def load_allow_names(path: Path) -> list[str]:
    """The allow-list as written in the policy file (`Policy` keeps only lower-case forms)."""
    with path.open("rb") as handle:
        names = tomllib.load(handle).get("allow", [])
    return [str(name) for name in names]


def known_technologies(policy: Policy, allow_names: Sequence[str]) -> list[Technology]:
    """The allow-list, the policy aliases and the stack dictionary, as (name, spellings)."""
    found: dict[str, Technology] = {}
    for name, spells in dictionary._TECHNOLOGIES.items():  # read-only; see the module docstring
        found.setdefault(normal(name), Technology(name, (name, *spells)))
    for name in allow_names:
        found.setdefault(normal(name), Technology(name, (name,)))
    for alias, targets in policy.aliases.items():
        if len(targets) == 1:
            target = found.get(normal(targets[0]))
            base = target.spellings if target else (targets[0],)
            found[normal(targets[0])] = Technology(
                target.name if target else targets[0], (*base, alias)
            )
    for name in policy.deny | policy.fragments:
        found.pop(name, None)
    return list(found.values())


@dataclass(frozen=True)
class Vocabulary:
    """Every technology spelling as one pattern, to count the technologies named in a sentence."""

    pattern: re.Pattern[str]
    owner: Mapping[str, str]  # folded spelling -> technology name
    spellings: frozenset[str]  # every folded spelling, to tell a technology word from a name word

    def others(self, sentence: str, own: str) -> set[str]:
        """Other technologies in `sentence` (a bare "go" and lower-case swift do not count)."""
        found: set[str] = set()
        for match in self.pattern.finditer(sentence):
            word = fold(match.group())
            if word == GO or (word in CAPITALISED_ONLY and not match.group()[0].isupper()):
                continue
            found.add(self.owner[word])
        found.discard(own)
        return found


@lru_cache(maxsize=4)
def build_vocabulary(technologies: tuple[Technology, ...]) -> Vocabulary:
    """Built once per technology list (`lru_cache` remembers the result for the same argument)."""
    owner = {fold(s): t.name for t in technologies for s in t.spellings}
    longest_first = sorted(owner, key=len, reverse=True)
    alternatives = "|".join(re.escape(spelling) for spelling in longest_first)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9.])(?:{alternatives})(?![A-Za-z0-9])(?!\.[A-Za-z0-9])", re.IGNORECASE
    )
    return Vocabulary(pattern, owner, frozenset(owner))


@dataclass(frozen=True)
class Mention:
    technology: Technology
    position: int
    sentence: str
    ambiguous_capitalised: bool
    match_type: str


@dataclass
class Found:
    """What `_mentions` saw: the mentions that count, and why the others were thrown out."""

    mentions: list[Mention] = field(default_factory=list)
    only_lower: bool = False  # an ambiguous word seen only in lower case
    skipped: list[str] = field(default_factory=list)


_HYPHEN_BEFORE = re.compile(r"[A-Za-z0-9]-$")
_HYPHEN_AFTER = re.compile(r"-[A-Za-z0-9]")
_NEXT_WORD = re.compile(r" ([A-Za-z]+)")
_TITLE_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-Z][a-z0-9]+) $")
_TITLE_AFTER = re.compile(r" ([A-Z][a-z0-9]+)(?![A-Za-z0-9])")
_GO_PAREN = re.compile(r" ?\(")
_NEGATION = re.compile(r"\b(?:not|except|excluding|simple)\b|n['’]t\b", re.IGNORECASE)


def _hyphenated(text: str, start: int, end: int) -> bool:
    """A hyphen touches the word: "Go-To-Market", "go-to-market", "Python-based"."""
    return bool(
        _HYPHEN_BEFORE.search(text[max(0, start - 2) : start]) or _HYPHEN_AFTER.match(text, end)
    )


def _before_a_name_word(text: str, end: int) -> bool:
    """The next word is one that makes a company or product name ("Spark Capital")."""
    following = _NEXT_WORD.match(text, end, end + 25)
    return bool(following and following.group(1).lower() in NAME_FOLLOWERS)


def _in_capitalised_name(text: str, start: int, end: int, vocab: Vocabulary) -> bool:
    """A capitalised word next to another capitalised word ("Maven Clinic", "Spark Capital").

    It is a name only when the neighbour is not a technology itself ("Python Django" is a list)
    and the two words together are not a dictionary or allow-list name ("React Native").
    """
    shown = text[start:end]
    if not shown[0].isupper():
        return False
    before = _TITLE_BEFORE.search(text[max(0, start - 30) : start])
    after = _TITLE_AFTER.match(text, end, end + 30)
    pairs = []
    if before:
        pairs.append((before.group(1), f"{before.group(1)} {shown}"))
    if after:
        pairs.append((after.group(1), f"{shown} {after.group(1)}"))
    return any(
        fold(word) not in vocab.spellings and fold(phrase) not in vocab.spellings
        for word, phrase in pairs
    )


def _is_company_in_headline(text: str, start: int, end: int) -> bool:
    """The mention is the company in the headline's "Company: Role" (the first line)."""
    offset = len(text) - len(text.lstrip())
    first_line = text[offset:].split("\n", 1)[0]
    colon = first_line.find(":")
    if colon < 0 or start >= offset + colon:
        return False
    return fold(first_line[:colon]) == fold(text[start:end])


def _has_list_shape(sentence: str, technology: Technology, vocab: Vocabulary) -> bool:
    """Commas or slashes in the sentence and at least two other technologies named in it."""
    separated = "," in sentence or "/" in sentence
    return separated and len(vocab.others(sentence, technology.name)) >= 2


def _go_without_context(
    text: str, match: re.Match[str], sentence: str, technology: Technology, vocab: Vocabulary
) -> bool:
    """A bare "Go" is the language only in a list, as "Go (", or next to "golang"."""
    if fold(match.group()) != GO:
        return False
    nearby = text[max(0, match.start() - 60) : match.end() + 60].lower()
    in_parens = bool(_GO_PAREN.match(text, match.end(), match.end() + 2))
    return not (in_parens or "golang" in nearby or _has_list_shape(sentence, technology, vocab))


def _match_type(sentence: str, technology: Technology, vocab: Vocabulary) -> str:
    """list = commas or slashes and two other technologies; standalone = no other technology."""
    if _has_list_shape(sentence, technology, vocab):
        return "list"
    return "other" if vocab.others(sentence, technology.name) else "standalone"


def _skip_reason(
    text: str, match: re.Match[str], sentence: str, technology: Technology, vocab: Vocabulary
) -> str | None:
    """Why this mention does not count, or None when it does."""
    start, end = match.span()
    if _hyphenated(text, start, end):
        return "inside a hyphenated phrase"
    if _is_company_in_headline(text, start, end):
        return "the company named in the headline"
    if _before_a_name_word(text, end):
        return "part of a proper name"
    if _in_capitalised_name(text, start, end, vocab):
        return "part of a multi-word capitalised name"
    if _go_without_context(text, match, sentence, technology, vocab):
        return "'Go' with no list, 'Go (' or golang near it"
    if _NEGATION.search(sentence):
        return "the sentence says not, except, excluding or simple"
    return None


def _mentions(text: str, technology: Technology, vocab: Vocabulary) -> Found:
    """Mentions that count. A mention is a whole word (not inside "Node.js" or "react.js"), not in
    a customer list, not an alternative phrase, and not thrown out by `_skip_reason`. Ambiguous
    words need a capital letter."""
    found = Found()
    lower_only = False
    for spelling in dict.fromkeys(technology.spellings):
        ambiguous = normal(spelling) in CAPITALISED_ONLY
        pattern = rf"(?<![A-Za-z0-9.]){re.escape(spelling)}(?![A-Za-z0-9])(?!\.[A-Za-z0-9])"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _is_reference(text, match.start(), match.end()):
                continue
            if ambiguous and not match.group()[0].isupper():
                lower_only = True
                continue
            sentence, _ = _sentence(text, match.start(), match.end())
            reason = _skip_reason(text, match, sentence, technology, vocab)
            if reason:
                found.skipped.append(reason)
                continue
            kind = _match_type(sentence, technology, vocab)
            found.mentions.append(
                Mention(technology, match.start(), _cut(sentence), ambiguous, kind)
            )
    found.mentions.sort(key=lambda m: m.position)
    found.only_lower = lower_only and not found.mentions
    return found


def _whole(word: str, folded: str) -> bool:
    """`word` is a whole word of `folded` ("js" is not part of "node.js")."""
    pattern = rf"(?<![a-z0-9.]){re.escape(word)}(?![a-z0-9])(?!\.[a-z0-9])"
    return re.search(pattern, folded) is not None


def _already_in_label(technology: Technology, entries: Sequence[str]) -> bool:
    """The label holds it, or an entry that contains it or is contained in it ("Apache Airflow")."""
    wanted = {fold(s) for s in technology.spellings} | {stack_key(technology.name)}
    for entry in entries:
        folded = fold(entry)
        if stack_key(entry) in wanted or folded in wanted:
            return True
        if any(_whole(name, folded) or _whole(folded, name) for name in wanted):
            return True
    return False


def _named_technologies(
    number: int,
    text: str,
    technologies: Sequence[Technology],
    result: PostResult,
) -> list[tuple[Technology, list[Mention]]]:
    """The technologies the post names (by a mention that counts); notes the ones thrown out."""
    vocab = build_vocabulary(tuple(technologies))
    named: list[tuple[Technology, list[Mention]]] = []
    for technology in technologies:
        found = _mentions(text, technology, vocab)
        if found.mentions:
            named.append((technology, found.mentions))
        elif found.only_lower:
            result.infos.append(
                f"post {number}: {technology.name!r} appears only in lower case: not added"
            )
        elif found.skipped:
            result.infos.append(f"post {number}: {technology.name!r} not added: {found.skipped[0]}")
    return named


def complete_stack(
    number: int,
    labels: Labels,
    text: str,
    technologies: Sequence[Technology],
    result: PostResult,
    max_adds: int = DEFAULT_MAX_ADDS,
) -> Labels:
    """Add the technologies the post names that the label left out (rule b)."""
    named = _named_technologies(number, text, technologies, result)
    if len(named) > MAX_TECHNOLOGIES:
        shows = f"the post names {len(named)} technologies (over {MAX_TECHNOLOGIES}): none added"
        result.reviews.append(Review(number, "stack", f"{len(labels.stack)} entries", shows))
        return labels
    added: list[Change] = []
    for technology, mentions in sorted(named, key=lambda item: item[1][0].position):
        if _already_in_label(technology, [*labels.stack, *(c.after[2:] for c in added)]):
            continue
        first = mentions[0]
        flag = "FLAG capitalised ambiguous word; " if first.ambiguous_capitalised else ""
        added.append(
            Change(
                number,
                "stack",
                "(not in label)",
                f"+ {technology.name}",
                RULE_STACK,
                flag + first.sentence,
                first.match_type,
            )
        )
    if len(added) > max_adds:
        shows = f"{len(added)} entries would be added (over {max_adds}): none added"
        result.reviews.append(Review(number, "stack", f"{len(labels.stack)} entries", shows))
        return labels
    result.changes.extend(added)
    names = [change.after[2:] for change in added]
    return labels.model_copy(update={"stack": [*labels.stack, *names]}) if names else labels


# --- the whole pass ------------------------------------------------------------------------------


@dataclass
class Plan:
    fixed: LabelFile
    changes: list[Change]
    reviews: list[Review]
    infos: list[str]


def plan_completion(
    label_file: LabelFile,
    texts: Mapping[uuid.UUID, str],
    policy: Policy,
    allow_names: Sequence[str],
    max_adds: int = DEFAULT_MAX_ADDS,
) -> Plan:
    technologies = known_technologies(policy, allow_names)
    posts: list[LabelEntry] = []
    changes: list[Change] = []
    reviews: list[Review] = []
    infos: list[str] = []
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
        result = PostResult(entry.labels)
        result.labels = complete_salary(number, result.labels, text, result)
        result.labels = complete_stack(number, result.labels, text, technologies, result, max_adds)
        changes.extend(result.changes)
        reviews.extend(result.reviews)
        infos.extend(result.infos)
        posts.append(entry.model_copy(update={"labels": result.labels}))
    return Plan(label_file.model_copy(update={"posts": posts}), changes, reviews, infos)


CHANGEABLE = frozenset({"salary_min", "salary_max", "salary_currency", "stack"})


def verify_only_allowed_changes(before: LabelFile, after: LabelFile) -> None:
    """Refuse to write unless only blank salary parts were filled and stack entries appended."""
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
        if old.labels.model_dump(exclude=set(CHANGEABLE)) != new.labels.model_dump(
            exclude=set(CHANGEABLE)
        ):
            raise ScriptError("refusing to write: a field outside salary and stack changed")
        for name in ("salary_min", "salary_max", "salary_currency"):
            was = getattr(old.labels, name)
            if was is not None and was != getattr(new.labels, name):
                raise ScriptError("refusing to write: a filled salary part was changed")
        if new.labels.stack[: len(old.labels.stack)] != old.labels.stack:
            raise ScriptError("refusing to write: a stack entry was removed or reordered")


# --- output and the command ----------------------------------------------------------------------


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [" | ".join(cell.ljust(w) for cell, w in zip(header, widths, strict=True))]
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows)
    return lines


def format_changes(changes: Sequence[Change]) -> list[str]:
    rows = [
        (str(c.post), c.field, c.before, c.after, c.rule, c.match or "-", c.note) for c in changes
    ]
    return _table(("post", "field", "before", "after", "rule", "match", "sentence"), rows)


def format_reviews(reviews: Sequence[Review]) -> list[str]:
    rows = [
        (str(r.post), r.field, r.current, r.shows)
        for r in sorted(reviews, key=lambda r: (r.post, r.field))
    ]
    return _table(("post", "field", "current label", "what the post shows"), rows)


def back_up(path: Path) -> tuple[Path, bool]:
    """Copy the labels to `<name>.pass1.json` first. An existing copy is never overwritten."""
    backup = path.with_name(f"{path.stem}{BACKUP_SUFFIX}{path.suffix}")
    if backup.exists():
        return backup, False
    shutil.copyfile(path, backup)
    return backup, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_complete_labels",
        description="Second pass: add the salary parts and technologies the post states.",
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="show the table; write nothing")
    parser.add_argument(
        "--max-adds-per-post",
        type=int,
        default=DEFAULT_MAX_ADDS,
        help=f"a post gaining more stack entries goes to NEEDS_HUMAN (default {DEFAULT_MAX_ADDS})",
    )
    return parser


def _show_plan(plan: Plan, session: Session) -> None:
    if plan.changes:
        session.out("\n".join(format_changes(plan.changes)))
    posts = len({c.post for c in plan.changes})
    flagged = sum(c.note.startswith("FLAG") for c in plan.changes)
    session.out(
        f"\n{len(plan.changes)} change(s) in {posts} post(s); {flagged} flagged "
        "(capitalised ambiguous word)"
    )
    for line in plan.infos:
        session.out(f"  {INFO}: {line}")
    session.out(f"  {NEEDS_HUMAN}: {len(plan.reviews)} item(s); nothing in that table is written")
    if plan.reviews:
        session.out("\n" + NEEDS_HUMAN + "\n" + "\n".join(format_reviews(plan.reviews)))


def run_complete(
    labels_path: Path,
    settings: Settings,
    session: Session,
    dry_run: bool = False,
    max_adds: int = DEFAULT_MAX_ADDS,
) -> int:
    """Plan, show, ask, write, re-check. Reads the database read-only."""
    if max_adds < 1:
        raise ScriptError("--max-adds-per-post must be at least 1")
    if not labels_path.is_file():
        raise ScriptError(f"{labels_path} does not exist")
    label_file = load_label_file(labels_path)
    target = database_target(settings.database_url)  # refuses the wrong database up front
    ids = [entry.job_id for entry in label_file.posts if entry.labels is not None]
    session.out(f"Reading post text (read-only) from database {target}")
    texts = dict(asyncio.run(read_posts_by_id(settings.database_url, ids)))
    policy = load_policy(DEFAULT_POLICY_PATH)
    allow_names = load_allow_names(DEFAULT_POLICY_PATH)
    plan = plan_completion(label_file, texts, policy, allow_names, max_adds)
    verify_only_allowed_changes(label_file, plan.fixed)
    _show_plan(plan, session)
    if dry_run:
        session.out("Dry run: no backup, no question, nothing is written.")
        return 0
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
    backup, created = back_up(labels_path)
    session.out(f"Backup: {backup} ({'written' if created else 'already there, kept as it is'})")
    save_label_file(labels_path, plan.fixed)
    session.out("Written. Checking the labels against the post text again (codes only):")
    report = check_labels(load_label_file(labels_path), texts)
    for line in format_report(report):
        session.out(line)
    return 0 if not report.problems else 1


def main(argv: Sequence[str] | None = None, session: Session | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_complete(
            args.labels,
            get_settings(),
            session or Session(),
            args.dry_run,
            args.max_adds_per_post,
        )
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
