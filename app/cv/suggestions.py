"""Suggestions read from a CV's text: years of experience and a seniority level.

They are only ever *shown* to the user, each with the kind of match that produced it, so the
user can judge it. Nothing scores or filters with them: a heuristic guess about a person must not
silently decide which jobs they see.

Pure, with bounded patterns (a CV is untrusted text of up to 30,000 characters).
"""

import re
from dataclasses import dataclass

from app.extraction.seniority import levels_in
from app.models import Seniority

YEARS_PHRASE = "years phrase"
TITLE_IN_HEADLINE = "title keyword in the first lines"
TITLE_IN_CURRENT_ROLE = "title keyword in the current role"

MAX_YEARS = 50  # a larger number is a typo or a date, not a career length
_HEADLINE_LINES = 3  # name, title, contact line: further down a line is a summary or a role entry
_LINE_CHARS = 120  # a title line is short; a long line is a sentence
_BULLETS = "-*•–—"

# "6 years of experience", "6+ years' experience", "over 6 years professional experience".
# It must say "experience" right after the years (and a few adjectives): "5 years of Python" is
# one skill's length, not the career's. Every quantifier is bounded, so the scan is linear.
_YEARS = re.compile(
    r"\b(\d{1,2})[ \t]{0,2}\+?[ \t]{0,2}(?:years?|yrs?)['’]?[ \t]{1,3}(?:of[ \t]{1,3})?"
    r"(?:(?:professional|industry|hands-on|relevant|total|working|software|commercial|"
    r"engineering|development|backend|frontend|full-stack)[ \t]{1,3}){0,3}experience\b",
    re.IGNORECASE,
)
_CURRENT = re.compile(r"\b(?:present|current|now|today)\b", re.IGNORECASE)


@dataclass(frozen=True)
class YearsSuggestion:
    value: int
    match: str


@dataclass(frozen=True)
class SenioritySuggestion:
    value: Seniority
    match: str


@dataclass(frozen=True)
class Suggestions:
    years_of_experience: YearsSuggestion | None
    seniority: SenioritySuggestion | None


def suggest_years(text: str) -> YearsSuggestion | None:
    """The largest "N years of experience" phrase, when it is a believable number."""
    years = [int(m.group(1)) for m in _YEARS.finditer(text)]
    believable = [n for n in years if 0 < n <= MAX_YEARS]
    return YearsSuggestion(max(believable), YEARS_PHRASE) if believable else None


def _title_lines(lines: list[str], *, dated: bool) -> list[str]:
    """Lines that can be a job title: short, and not a bullet point.

    A headline carries no digits ("Senior Backend Engineer"); a line with a year is a role entry,
    which is only used through the "until present" rule (`dated=True` keeps those lines).
    """
    return [
        line
        for line in lines
        if len(line) <= _LINE_CHARS
        and not line.startswith(tuple(_BULLETS))
        and (dated or not any(char.isdigit() for char in line))
    ]


def _single_level(lines: list[str]) -> Seniority | None:
    """The one level the lines name; None when they name none or disagree ("Senior ... Junior")."""
    levels = levels_in("\n".join(lines))
    return levels.pop() if len(levels) == 1 else None


def suggest_seniority(text: str) -> SenioritySuggestion | None:
    """A level from a title keyword: the first lines first, then a role that runs until today."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    headline = _single_level(_title_lines(lines[:_HEADLINE_LINES], dated=False))
    if headline is not None:
        return SenioritySuggestion(headline, TITLE_IN_HEADLINE)
    current_roles = [line for line in lines if _CURRENT.search(line)]
    current = _single_level(_title_lines(current_roles, dated=True))
    if current is not None:
        return SenioritySuggestion(current, TITLE_IN_CURRENT_ROLE)
    return None


def suggest(text: str) -> Suggestions:
    return Suggestions(years_of_experience=suggest_years(text), seniority=suggest_seniority(text))
