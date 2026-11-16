"""Seniority words, shared by the job-post rules and the CV suggestions. Pure: no I/O."""

import re

from app.models import Seniority

# Matched against lower-case text.
SENIORITY_WORDS: dict[Seniority, re.Pattern[str]] = {
    Seniority.INTERN: re.compile(r"\b(?:intern(?:ship)?|trainee|apprentice)\b"),
    Seniority.JUNIOR: re.compile(r"\b(?:junior|jr|entry[- ]level|graduate)\b"),
    Seniority.MID: re.compile(r"\b(?:mid[- ]?level|mid|intermediate|midweight)\b"),
    Seniority.SENIOR: re.compile(r"\b(?:senior|sr)\b"),
    Seniority.LEAD: re.compile(r"\blead\b(?!\s+gen)|\bhead of\b"),
    Seniority.PRINCIPAL: re.compile(r"\b(?:principal|staff|distinguished)\b"),
}


def levels_in(text: str) -> set[Seniority]:
    """Every seniority level whose keyword appears in the text."""
    lowered = text.lower()
    return {level for level, pattern in SENIORITY_WORDS.items() if pattern.search(lowered)}
