"""Suggestions from a CV: years of experience and seniority, each with its kind of match."""

import time

import pytest

from app.cv.suggestions import (
    TITLE_IN_CURRENT_ROLE,
    TITLE_IN_HEADLINE,
    YEARS_PHRASE,
    suggest,
    suggest_seniority,
    suggest_years,
)
from app.models import Seniority
from tests.cv_helpers import synthetic_cv_text

# --- years of experience ---------------------------------------------------------------------


def test_the_synthetic_cv_suggests_six_years_from_its_summary() -> None:
    result = suggest_years(synthetic_cv_text())

    assert result is not None
    assert (result.value, result.match) == (6, YEARS_PHRASE)


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("6 years of experience", 6),
        ("8+ years experience in backend", 8),
        ("Over 10 years' experience", 10),
        ("12 yrs of professional experience", 12),
        ("3 years of hands-on software development experience", 3),
        ("5 YEARS OF EXPERIENCE", 5),
        ("2 years of experience and, later, 7 years of experience", 7),  # the largest
    ],
)
def test_years_phrases_that_state_the_whole_career(text: str, years: int) -> None:
    result = suggest_years(text)

    assert result is not None and result.value == years


@pytest.mark.parametrize(
    "text",
    [
        "5 years of Python experience",  # one skill's length, not the career's
        "Python: 5 years",
        "Founded in 1990, 30 years ago",
        "0 years of experience",
        "99 years of experience",  # not believable
        "experience: 6 years",  # the number must come first
        "",
    ],
)
def test_phrases_that_are_not_a_career_length_are_ignored(text: str) -> None:
    assert suggest_years(text) is None


# --- seniority -------------------------------------------------------------------------------


def test_the_synthetic_cv_suggests_senior_from_the_current_role() -> None:
    result = suggest_seniority(synthetic_cv_text())

    assert result is not None
    assert (result.value, result.match) == (Seniority.SENIOR, TITLE_IN_CURRENT_ROLE)


def test_a_level_in_the_first_lines_is_a_headline_match() -> None:
    result = suggest_seniority("Alex Example\nStaff Software Engineer\nSUMMARY\nBuilds things")

    assert result is not None
    assert (result.value, result.match) == (Seniority.PRINCIPAL, TITLE_IN_HEADLINE)


def test_the_headline_wins_over_an_older_role() -> None:
    text = (
        "Alex\nLead Engineer\n\n"
        "Junior Developer, Old Co (2015 - 2017)\nSenior Developer, New Co (2020 - present)"
    )

    result = suggest_seniority(text)

    assert result is not None and result.value is Seniority.LEAD


@pytest.mark.parametrize(
    "text",
    [
        "Alex\nSenior or Junior Engineer",  # the line names two levels: no guess
        "Alex\nBackend Engineer\nSUMMARY\nI mentored two junior engineers",  # prose is not a title
        "Alex\nEngineer\n- Mentoring junior engineers (2022 - present)",  # a bullet is not a title
        "Alex\nBackend Engineer\nAcme (2019 - 2021)",  # a senior title that is over is not used
        "",
    ],
)
def test_no_suggestion_when_the_text_does_not_say_one_level(text: str) -> None:
    assert suggest_seniority(text) is None


def test_a_senior_title_that_ended_long_ago_is_not_the_current_level() -> None:
    assert suggest_seniority("Alex\nEngineer\nSenior Developer, Acme (2015 - 2018)") is None


def test_suggest_returns_both_parts() -> None:
    result = suggest(synthetic_cv_text())

    assert result.years_of_experience is not None and result.seniority is not None
    assert suggest("hello").years_of_experience is None


@pytest.mark.parametrize(
    "text",
    [
        "1" + " " * 30_000,
        "1 +" * 10_000,
        "1 years of " * 3_000,
        "senior\n" * 5_000,
        "present " * 4_000,
        "7 years " + "professional " * 2_500,
    ],
    ids=["spaces", "plusses", "repeated phrase", "many lines", "present", "adjectives"],
)
def test_hostile_cv_text_is_scanned_quickly(text: str) -> None:
    started = time.perf_counter()
    suggest(text)

    assert time.perf_counter() - started < 0.5
