r"""The rules read untrusted text, so no input may make them slow or make them raise.

Each case is a 20,000-character string built to hurt a naive pattern: long runs of characters a
regex wants to consume, many repeated starts, unbalanced brackets, and so on. The limit is
about 25 times the typical time (a few milliseconds), yet below what a naive `[\w.]+@[\w.]+` email
pattern needs on the same input (over a second), so a quadratic pattern fails here. The per-test
pytest-timeout is the backstop that turns a true hang into a failure with a stack trace.
"""

import time

import pytest

from app.extraction.contacts import find_emails, mask_contact_details
from app.extraction.merge import merge
from app.extraction.rules import extract_with_rules
from app.extraction.salary import find_salary
from app.extraction.stack import find_stack
from app.ingestion.text import MAX_TEXT_CHARS
from app.llm.prompts import prepare_post_text

TIME_LIMIT_SECONDS = 0.5

HOSTILE_INPUTS = {
    "one long word": "a" * MAX_TEXT_CHARS,
    "only at signs": "@" * MAX_TEXT_CHARS,
    "word characters then no at": ("user" * 5000) + " nothing",
    "many near-emails": "a@" * (MAX_TEXT_CHARS // 2),
    "long local part": ("a" * 70 + "@") * 250,
    "long domain": "x@" + ("a." * 10_000),
    "whitespace only": " " * MAX_TEXT_CHARS,
    "newlines only": "\n" * MAX_TEXT_CHARS,
    "spaces around at-words": "[ " + " " * 3 + "at" + " " * 3 + "]" + " " * 10_000,
    "obfuscation brackets": "[at]" * (MAX_TEXT_CHARS // 4),
    "open brackets": "(" * MAX_TEXT_CHARS,
    "square brackets": "[" * MAX_TEXT_CHARS,
    "scheme repeated": "http://" * (MAX_TEXT_CHARS // 7),
    "one huge url": "https://" + "a" * (MAX_TEXT_CHARS - 8),
    "dollar signs": "$" * MAX_TEXT_CHARS,
    "digit run": "1" * MAX_TEXT_CHARS,
    "numbers with commas": "1,234," * (MAX_TEXT_CHARS // 6),
    "money ranges": "$1k-" * (MAX_TEXT_CHARS // 4),
    "k suffixes": "9k " * (MAX_TEXT_CHARS // 3),
    "dashes": "- " * (MAX_TEXT_CHARS // 2),
    "colons": "a: " * (MAX_TEXT_CHARS // 3),
    "pipes in headline": "Engineer | " * (MAX_TEXT_CHARS // 11),
    "hiring sentence prefix": "Capital " * (MAX_TEXT_CHARS // 8) + "is",
    "capitalised words": "Word " * (MAX_TEXT_CHARS // 5),
    "labels": "Company: " * (MAX_TEXT_CHARS // 9),
    "technology words": "python " * (MAX_TEXT_CHARS // 7),
    "dotted tokens": "a." * (MAX_TEXT_CHARS // 2),
    "unicode": "é字🙂" * (MAX_TEXT_CHARS // 3),
    "control characters": "\x00\x1f\x7f" * (MAX_TEXT_CHARS // 3),
    "phone-like": "+1 " + "2" * (MAX_TEXT_CHARS - 3),
    "remote everywhere": "remote " * (MAX_TEXT_CHARS // 7),
}


@pytest.mark.parametrize("text", HOSTILE_INPUTS.values(), ids=HOSTILE_INPUTS.keys())
def test_the_rules_stay_fast_and_do_not_raise_on_hostile_input(text: str) -> None:
    assert len(text) <= MAX_TEXT_CHARS + 10  # the cases really are post-sized

    started = time.perf_counter()
    result = extract_with_rules(text)
    elapsed = time.perf_counter() - started

    assert elapsed < TIME_LIMIT_SECONDS, f"rules took {elapsed:.2f}s"
    assert len(result.stack) <= 30


@pytest.mark.parametrize("text", HOSTILE_INPUTS.values(), ids=HOSTILE_INPUTS.keys())
def test_each_building_block_is_fast_on_hostile_input(text: str) -> None:
    started = time.perf_counter()
    find_emails(text)
    mask_contact_details(text)
    find_salary(text)
    find_stack(text)
    prepare_post_text(text, 12_000)
    merge(extract_with_rules(text), None, text)
    elapsed = time.perf_counter() - started

    assert elapsed < TIME_LIMIT_SECONDS * 2, f"took {elapsed:.2f}s"


def test_input_longer_than_a_post_is_cut_not_processed_whole() -> None:
    started = time.perf_counter()
    extract_with_rules("a " * 5_000_000)

    assert time.perf_counter() - started < TIME_LIMIT_SECONDS
