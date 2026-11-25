"""The headline numbers in the README must match the documents they come from.

If a result is re-measured, the doc changes; this test then fails until the README says the same,
so the first screen can never quietly quote an old number.
"""

from pathlib import Path

import pytest

from scripts import seed_demo

ROOT = Path(__file__).resolve().parent.parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
ACCURACY = (ROOT / "docs" / "ACCURACY.md").read_text(encoding="utf-8")
MATCHING = (ROOT / "docs" / "MATCHING.md").read_text(encoding="utf-8")

EXTRACTION_NUMBERS = [
    "5/30 = 16.7%",
    "8/30 = 26.7%",
    "13/30 = 43.3%",
    "7/30 = 23.3%",
    "4/30 = 13.3%",
]
FIELD_NUMBERS = ["93.3%", "96.7%", "56.7%", "80.0%", "60.0%"]
MATCHING_NUMBERS = ["6/15", "4/15", "9/15", "12/30"]


@pytest.mark.parametrize("number", EXTRACTION_NUMBERS + FIELD_NUMBERS)
def test_an_extraction_number_in_the_readme_is_in_the_accuracy_doc(number: str) -> None:
    assert number in README
    assert number in ACCURACY


@pytest.mark.parametrize("number", MATCHING_NUMBERS)
def test_a_matching_number_in_the_readme_is_in_the_matching_doc(number: str) -> None:
    assert number in README
    assert number in MATCHING


def test_the_readme_still_says_the_matching_evaluation_failed() -> None:
    assert "FAIL" in README
    assert "**Verdict: FAIL against the pass rule" in MATCHING


def test_the_first_screen_quotes_the_measured_results() -> None:
    first_screen = "\n".join(README.splitlines()[:16])

    assert "13/30" in first_screen
    assert "not met" in first_screen
    assert "failed" in first_screen


def test_the_demo_login_in_the_readme_is_the_scripts_fixed_password() -> None:
    assert f"{seed_demo.DEMO_EMAIL}` / `{seed_demo.DEMO_FIXED_PASSWORD}" in README
