import time

import pytest

from app.cv.text import LINK_PLACEHOLDER, clean_cv_text, mask_cv_text
from tests.cv_helpers import synthetic_cv_text

# --- cleaning -----------------------------------------------------------------------------------


def test_line_endings_are_unified_and_lines_are_kept() -> None:
    assert clean_cv_text("EXPERIENCE\r\nBuilt APIs\rLed a team") == (
        "EXPERIENCE\nBuilt APIs\nLed a team"
    )


def test_runs_of_spaces_and_blank_lines_are_collapsed() -> None:
    assert clean_cv_text("Python    FastAPI\n\n\n\n\nDocker\t\t Redis") == (
        "Python FastAPI\n\nDocker Redis"
    )


def test_control_and_invisible_characters_are_removed_but_tabs_and_newlines_stay() -> None:
    dirty = (
        "Py"
        + chr(0)
        + "thon"
        + chr(7)
        + " and"
        + chr(0x200B)
        + " Dock"
        + chr(0xAD)
        + "er"
        + chr(10)
        + "line two"
    )

    assert clean_cv_text(dirty) == "Python and Docker\nline two"


def test_unicode_is_normalised_so_accented_letters_compare_equal() -> None:
    assert clean_cv_text("Cafe" + chr(0x301)) == "Caf" + chr(0xE9)


def test_text_with_only_whitespace_cleans_to_nothing() -> None:
    assert clean_cv_text(" \n\t \r\n ") == ""


# --- masking ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "write to alex.example@example.com today",
        "write to alex.example [at] example [dot] com today",
    ],
)
def test_email_addresses_are_masked(text: str) -> None:
    masked = mask_cv_text(text)

    assert "example.com" not in masked
    assert "[email]" in masked


def test_phone_numbers_are_masked() -> None:
    masked = mask_cv_text("call +1 555 010 0199 or (415) 555-0100 or 415-555-0100")

    assert "555" not in masked
    assert masked.count("[phone]") == 3


@pytest.mark.parametrize(
    "link",
    [
        "https://github.com/alex-example",
        "http://example.com/portfolio?x=1",
        "www.alex-example.dev",
        "linkedin.com/in/alex-example",
        "github.com/alex-example/project",
    ],
)
def test_links_are_masked(link: str) -> None:
    masked = mask_cv_text(f"see {link} for more")

    assert masked == f"see {LINK_PLACEHOLDER} for more"


def test_technology_names_that_look_like_domains_are_left_alone() -> None:
    text = "Node.js, Next.js, ASP.NET, Vue.js, socket.io and D3.js"

    assert mask_cv_text(text) == text


def test_the_synthetic_cv_loses_every_contact_detail_and_keeps_its_skills() -> None:
    masked = mask_cv_text(clean_cv_text(synthetic_cv_text()))

    for leaked in ("@example.com", "555 010", "github.com", "linkedin.com"):
        assert leaked not in masked
    assert "Python" in masked and "FastAPI" in masked and "PostgreSQL" in masked


@pytest.mark.parametrize(
    "hostile",
    [
        "a" * 30_000,
        "http://" + "a" * 30_000,
        "www." + "." * 30_000,
        "+1" + "1 " * 15_000,
        "@" * 30_000,
        "a@" * 15_000,
        "(" * 30_000,
        "linkedin.com/" + "x" * 30_000,
    ],
)
def test_hostile_input_is_processed_quickly(hostile: str) -> None:
    started = time.perf_counter()

    mask_cv_text(clean_cv_text(hostile))

    assert time.perf_counter() - started < 1.0
