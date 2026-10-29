import pytest

from app.extraction.contacts import (
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    deobfuscate,
    find_emails,
    mask_contact_details,
)


@pytest.mark.parametrize(
    ("text", "address"),
    [
        ("Write to jobs@acme.com today", "jobs@acme.com"),
        ("Write to Jobs@Acme.COM.", "jobs@acme.com"),  # lowercased, sentence-final dot dropped
        ("(careers@sub.acme.co.uk)", "careers@sub.acme.co.uk"),
        ("send to first.last+tag@acme.io;", "first.last+tag@acme.io"),
        ("<hr@acme.dev>", "hr@acme.dev"),
    ],
)
def test_emails_are_found_by_shape(text: str, address: str) -> None:
    assert [span.address for span in find_emails(text)] == [address]


@pytest.mark.parametrize("text", ["@acme", "user@", "a@b", "x@@y.com", "not an email", "a@b.c"])
def test_things_that_are_not_emails_are_not_found(text: str) -> None:
    assert find_emails(text) == []


def test_spans_point_at_the_address_in_the_text() -> None:
    text = "mail jobs@acme.com now"

    [span] = find_emails(text)

    assert text[span.start : span.end].lower() == "jobs@acme.com"


@pytest.mark.parametrize(
    "text",
    [
        "jobs [at] acme [dot] com",
        "jobs (at) acme (dot) com",
        "jobs{at}acme{dot}com",
        "jobs  [ AT ]  acme [DOT] com",
    ],
)
def test_obfuscated_emails_are_restored(text: str) -> None:
    assert deobfuscate(text) == "jobs@acme.com"


def test_masking_replaces_emails_including_obfuscated_ones_and_phone_numbers() -> None:
    text = "Mail jobs@acme.com or hr [at] acme [dot] com, call +44 20 7946 0958 or (415) 555-0100."

    masked = mask_contact_details(text)

    assert masked.count(EMAIL_PLACEHOLDER) == 2
    assert masked.count(PHONE_PLACEHOLDER) == 2
    assert "acme" not in masked
    assert "7946" not in masked


def test_masking_leaves_salaries_and_dates_alone() -> None:
    text = "Salary $120,000 - $150,000, posted 2026-10-28, 5 years experience"

    assert mask_contact_details(text) == text
