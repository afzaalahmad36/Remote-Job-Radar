import pytest

from app.extraction.salary import Salary, find_salary
from app.extraction.types import Confidence
from app.models import SalaryPeriod

Y, M, W, D, H = (
    SalaryPeriod.YEAR,
    SalaryPeriod.MONTH,
    SalaryPeriod.WEEK,
    SalaryPeriod.DAY,
    SalaryPeriod.HOUR,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Salary: $90,000 - $120,000 per year", Salary(90_000, 120_000, "USD", Y)),
        ("$80k-$120k a year", Salary(80_000, 120_000, "USD", Y)),
        ("$80-120k annually", Salary(80_000, 120_000, "USD", Y)),
        ("€50.000 – €65.000 / year", Salary(50_000, 65_000, "EUR", Y)),
        ("£45,000 to £55,000 per annum", Salary(45_000, 55_000, "GBP", Y)),
        ("USD 5000 - 7000 per month", Salary(5_000, 7_000, "USD", M)),
        ("80,000 - 100,000 EUR yearly", Salary(80_000, 100_000, "EUR", Y)),
        ("PKR 400,000 monthly", Salary(400_000, 400_000, "PKR", M)),
        ("$50/hr", Salary(50, 50, "USD", H)),
        ("$45 per hour", Salary(45, 45, "USD", H)),
        ("$400 per day", Salary(400, 400, "USD", D)),
        ("$1,500 a week", Salary(1_500, 1_500, "USD", W)),
        ("C$100k - C$130k per year", Salary(100_000, 130_000, "CAD", Y)),
        ("up to $150k per year", Salary(None, 150_000, "USD", Y)),
        ("from $90k per year", Salary(90_000, None, "USD", Y)),
        ("$100k+ per year", Salary(100_000, None, "USD", Y)),
        ("Compensation: $120,000", Salary(120_000, 120_000, "USD", None)),
    ],
)
def test_salary_formats_are_read_as_stated(text: str, expected: Salary) -> None:
    found = find_salary(text)

    assert found is not None
    assert found.value == expected


def test_a_stated_period_or_a_salary_word_makes_it_high_confidence() -> None:
    assert find_salary("$90k per year").confidence is Confidence.HIGH  # type: ignore[union-attr]
    assert find_salary("Salary $90k").confidence is Confidence.HIGH  # type: ignore[union-attr]


def test_a_bare_amount_without_period_or_salary_word_is_low_confidence() -> None:
    found = find_salary("Join us for $90k")

    assert found is not None
    assert found.confidence is Confidence.LOW


@pytest.mark.parametrize(
    "text",
    [
        "5 years of experience",
        "10k users and 3M requests",
        "Call 415-555-0100 now",
        "Posted 2026-10-28",
        "We are 50 people in 12 countries",
        "Salary: 80-120k",  # no currency: not provably money
        "$5",  # implausible as a yearly amount
        "",
    ],
)
def test_numbers_that_are_not_salaries_are_ignored(text: str) -> None:
    assert find_salary(text) is None


def test_a_number_after_the_amount_is_not_taken_as_the_top_of_a_range() -> None:
    found = find_salary("$90k and 5 years of experience")

    assert found is not None
    assert found.value == Salary(90_000, 90_000, "USD", None)


def test_a_range_is_never_returned_upside_down() -> None:
    found = find_salary("$150,000 - $90,000 per year")

    assert found is not None
    assert (
        found.value.minimum is None
        or found.value.maximum is None
        or (found.value.minimum <= found.value.maximum)
    )


def test_two_currencies_are_never_joined_into_one_range() -> None:
    found = find_salary("$80k - €100k per year")

    assert found is not None
    assert found.value == Salary(80_000, 80_000, "USD", Y)  # only the first amount, as a single


def test_the_best_backed_candidate_wins_over_an_earlier_weak_one() -> None:
    found = find_salary("Bonus $2,000 once. Salary $90,000 - $110,000 per year.")

    assert found is not None
    assert found.value == Salary(90_000, 110_000, "USD", Y)


def test_amounts_outside_the_plausible_range_for_their_period_are_dropped() -> None:
    assert find_salary("$900,000 per hour") is None
    assert find_salary("$3 per month") is None
