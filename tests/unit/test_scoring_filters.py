"""The hard filters: pass, fail or unknown, and an unknown never fails. Pure, invented values."""

import pytest

from app.matching.scoring import PERIODS_PER_YEAR, run_filters, salary_filter
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from tests.matching_helpers import job_facts, preferences


def by_name(job_kwargs: dict[str, object], prefs: dict[str, object] | None) -> dict[str, object]:
    filters = run_filters(
        job_facts(**job_kwargs), preferences(**prefs) if prefs is not None else None
    )
    return {f.name: f for f in filters}


# --- remote policy and seniority share one rule ---
@pytest.mark.parametrize(
    ("job_value", "wanted", "outcome", "reason"),
    [
        (RemotePolicy.REMOTE, [RemotePolicy.REMOTE], "pass", None),
        (RemotePolicy.REMOTE, [RemotePolicy.REMOTE, RemotePolicy.HYBRID], "pass", None),
        (RemotePolicy.ONSITE, [RemotePolicy.REMOTE], "fail", None),
        (RemotePolicy.HYBRID, [RemotePolicy.REMOTE], "fail", None),
        (None, [RemotePolicy.REMOTE], "unknown", "job_value_missing"),
        (RemotePolicy.ONSITE, [], "unknown", "no_preference"),
        (None, [], "unknown", "no_preference"),
    ],
)
def test_remote_policy_filter(job_value, wanted, outcome, reason) -> None:
    result = by_name({"remote_policy": job_value}, {"remote_policies": wanted})["remote_policy"]

    assert (result.outcome, result.reason) == (outcome, reason)


@pytest.mark.parametrize(
    ("job_value", "wanted", "outcome", "reason"),
    [
        (Seniority.SENIOR, [Seniority.SENIOR, Seniority.LEAD], "pass", None),
        (Seniority.JUNIOR, [Seniority.SENIOR, Seniority.LEAD], "fail", None),
        (None, [Seniority.SENIOR], "unknown", "job_value_missing"),
        (Seniority.MID, [], "unknown", "no_preference"),
    ],
)
def test_seniority_filter(job_value, wanted, outcome, reason) -> None:
    result = by_name({"seniority": job_value}, {"seniority_levels": wanted})["seniority"]

    assert (result.outcome, result.reason) == (outcome, reason)


def test_the_filter_reports_the_values_it_compared() -> None:
    result = by_name(
        {"remote_policy": RemotePolicy.ONSITE},
        {"remote_policies": [RemotePolicy.REMOTE, RemotePolicy.HYBRID]},
    )["remote_policy"]

    assert result.job == "onsite"
    assert result.preference == ["remote", "hybrid"]


def test_no_preferences_row_at_all_makes_every_filter_unknown() -> None:
    results = by_name({"remote_policy": RemotePolicy.ONSITE, "seniority": Seniority.JUNIOR}, None)

    assert {f.outcome for f in results.values()} == {"unknown"}
    assert {f.reason for f in results.values()} == {"no_preference"}


# --- salary ---
WANT = {"min_salary": 100_000, "salary_currency": "USD", "salary_period": SalaryPeriod.YEAR}


def salary(job_kwargs: dict[str, object], want: dict[str, object] | None = None):
    return salary_filter(job_facts(**job_kwargs), preferences(**(WANT if want is None else want)))


def test_a_top_that_reaches_the_minimum_passes() -> None:
    result = salary(
        {
            "salary_min": 90_000,
            "salary_max": 120_000,
            "salary_currency": "USD",
            "salary_period": SalaryPeriod.YEAR,
        }
    )

    assert result.outcome == "pass"
    assert (result.job_per_year, result.preference_per_year) == (120_000, 100_000)


def test_a_stated_maximum_below_the_minimum_fails() -> None:
    result = salary(
        {
            "salary_min": 50_000,
            "salary_max": 80_000,
            "salary_currency": "USD",
            "salary_period": SalaryPeriod.YEAR,
        }
    )

    assert result.outcome == "fail"
    assert result.reason is None


def test_a_single_amount_equal_to_the_minimum_passes() -> None:
    result = salary(
        {
            "salary_min": 100_000,
            "salary_max": 100_000,
            "salary_currency": "USD",
            "salary_period": SalaryPeriod.YEAR,
        }
    )

    assert result.outcome == "pass"


def test_from_x_under_the_minimum_stays_unknown_because_the_pay_may_be_higher() -> None:
    result = salary(
        {"salary_min": 70_000, "salary_currency": "USD", "salary_period": SalaryPeriod.YEAR}
    )

    assert (result.outcome, result.reason) == ("unknown", "open_ended_below_minimum")


def test_from_x_at_or_above_the_minimum_passes() -> None:
    result = salary(
        {"salary_min": 120_000, "salary_currency": "USD", "salary_period": SalaryPeriod.YEAR}
    )

    assert result.outcome == "pass"


def test_up_to_x_that_reaches_the_minimum_passes_and_below_it_fails() -> None:
    reaching = salary(
        {"salary_max": 110_000, "salary_currency": "USD", "salary_period": SalaryPeriod.YEAR}
    )
    short = salary(
        {"salary_max": 90_000, "salary_currency": "USD", "salary_period": SalaryPeriod.YEAR}
    )

    assert (reaching.outcome, short.outcome) == ("pass", "fail")


def test_currencies_are_never_converted() -> None:
    result = salary(
        {"salary_max": 500_000, "salary_currency": "EUR", "salary_period": SalaryPeriod.YEAR}
    )

    assert (result.outcome, result.reason) == ("unknown", "currency_differs")


def test_an_amount_without_a_period_is_unknown() -> None:
    result = salary({"salary_max": 150_000, "salary_currency": "USD", "salary_period": None})

    assert (result.outcome, result.reason) == ("unknown", "salary_period_missing")


def test_an_amount_without_a_currency_is_unknown() -> None:
    result = salary(
        {"salary_max": 150_000, "salary_currency": None, "salary_period": SalaryPeriod.YEAR}
    )

    assert (result.outcome, result.reason) == ("unknown", "job_value_missing")


def test_a_post_without_a_salary_is_unknown_and_shows_no_job_value() -> None:
    result = salary({})

    assert (result.outcome, result.reason) == ("unknown", "job_value_missing")
    assert result.job is None


def test_no_minimum_set_is_unknown() -> None:
    result = salary_filter(job_facts(salary_max=1), preferences())
    none_at_all = salary_filter(job_facts(salary_max=1), None)

    assert (result.outcome, result.reason) == ("unknown", "no_preference")
    assert (none_at_all.outcome, none_at_all.reason) == ("unknown", "no_preference")


@pytest.mark.parametrize(
    ("period", "amount", "outcome"),
    [
        (SalaryPeriod.MONTH, 8_400, "pass"),  # 100,800 a year
        (SalaryPeriod.MONTH, 8_300, "fail"),  # 99,600
        (SalaryPeriod.WEEK, 1_925, "pass"),  # 100,100
        (SalaryPeriod.DAY, 385, "pass"),  # 100,100 on 260 working days
        (SalaryPeriod.DAY, 384, "fail"),  # 99,840
        (SalaryPeriod.HOUR, 49, "pass"),  # 101,920 on 2,080 hours
        (SalaryPeriod.HOUR, 48, "fail"),  # 99,840
    ],
)
def test_periods_are_compared_per_year(period: SalaryPeriod, amount: int, outcome: str) -> None:
    result = salary({"salary_max": amount, "salary_currency": "USD", "salary_period": period})

    assert result.outcome == outcome
    assert result.job_per_year == amount * PERIODS_PER_YEAR[period]


def test_the_preference_period_is_converted_too() -> None:
    result = salary(
        {"salary_max": 9_000, "salary_currency": "USD", "salary_period": SalaryPeriod.MONTH},
        {"min_salary": 8_000, "salary_currency": "USD", "salary_period": SalaryPeriod.MONTH},
    )

    assert result.outcome == "pass"
    assert (result.job_per_year, result.preference_per_year) == (108_000, 96_000)
