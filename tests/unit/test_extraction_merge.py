import pytest

from app.extraction.merge import (
    KEY_FIELDS,
    MergedExtraction,
    is_too_short_to_enrich,
    merge,
    missing_key_fields,
)
from app.extraction.rules import RuleResult, extract_with_rules
from app.extraction.salary import Salary
from app.extraction.schema import JobExtraction
from app.extraction.types import Confidence, Found
from app.models import RemotePolicy, SalaryPeriod, Seniority

HIGH, LOW = Confidence.HIGH, Confidence.LOW

COMPLETE_POST = """Senior Python Developer at Acme Corp

Acme Corp is hiring. Fully remote. Stack: Python, Django, PostgreSQL.
Salary $90,000 - $120,000 per year. Apply: https://boards.greenhouse.io/acme/jobs/1
"""


def model(**fields: object) -> JobExtraction:
    return JobExtraction.model_validate({"is_job_posting": True, **fields})


# --- which fields count as missing ------------------------------------------------------------


def test_a_post_with_a_title_and_stack_is_complete_without_a_company_or_contact() -> None:
    rules = extract_with_rules(COMPLETE_POST)  # the company is only a LOW-confidence guess

    assert rules.company is not None and not rules.company.is_high
    assert missing_key_fields(rules) == ()


def test_an_empty_result_misses_every_key_field() -> None:
    assert missing_key_fields(RuleResult()) == KEY_FIELDS == ("title", "stack")


def test_low_confidence_values_count_as_missing() -> None:
    rules = RuleResult(title=Found("Engineer", LOW), stack=("Python",))

    assert missing_key_fields(rules) == ("title",)


def test_everything_but_title_and_stack_never_triggers_a_model_call() -> None:
    rules = RuleResult(title=Found("Engineer", HIGH), stack=("Python",))

    assert missing_key_fields(rules) == ()  # no company, contact, remote policy, salary, seniority
    for optional in ("company", "apply_contact", "remote_policy", "salary", "seniority"):
        assert optional not in KEY_FIELDS


def test_a_post_with_only_a_stack_still_needs_its_title() -> None:
    rules = RuleResult(stack=("Python",), apply_email=Found("jobs@acme.com", HIGH))

    assert missing_key_fields(rules) == ("title",)


# --- the minimum-length gate -----------------------------------------------------------------


def test_a_headline_only_post_is_too_short_to_enrich() -> None:
    stub = "Acme (YC S21) Is Hiring a Senior Backend Engineer " * 4  # ~200 characters

    assert is_too_short_to_enrich(stub, 500)


def test_a_full_post_is_long_enough() -> None:
    assert not is_too_short_to_enrich(COMPLETE_POST * 10, 500)


def test_the_limit_itself_is_long_enough_and_surrounding_whitespace_does_not_count() -> None:
    assert not is_too_short_to_enrich("x" * 500, 500)
    assert is_too_short_to_enrich("x" * 499, 500)
    padded = " " * 400 + "x" * 10 + " " * 400
    assert is_too_short_to_enrich(padded, 500)


def test_a_limit_of_zero_turns_the_gate_off() -> None:
    assert not is_too_short_to_enrich("", 0)


# --- merging ----------------------------------------------------------------------------------


def test_without_a_model_the_rules_result_is_returned_as_is() -> None:
    rules = extract_with_rules(COMPLETE_POST)

    merged = merge(rules, None, COMPLETE_POST)

    assert merged.title == "Senior Python Developer"
    assert merged.seniority is Seniority.SENIOR
    assert merged.remote_policy is RemotePolicy.REMOTE
    assert (merged.salary_min, merged.salary_max, merged.salary_period) == (
        90_000,
        120_000,
        SalaryPeriod.YEAR,
    )
    assert merged.apply_url == "https://boards.greenhouse.io/acme/jobs/1"
    assert merged.is_job_posting is True
    assert merged.conflicts == ()


def test_the_model_fills_gaps_the_rules_left() -> None:
    merged = merge(RuleResult(), model(title="Data Analyst", company="Globex"), "text")

    assert (merged.title, merged.company) == ("Data Analyst", "Globex")


def test_a_high_confidence_rule_value_beats_a_different_model_value_and_is_logged() -> None:
    rules = RuleResult(title=Found("Backend Engineer", HIGH))

    merged = merge(rules, model(title="Totally Different"), "text")

    assert merged.title == "Backend Engineer"
    assert merged.conflicts == ("title",)


def test_the_same_value_in_a_different_case_is_not_a_conflict() -> None:
    rules = RuleResult(title=Found("Backend Engineer", HIGH))

    assert merge(rules, model(title="backend engineer"), "t").conflicts == ()


def test_a_high_confidence_seniority_beats_a_different_model_seniority() -> None:
    rules = RuleResult(seniority=Found(Seniority.SENIOR, HIGH))

    merged = merge(rules, model(seniority="junior"), "text")

    assert merged.seniority is Seniority.SENIOR
    assert merged.conflicts == ("seniority",)


def test_the_model_replaces_a_low_confidence_rule_value() -> None:
    rules = RuleResult(company=Found("Wrong Guess", LOW))

    assert merge(rules, model(company="Acme"), "t").company == "Acme"


def test_a_low_confidence_rule_value_is_kept_when_the_model_has_nothing() -> None:
    rules = RuleResult(company=Found("Acme", LOW))

    assert merge(rules, model(), "t").company == "Acme"


def test_stack_is_the_union_with_rule_hits_first_canonicalised_and_deduplicated() -> None:
    rules = RuleResult(stack=("Python", "PostgreSQL"))

    text = "We use Python, postgres, Terraform and an Obscure Tool."
    merged = merge(rules, model(stack=["postgres", "Terraform", "Obscure Tool", "python"]), text)

    assert merged.stack == ("Python", "PostgreSQL", "Terraform", "Obscure Tool")
    assert merged.dropped_stack == 0


def test_apply_email_comes_from_the_rules_only() -> None:
    rules = RuleResult(apply_email=Found("jobs@acme.com", HIGH))

    assert merge(rules, model(), "t").apply_email == "jobs@acme.com"
    assert merge(RuleResult(), model(), "t").apply_email is None


def test_a_not_a_job_verdict_is_passed_on() -> None:
    merged = merge(RuleResult(), model(is_job_posting=False), "t")

    assert merged.is_job_posting is False


# --- the model's output is checked against the post --------------------------------------------


def test_a_model_link_that_is_in_the_post_is_accepted() -> None:
    text = "Apply at https://acme.com/join/9 today"

    merged = merge(RuleResult(), model(apply_url="https://acme.com/join/9"), text)

    assert merged.apply_url == "https://acme.com/join/9"


def test_a_model_link_that_is_not_in_the_post_is_dropped() -> None:
    merged = merge(RuleResult(), model(apply_url="https://evil.example/phish"), "Apply by email")

    assert merged.apply_url is None


def test_unsafe_model_links_are_dropped_even_if_the_post_contains_them() -> None:
    for url in ("javascript:alert(1)", "http://localhost/x", "http://10.0.0.1/x", "ftp://a.com/x"):
        assert merge(RuleResult(), model(apply_url=url), f"see {url}").apply_url is None, url


def test_a_model_salary_is_accepted_only_if_its_numbers_are_in_the_post() -> None:
    supported = merge(
        RuleResult(),
        model(salary_min=80_000, salary_max=100_000, salary_currency="USD", salary_period="year"),
        "Pay is 80k to 100,000 per year",
    )
    invented = merge(
        RuleResult(),
        model(salary_min=500_000, salary_max=900_000, salary_currency="USD", salary_period="year"),
        "Competitive pay",
    )

    assert (supported.salary_min, supported.salary_max) == (80_000, 100_000)
    assert (invented.salary_min, invented.salary_max, invented.salary_currency) == (None,) * 3


def test_a_high_confidence_rule_salary_is_never_mixed_with_the_models() -> None:
    rules = RuleResult(salary=Found(Salary(90_000, 120_000, "USD", SalaryPeriod.YEAR), HIGH))

    merged = merge(rules, model(salary_min=1, salary_max=999_999, salary_currency="EUR"), "t")

    assert (merged.salary_min, merged.salary_max, merged.salary_currency) == (
        90_000,
        120_000,
        "USD",
    )


def test_a_low_confidence_rule_salary_is_replaced_by_a_supported_model_salary() -> None:
    rules = RuleResult(salary=Found(Salary(50_000, 50_000, "USD", None), LOW))

    merged = merge(
        rules,
        model(salary_min=60_000, salary_max=70_000, salary_currency="EUR", salary_period="year"),
        "We pay 60,000 - 70,000 euros per year",
    )

    assert (merged.salary_min, merged.salary_max, merged.salary_currency) == (60_000, 70_000, "EUR")


def test_the_merged_result_is_a_plain_immutable_value() -> None:
    merged = merge(RuleResult(), None, "")

    assert isinstance(merged, MergedExtraction)
    assert merged.stack == ()


# --- the model's stack is only accepted when the post contains it -------------------------------


def test_a_model_technology_the_post_does_not_contain_is_dropped_and_counted() -> None:
    merged = merge(RuleResult(), model(stack=["Terraform", "Kubernetes"]), "We use Terraform.")

    assert merged.stack == ("Terraform",)
    assert merged.dropped_stack == 1


def test_grounding_ignores_case() -> None:
    merged = merge(
        RuleResult(), model(stack=["terraform", "OBSCURE tool"]), "TERRAFORM, Obscure Tool"
    )

    assert merged.stack == ("Terraform", "OBSCURE tool")  # the dictionary spelling wins
    assert merged.dropped_stack == 0


def test_grounding_needs_a_whole_word() -> None:
    merged = merge(
        RuleResult(), model(stack=["Go", "Rust"]), "We love gopher plushies and trusty tools"
    )

    assert merged.stack == ()
    assert merged.dropped_stack == 2


@pytest.mark.parametrize("name", ["C++", "C#", ".NET", "Node.js", "Obscure-Tool"])
def test_grounding_handles_names_with_symbols(name: str) -> None:
    assert merge(RuleResult(), model(stack=[name]), f"Skills: {name}, and more").stack == (name,)
    assert merge(RuleResult(), model(stack=[name]), f"Skills: {name}x, and more").dropped_stack == 1


def test_the_canonical_spelling_in_the_post_grounds_a_variant_the_model_wrote() -> None:
    merged = merge(RuleResult(), model(stack=["postgres"]), "Experience with PostgreSQL")

    assert merged.stack == ("PostgreSQL",) and merged.dropped_stack == 0


def test_rule_hits_are_never_dropped_and_nothing_is_dropped_without_a_model() -> None:
    merged = merge(RuleResult(stack=("Python",)), None, "no technologies written here")

    assert merged.stack == ("Python",) and merged.dropped_stack == 0


def test_grounding_survives_a_hostile_stack_entry() -> None:
    entry = "(" * 50 + "a" * 50
    merged = merge(RuleResult(), model(stack=[entry]), "a" * 20_000)

    assert merged.stack == () and merged.dropped_stack == 1


# --- salary_period is never mixed between the rules and the model -------------------------------


def test_amounts_found_by_the_rules_without_a_period_keep_a_null_period() -> None:
    rules = RuleResult(salary=Found(Salary(90_000, 120_000, "USD", None), HIGH))
    text = "Pay is 90,000 - 120,000 USD"

    merged = merge(rules, model(salary_min=90_000, salary_max=120_000, salary_period="hour"), text)

    assert (merged.salary_min, merged.salary_max) == (90_000, 120_000)
    assert merged.salary_period is None  # the model's "hour" is not mixed in


def test_a_low_rule_salary_without_a_period_is_replaced_whole_never_blended() -> None:
    rules = RuleResult(salary=Found(Salary(90_000, 120_000, "USD", None), LOW))
    text = "Pay is 95,000 - 125,000 USD yearly"
    answer = model(
        salary_min=95_000, salary_max=125_000, salary_currency="USD", salary_period="year"
    )

    merged = merge(rules, answer, text)

    # One group: all of it from the model, none of it from the rules.
    assert (merged.salary_min, merged.salary_max, merged.salary_period) == (
        95_000,
        125_000,
        SalaryPeriod.YEAR,
    )


def test_an_unsupported_model_salary_never_lends_its_period_to_the_rules_amounts() -> None:
    rules = RuleResult(salary=Found(Salary(90_000, None, "USD", None), LOW))

    merged = merge(rules, model(salary_min=1, salary_max=2, salary_period="month"), "USD 90,000")

    assert (merged.salary_min, merged.salary_period) == (90_000, None)


# --- when the post "supports" a salary the model gave -------------------------------------------


def model_salary(text: str, **fields: object) -> MergedExtraction:
    return merge(RuleResult(), model(**fields), text)


@pytest.mark.parametrize(
    "written",
    ["80,000", "80 000", "80.000", "80000", "80k", "80K", "80 k", "$80k+", "up to 80,000 USD"],
)
def test_a_number_the_post_writes_in_any_of_these_ways_supports_the_models_amount(
    written: str,
) -> None:
    merged = model_salary(f"Salary: {written}", salary_min=80_000)

    assert merged.salary_min == 80_000


@pytest.mark.parametrize(
    "text",
    [
        "Competitive salary",  # no number at all
        "We have 80 years of history",  # a bare 80 is not 80,000: only a k suffix means thousands
        "80 kids",  # the k belongs to a word
        "Salary 800,000",  # a different number that merely contains the digits
        "Salary 80,00",
        "Salary 79,999",
    ],
)
def test_a_number_the_post_does_not_write_is_a_fabrication_and_is_rejected(text: str) -> None:
    merged = model_salary(text, salary_min=80_000, salary_currency="USD")

    assert (merged.salary_min, merged.salary_max, merged.salary_currency) == (None, None, None)


def test_a_fabricated_amount_falls_back_to_the_rules_not_to_the_model() -> None:
    rules = RuleResult(salary=Found(Salary(50_000, None, "USD", None), LOW))

    merged = merge(rules, model(salary_min=95_000, salary_currency="USD"), "Pay up to 50,000 USD")

    assert (merged.salary_min, merged.salary_currency) == (50_000, "USD")


def test_both_amounts_must_appear_so_one_invented_bound_rejects_the_whole_group() -> None:
    merged = model_salary("Salary from 80,000", salary_min=80_000, salary_max=120_000)

    assert (merged.salary_min, merged.salary_max) == (None, None)


def test_the_models_period_must_be_stated_in_the_post_too() -> None:
    answer = {"salary_min": 80_000, "salary_period": "year"}

    stated = model_salary("Salary 80,000 per year", **answer)
    unstated = model_salary("Salary 80,000", **answer)
    other_period = model_salary("Salary 80,000 per month", **answer)

    assert (stated.salary_min, stated.salary_period) == (80_000, SalaryPeriod.YEAR)
    # The amount is real, but the period the model added is not in the post: the group is
    # rejected as a whole, never kept without its period.
    assert (unstated.salary_min, unstated.salary_period) == (None, None)
    assert (other_period.salary_min, other_period.salary_period) == (None, None)


@pytest.mark.parametrize(
    ("period", "phrase"),
    [
        ("year", "80,000 annually"),
        ("month", "80,000/mo"),
        ("week", "80,000 weekly"),
        ("day", "80,000 per day"),
        ("hour", "80,000 hourly"),
    ],
)
def test_each_period_is_supported_by_its_own_words(period: str, phrase: str) -> None:
    merged = model_salary(f"Pay {phrase}", salary_min=80_000, salary_period=period)

    assert merged.salary_period == SalaryPeriod(period)


def test_a_model_without_a_period_needs_only_its_numbers_to_be_in_the_post() -> None:
    merged = model_salary("Salary 80,000", salary_min=80_000)

    assert (merged.salary_min, merged.salary_period) == (80_000, None)
