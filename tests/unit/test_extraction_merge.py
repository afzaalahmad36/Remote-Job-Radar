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

    merged = merge(rules, model(stack=["postgres", "Terraform", "Obscure Tool", "python"]), "t")

    assert merged.stack == ("Python", "PostgreSQL", "Terraform", "Obscure Tool")


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
        "Pay is 80k to 100,000 for the year",
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
        "We pay 60,000 - 70,000 euros",
    )

    assert (merged.salary_min, merged.salary_max, merged.salary_currency) == (60_000, 70_000, "EUR")


def test_the_merged_result_is_a_plain_immutable_value() -> None:
    merged = merge(RuleResult(), None, "")

    assert isinstance(merged, MergedExtraction)
    assert merged.stack == ()
