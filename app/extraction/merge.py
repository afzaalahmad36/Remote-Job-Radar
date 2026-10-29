"""Combining the rules' result with the language model's.

The one principle: the model never overrides a HIGH-confidence rule value (the rules point at
evidence in the text). It only fills gaps and replaces LOW guesses. The model's output is
untrusted, so a link and a salary are also checked against the post before they are accepted.
"""

import re
from dataclasses import dataclass

from app.core.url_safety import UnsafeUrlError, check_url_syntax
from app.extraction.rules import RuleResult
from app.extraction.salary import Salary
from app.extraction.schema import JobExtraction
from app.extraction.stack import MAX_STACK, canonical_name
from app.extraction.types import Found
from app.models import RemotePolicy, SalaryPeriod, Seniority

# A post is "complete" when its title and stack are known with confidence. Everything else is
# optional and never costs a request: company and remote policy are often absent or only
# guessable, salary and seniority are legitimately missing from most posts, and the apply contact
# is left out on purpose. On two real public feeds, a post's own
# link is not in its text and a job board's apply link is deliberately a LOW-confidence value, so
# a model could not fix a missing contact there; requiring one only spent requests for nothing.
KEY_FIELDS = ("title", "stack")

_NUMBER = re.compile(r"\d{1,3}(?:[,.\s]\d{3})+|\d+(?:\.\d+)?")


def missing_key_fields(rules: RuleResult) -> tuple[str, ...]:
    """Key fields the rules did not find, or found only as a low-confidence guess."""
    known = {
        "title": rules.title is not None and rules.title.is_high,
        "stack": bool(rules.stack),
    }
    return tuple(name for name in KEY_FIELDS if not known[name])


def is_too_short_to_enrich(text: str, min_chars: int) -> bool:
    """True for a headline or stub the model could not add anything to: it stays rules-only.

    Whitespace around the text does not count. `min_chars` of 0 turns the gate off.
    """
    return len(text.strip()) < min_chars


@dataclass(frozen=True)
class MergedExtraction:
    is_job_posting: bool
    title: str | None
    company: str | None
    stack: tuple[str, ...]
    seniority: Seniority | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    remote_policy: RemotePolicy | None
    apply_email: str | None
    apply_url: str | None
    conflicts: tuple[str, ...]  # fields where a HIGH rule value won over a different model value


def merge(rules: RuleResult, model: JobExtraction | None, text: str) -> MergedExtraction:
    """The final values for one post. `model` is None when no model was (or could be) asked."""
    conflicts: list[str] = []

    def pick[T: str](name: str, found: Found[T] | None, from_model: T | None) -> T | None:
        if found is not None and (found.is_high or from_model is None):
            if found.is_high and from_model is not None and _differs(found.value, from_model):
                conflicts.append(name)
            return found.value
        return from_model

    salary = _salary(rules.salary, model, text)
    return MergedExtraction(
        is_job_posting=model.is_job_posting if model else True,
        title=pick("title", rules.title, model.title if model else None),
        company=pick("company", rules.company, model.company if model else None),
        stack=_stack(rules.stack, model),
        seniority=pick("seniority", rules.seniority, model.seniority if model else None),
        salary_min=salary.minimum if salary else None,
        salary_max=salary.maximum if salary else None,
        salary_currency=salary.currency if salary else None,
        salary_period=salary.period if salary else None,
        remote_policy=pick(
            "remote_policy", rules.remote_policy, model.remote_policy if model else None
        ),
        apply_email=rules.apply_email.value if rules.apply_email else None,
        apply_url=pick("apply_url", rules.apply_url, _safe_url(model, text)),
        conflicts=tuple(conflicts),
    )


def _differs(left: str, right: str) -> bool:
    return left.casefold() != right.casefold()  # enums are strings too


def _stack(rule_hits: tuple[str, ...], model: JobExtraction | None) -> tuple[str, ...]:
    """Every technology either side found; the rules' dictionary hits come first."""
    merged: dict[str, str] = {}
    candidates = [
        *rule_hits,
        *((canonical_name(item) or item) for item in (model.stack if model else [])),
    ]
    for item in candidates:
        merged.setdefault(item.casefold(), item)
    return tuple(merged.values())[:MAX_STACK]


def _safe_url(model: JobExtraction | None, text: str) -> str | None:
    """The model's link, only if it is a plain http(s) URL that really appears in the post."""
    if model is None or model.apply_url is None:
        return None
    url = model.apply_url
    try:
        check_url_syntax(url)
    except UnsafeUrlError:
        return None
    return url if url.rstrip("/").casefold() in text.casefold() else None


def _salary(
    from_rules: Found[Salary] | None, model: JobExtraction | None, text: str
) -> Salary | None:
    """One salary group, never a mix: the rules' if HIGH, else the model's if the numbers are in
    the post, else whatever the rules guessed."""
    if from_rules is not None and from_rules.is_high:
        return from_rules.value
    if model is not None and _model_salary_is_supported(model, text):
        return Salary(
            model.salary_min, model.salary_max, model.salary_currency, model.salary_period
        )
    return from_rules.value if from_rules else None


def _model_salary_is_supported(model: JobExtraction, text: str) -> bool:
    amounts = [value for value in (model.salary_min, model.salary_max) if value is not None]
    if not amounts:
        return False
    present = _numbers_in(text)
    return all(amount in present for amount in amounts)


def _numbers_in(text: str) -> set[int]:
    """Every whole number the text writes, also as thousands: "80k" gives 80 and 80000."""
    present: set[int] = set()
    for match in _NUMBER.finditer(text):
        digits = (
            re.sub(r"[,.\s]", "", match.group())
            if re.search(r"[,.\s]\d{3}", match.group())
            else match.group()
        )
        number = float(digits)  # only digits and one optional decimal point remain
        present.add(int(number))
        present.add(int(number * 1000))  # "80k" means 80,000
    return present
