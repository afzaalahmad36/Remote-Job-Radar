"""Combining the rules' result with the language model's.

The one principle: the model never overrides a HIGH-confidence rule value (the rules point at
evidence in the text). It only fills gaps and replaces LOW guesses. The model's output is
untrusted, so a link and a salary are also checked against the post before they are accepted.
"""

import re
from dataclasses import dataclass

from app.core.url_safety import UnsafeUrlError, check_url_syntax
from app.extraction.rules import RuleResult
from app.extraction.salary import Salary, period_is_stated
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

# A number as a post writes it: "90,000", "90 000", "90.000", "90000", "92.5", and an optional
# k suffix ("90k", "90 K") that means thousands.
_NUMBER = re.compile(r"(\d{1,3}(?:[,.\s]\d{3})+|\d+(?:\.\d+)?)(\s?[kK](?![A-Za-z]))?")


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
    dropped_stack: int = 0  # model stack entries left out because the post does not contain them


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
    stack, dropped_stack = _stack(rules.stack, model, text)
    return MergedExtraction(
        is_job_posting=model.is_job_posting if model else True,
        title=pick("title", rules.title, model.title if model else None),
        company=pick("company", rules.company, model.company if model else None),
        stack=stack,
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
        dropped_stack=dropped_stack,
    )


def _differs(left: str, right: str) -> bool:
    return left.casefold() != right.casefold()  # enums are strings too


def _stack(
    rule_hits: tuple[str, ...], model: JobExtraction | None, text: str
) -> tuple[tuple[str, ...], int]:
    """Every technology the rules found plus the model's that the post really contains, and how
    many of the model's were dropped. The rules' dictionary hits come first."""
    merged: dict[str, str] = {}
    for item in rule_hits:
        merged.setdefault(item.casefold(), item)
    dropped = 0
    for entry in model.stack if model else []:
        name = canonical_name(entry) or entry
        if _appears_in(entry, text) or _appears_in(name, text):
            merged.setdefault(name.casefold(), name)
        else:
            dropped += 1  # the model named something the post does not say
    return tuple(merged.values())[:MAX_STACK], dropped


def _appears_in(name: str, text: str) -> bool:
    """True if `name` is in `text` as a whole word, ignoring case. The boundary is "not next to a
    letter or digit", because a plain word-boundary check breaks on names that end in + or #,
    like C++ and C#."""
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


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
    """One salary group, never a mix: the rules' if HIGH, else the model's if the post supports
    it (see `_model_salary_is_supported`), else whatever the rules guessed."""
    if from_rules is not None and from_rules.is_high:
        return from_rules.value
    if model is not None and _model_salary_is_supported(model, text):
        return Salary(
            model.salary_min, model.salary_max, model.salary_currency, model.salary_period
        )
    return from_rules.value if from_rules else None


def _model_salary_is_supported(model: JobExtraction, text: str) -> bool:
    """The model's salary group is accepted only if the post itself backs all of it:

    * every amount it gives (min and max) literally appears in the post as a number, written with
      commas, dots or spaces as thousands separators, or as a number with a k suffix ("90k" is
      90,000). A number is never accepted just because it is plausible;
    * and if it gives a period, the post says that period in words (per year, /mo, hourly, ...).
    """
    amounts = [value for value in (model.salary_min, model.salary_max) if value is not None]
    if not amounts:
        return False
    present = _numbers_in(text)
    if not all(amount in present for amount in amounts):
        return False
    return model.salary_period is None or period_is_stated(text, model.salary_period)


def _numbers_in(text: str) -> set[int]:
    """Every whole number the text writes. "90k" gives 90,000 (and not 90); a bare "90" gives 90."""
    present: set[int] = set()
    for match in _NUMBER.finditer(text):
        written, thousands = match.groups()
        digits = re.sub(r"[,.\s]", "", written) if re.search(r"[,.\s]\d{3}", written) else written
        number = float(digits)  # only digits and one optional decimal point remain
        present.add(round(number * 1000) if thousands else int(number))
    return present
