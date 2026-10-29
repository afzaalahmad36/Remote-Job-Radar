"""The contract for what the LLM may return. LLM output is untrusted: nothing reaches the database
without passing through `JobExtraction`."""

import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from app.models import RemotePolicy, SalaryPeriod, Seniority

MAX_TITLE_LENGTH = 500  # the jobs.title column size
MAX_COMPANY_LENGTH = 255
MAX_STACK_ITEMS = 30
MAX_STACK_ITEM_LENGTH = 100
MAX_URL_LENGTH = 2048
MAX_SALARY = 1_000_000_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁦-⁩]")
_WHITESPACE = re.compile(r"\s+")
_CURRENCY = re.compile(r"[A-Z]{3}")
_REMOTE_ALIASES = {
    "on-site": "onsite",
    "on_site": "onsite",
    "on site": "onsite",
    "in-office": "onsite",
}


def clean_text(value: str, max_length: int) -> str | None:
    """Plain single-spaced text without control or bidi-override characters, or None if empty."""
    text = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
    return text[:max_length].strip() or None


class JobExtraction(BaseModel):
    """What the model returns for one post. Unknown keys and wrong types are errors, not ignored.

    There is no apply_email on purpose: emails are removed from the text before it is sent to a
    provider (privacy), and the rules extractor owns that field.
    """

    model_config = ConfigDict(extra="forbid")

    is_job_posting: bool
    title: str | None = None
    company: str | None = None
    stack: list[str] = []
    seniority: Seniority | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None
    remote_policy: RemotePolicy | None = None
    apply_url: str | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, value: Any) -> Any:
        return clean_text(value, MAX_TITLE_LENGTH) if isinstance(value, str) else value

    @field_validator("company", mode="before")
    @classmethod
    def _company(cls, value: Any) -> Any:
        return clean_text(value, MAX_COMPANY_LENGTH) if isinstance(value, str) else value

    @field_validator("apply_url", mode="before")
    @classmethod
    def _url(cls, value: Any) -> Any:
        # Only trimmed here; whether it is a safe, real link is checked when results are merged.
        return clean_text(value, MAX_URL_LENGTH) if isinstance(value, str) else value

    @field_validator("stack", mode="before")
    @classmethod
    def _stack(cls, value: Any) -> Any:
        if value is None:
            return []  # "no technologies named" is the same as an empty list
        if not isinstance(value, list):
            return value  # the type check reports it
        seen: set[str] = set()
        items: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                return value  # let the type check report it
            item = clean_text(entry, MAX_STACK_ITEM_LENGTH)
            if item and item.casefold() not in seen:
                seen.add(item.casefold())
                items.append(item)
        return items[:MAX_STACK_ITEMS]

    @field_validator("seniority", "salary_period", "remote_policy", mode="before")
    @classmethod
    def _enum_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            return _REMOTE_ALIASES.get(text, text) or None
        return value

    @field_validator("salary_currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().upper()
            if text and not _CURRENCY.fullmatch(text):
                raise ValueError("salary_currency must be a 3-letter code")
            return text or None
        return value

    @field_validator("salary_min", "salary_max")
    @classmethod
    def _amount(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= MAX_SALARY:
            raise ValueError("salary out of range")
        return value

    @model_validator(mode="after")
    def _salary_is_consistent(self) -> Self:
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError("salary_min is greater than salary_max")
        if self.salary_min is None and self.salary_max is None:
            self.salary_currency = None  # a currency or period without amounts means nothing
            self.salary_period = None
        return self


class ExtractionFormatError(ValueError):
    """The model's output could not be used. `summary` is safe to log, to store and to show the
    model on the retry: it names fields and error kinds, never the values that were received."""

    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def parse_extraction(raw: str) -> JobExtraction:
    """Parse the model's text into a JobExtraction or raise ExtractionFormatError."""
    text = raw.strip()
    if text.startswith("```"):  # tolerate a markdown fence around the JSON
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return JobExtraction.model_validate_json(text)
    except ValidationError as error:
        problems = sorted(
            {
                f"{'.'.join(str(part) for part in item['loc']) or 'output'}: {item['type']}"
                for item in error.errors(include_input=False, include_url=False)
            }
        )
        raise ExtractionFormatError("; ".join(problems[:6])[:300]) from None
