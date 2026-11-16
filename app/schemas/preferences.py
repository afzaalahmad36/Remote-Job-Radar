import uuid
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models import RemotePolicy, SalaryPeriod, Seniority

MAX_NOTES_CHARS = 2_000
MAX_MIN_SALARY = 1_000_000_000  # fits the integer column, and no real salary is above it


class PreferencesWrite(BaseModel):
    """Everything the user wants, as one document: saving replaces what was stored.

    A field left out means "no preference" for it. An empty list means any level / any policy.
    """

    model_config = ConfigDict(extra="forbid")

    seniority_levels: list[Seniority] = Field(default_factory=list, max_length=len(Seniority))
    remote_policies: list[RemotePolicy] = Field(default_factory=list, max_length=len(RemotePolicy))
    min_salary: int | None = Field(default=None, gt=0, le=MAX_MIN_SALARY)
    salary_currency: str | None = None  # ISO 4217 code such as "USD"
    salary_period: SalaryPeriod | None = None
    notes: str | None = Field(default=None, max_length=MAX_NOTES_CHARS)

    @field_validator("seniority_levels", "remote_policies")
    @classmethod
    def _no_duplicates(cls, values: list[Seniority] | list[RemotePolicy]) -> list[object]:
        # dict.fromkeys drops repeats and keeps the order the user chose
        return list(dict.fromkeys(values))

    @field_validator("salary_currency")
    @classmethod
    def _currency_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        code = value.strip().upper()
        if len(code) != 3 or not (code.isascii() and code.isalpha()):
            raise ValueError("The currency must be a 3-letter code such as USD.")
        return code

    @field_validator("notes")
    @classmethod
    def _blank_notes(cls, value: str | None) -> str | None:
        return value if value and value.strip() else None

    @model_validator(mode="after")
    def _salary_is_complete(self) -> Self:
        parts = (self.min_salary, self.salary_currency, self.salary_period)
        if any(part is not None for part in parts) and any(part is None for part in parts):
            raise ValueError("A minimum salary needs an amount, a currency and a period together.")
        return self


class PreferencesRead(BaseModel):
    """The caller's saved preferences, or the empty defaults before anything was saved."""

    model_config = ConfigDict(from_attributes=True)

    seniority_levels: list[Seniority]
    remote_policies: list[RemotePolicy]
    min_salary: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    notes: str | None
    id: uuid.UUID | None = None
    updated_at: datetime | None = None  # None until the user saves
