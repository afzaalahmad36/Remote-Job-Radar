"""The explanation stored with every match (`job_matches.reasons`).

It is a Pydantic model so its shape is a contract, not whatever a dict happened to hold:
`extra="forbid"` means an unexpected key is an error, and every text field is either a name taken
from the job itself (a stack entry), an enum value, or one of the fixed codes below. No CV text and
no preference notes can get in. Stored as JSON with `exclude_none=True`, so a part that does not
apply (a filter reason, a component that was not computed) is absent instead of `null`.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FilterName = Literal["remote_policy", "seniority", "salary"]
FilterOutcome = Literal["pass", "fail", "unknown"]
# Why a filter could not decide. A code, never a sentence: the API or UI words it.
FilterReason = Literal[
    "no_preference",  # the user set nothing for this filter
    "job_value_missing",  # the post does not say
    "currency_differs",  # amounts in different currencies are never converted
    "salary_period_missing",  # an amount without "per year/month/..." cannot be compared
    "open_ended_below_minimum",  # "from X" with X under the minimum: it may still pay enough
]
# Facts about how the score was made. `NOTE_MESSAGES` has the sentence for each (PR 5 shows them).
Note = Literal[
    "filters_failed",
    "no_job_stack",
    "no_cv_skills",
    "no_embedding",
    "not_english",
    "skills_truncated",
]
NOTE_MESSAGES: dict[Note, str] = {
    "filters_failed": "A hard filter failed, so the score is capped.",
    "no_job_stack": "The job lists no technologies: the score is the embedding similarity only.",
    "no_cv_skills": "The CV has no skills listed: skills were not scored.",
    "no_embedding": "No comparable embedding: the score is skills only.",
    "not_english": "The post does not look English: the embedding was not used.",
    "skills_truncated": "Some skill names were left out of the lists (too many or too long).",
}


class _Strict(BaseModel):
    """Immutable, and rejects keys it does not declare."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WeightPair(_Strict):
    skills: float
    embedding: float


class Weights(_Strict):
    configured: WeightPair
    effective: WeightPair  # differs from `configured` when a component could not be computed


class SkillsComponent(_Strict):
    coverage: float
    matched: int
    total: int
    cv_skills: int  # how many skills the CV has, never which


class EmbeddingComponent(_Strict):
    cosine: float
    floor: float
    ceiling: float
    normalised: float


class Components(_Strict):
    skills: SkillsComponent | None = None
    embedding: EmbeddingComponent | None = None


class SkillLists(_Strict):
    """The job's own stack entries, split by whether the CV has them."""

    matched: list[str]
    missing: list[str]


class SalaryOfJob(_Strict):
    min: int | None = None
    max: int | None = None
    currency: str | None = None
    period: str | None = None


class SalaryWanted(_Strict):
    min: int
    currency: str
    period: str


class FilterResult(_Strict):
    name: FilterName
    outcome: FilterOutcome
    job: str | SalaryOfJob | None = None
    preference: list[str] | SalaryWanted | None = None
    job_per_year: int | None = None
    preference_per_year: int | None = None
    reason: FilterReason | None = None


class MatchReasons(_Strict):
    scorer_version: str
    score: int
    cap_applied: bool
    weights: Weights
    components: Components
    skills: SkillLists
    filters: list[FilterResult]
    notes: list[Note]

    def to_json(self) -> dict[str, Any]:
        """The JSON-ready dict that is stored (parts that do not apply are left out)."""
        return self.model_dump(mode="json", exclude_none=True)
