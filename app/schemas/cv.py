import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.cv.skills import MAX_SKILLS, SkillError, normalize_skills
from app.models import Seniority
from app.models.cv import CvFormat


class CvRead(BaseModel):
    """The caller's own CV. `text` is what was stored (emails, phones and links masked), so the
    user can check exactly what the system holds."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_format: CvFormat
    page_count: int | None
    char_count: int
    text: str
    created_at: datetime
    updated_at: datetime


class YearsSuggestionOut(BaseModel):
    value: int
    match: str  # the kind of match, e.g. "years phrase"


class SenioritySuggestionOut(BaseModel):
    value: Seniority
    match: str  # the kind of match, e.g. "title keyword in the current role"


class CvSuggestionsOut(BaseModel):
    """Guesses read from the CV text. Shown to the user only: they never affect scoring."""

    years_of_experience: YearsSuggestionOut | None
    seniority: SenioritySuggestionOut | None


class CvSkillsRead(BaseModel):
    skills: list[str]
    edited: bool  # False: found in the CV text; True: the user saved their own list
    suggestions: CvSuggestionsOut
    updated_at: datetime


class CvSkillsWrite(BaseModel):
    """The user's whole skills list (it replaces the stored one)."""

    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(max_length=MAX_SKILLS * 2)  # blanks and duplicates are dropped later

    @field_validator("skills")
    @classmethod
    def _clean(cls, skills: list[str]) -> list[str]:
        try:
            return normalize_skills(skills)
        except SkillError as error:
            raise ValueError(str(error)) from None  # a fixed phrase, never the submitted value
