import uuid
from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.core.url_safety import MAX_URL_LENGTH
from app.models.source import SourceType

SourceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class RssConfig(BaseModel):
    """What an RSS source's `config` must look like. Unknown keys are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")

    # Only the shape is checked here; whether the URL is safe to fetch is `core/url_safety.py`.
    url: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_URL_LENGTH)
    ]


# One config model per supported source type. Telegram is added here in a later step.
CONFIG_MODELS: dict[SourceType, type[BaseModel]] = {SourceType.RSS: RssConfig}


class SourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SourceType
    name: SourceName
    config: dict[str, Any] = Field(default_factory=dict)  # checked against `type` in the service


class SourceUpdate(BaseModel):
    """Partial update: send only the fields to change. `type` cannot be changed."""

    model_config = ConfigDict(extra="forbid")

    name: SourceName | None = None
    config: dict[str, Any] | None = None  # replaces the whole config
    is_active: bool | None = None

    @model_validator(mode="after")
    def _no_explicit_nulls(self) -> Self:
        # Omitting a field means "leave it alone"; sending null would mean "erase it", which
        # none of these columns allow.
        for field in self.model_fields_set:
            if getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: SourceType
    name: str
    config: dict[str, Any]
    is_active: bool
    last_synced_at: datetime | None
    last_error: str | None  # why the last sync failed (short, sanitized); null if it worked
    created_at: datetime
    updated_at: datetime
