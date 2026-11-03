import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.job import ExtractionStatus, RemotePolicy
from app.services.extraction_retry import RetryStatus


class ExtractAcceptedOut(BaseModel):
    job_id: uuid.UUID
    status: RetryStatus  # "queued", or "already_running" (a run is in progress; not an error)


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_id: uuid.UUID
    external_id: str
    # Plain text of the post. Never HTML, but it is untrusted content from the internet.
    raw_text: str
    posted_at: datetime | None
    extraction_status: ExtractionStatus
    created_at: datetime
    # Filled in by LLM extraction (Phase 3); null until then.
    title: str | None
    company: str | None
    stack: list[str]
    seniority: str | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    remote_policy: RemotePolicy | None
    apply_email: str | None
    apply_url: str | None
