"""Import every model here so `Base.metadata` knows all tables (Alembic relies on this)."""

from app.models.cv import Cv, CvFormat
from app.models.job import (
    ExtractionMethod,
    ExtractionStatus,
    Job,
    LlmEnrichment,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
)
from app.models.llm_usage import LlmUsageEntry
from app.models.refresh_token import RefreshToken
from app.models.source import Source, SourceType
from app.models.user import User

__all__ = [
    "Cv",
    "CvFormat",
    "ExtractionMethod",
    "ExtractionStatus",
    "Job",
    "LlmEnrichment",
    "LlmUsageEntry",
    "RefreshToken",
    "RemotePolicy",
    "SalaryPeriod",
    "Seniority",
    "Source",
    "SourceType",
    "User",
]
