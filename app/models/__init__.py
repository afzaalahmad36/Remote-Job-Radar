"""Import every model here so `Base.metadata` knows all tables (Alembic relies on this)."""

from app.models.job import ExtractionStatus, Job, RemotePolicy
from app.models.refresh_token import RefreshToken
from app.models.source import Source, SourceType
from app.models.user import User

__all__ = [
    "ExtractionStatus",
    "Job",
    "RefreshToken",
    "RemotePolicy",
    "Source",
    "SourceType",
    "User",
]
