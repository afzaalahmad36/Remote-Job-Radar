"""Import every model here so `Base.metadata` knows all tables (Alembic relies on this)."""

from app.models.job import ExtractionStatus, Job, RemotePolicy
from app.models.source import Source, SourceType
from app.models.user import User

__all__ = ["ExtractionStatus", "Job", "RemotePolicy", "Source", "SourceType", "User"]
