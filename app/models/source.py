import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, String, true
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum


class SourceType(enum.StrEnum):
    TELEGRAM = "telegram"
    RSS = "rss"


class Source(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "sources"

    # Each source belongs to one user; deleting the user deletes their sources (and their jobs).
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[SourceType] = mapped_column(pg_enum(SourceType, "source_type"))
    name: Mapped[str] = mapped_column(String(255))
    # Source-specific settings (feed URL, channel name, ...); the shape depends on `type`.
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default="{}")
    is_active: Mapped[bool] = mapped_column(server_default=true())
    last_synced_at: Mapped[datetime | None] = mapped_column(default=None)
    # Conditional GET validators the feed server sent last time (see ingestion/rss.py).
    etag: Mapped[str | None] = mapped_column(String(255), default=None)
    last_modified: Mapped[str | None] = mapped_column(String(100), default=None)
    # Short, sanitized reason the last sync failed; NULL after a successful one.
    last_error: Mapped[str | None] = mapped_column(String(500), default=None)
