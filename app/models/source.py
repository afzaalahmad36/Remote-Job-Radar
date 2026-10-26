import enum
from datetime import datetime
from typing import Any

from sqlalchemy import String, true
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum


class SourceType(enum.StrEnum):
    TELEGRAM = "telegram"
    RSS = "rss"


class Source(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "sources"

    type: Mapped[SourceType] = mapped_column(pg_enum(SourceType, "source_type"))
    name: Mapped[str] = mapped_column(String(255))
    # Source-specific settings (feed URL, channel name, ...); the shape depends on `type`.
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default="{}")
    is_active: Mapped[bool] = mapped_column(server_default=true())
    last_synced_at: Mapped[datetime | None] = mapped_column(default=None)
