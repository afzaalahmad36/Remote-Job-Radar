import enum
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum


class RemotePolicy(enum.StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class ExtractionStatus(enum.StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    EXTRACTION_FAILED = "extraction_failed"


class Job(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "jobs"
    # The same post must never be stored twice for one source. This also creates an index
    # starting with source_id, which serves the foreign key, so no separate index is needed.
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_jobs_source_id_external_id"),
        # Listing the newest jobs first is the main query.
        Index("ix_jobs_created_at", "created_at"),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(String(255))
    raw_text: Mapped[str] = mapped_column(Text)

    # Filled in by LLM extraction, so everything below starts empty (NULL).
    title: Mapped[str | None] = mapped_column(String(500), default=None)
    company: Mapped[str | None] = mapped_column(String(255), default=None)
    stack: Mapped[list[str]] = mapped_column(ARRAY(String(100)), default=list, server_default="{}")
    seniority: Mapped[str | None] = mapped_column(String(50), default=None)
    salary_min: Mapped[int | None] = mapped_column(default=None)
    salary_max: Mapped[int | None] = mapped_column(default=None)
    salary_currency: Mapped[str | None] = mapped_column(String(3), default=None)
    remote_policy: Mapped[RemotePolicy | None] = mapped_column(
        pg_enum(RemotePolicy, "remote_policy"), default=None
    )
    apply_email: Mapped[str | None] = mapped_column(String(320), default=None)
    apply_url: Mapped[str | None] = mapped_column(Text, default=None)
    posted_at: Mapped[datetime | None] = mapped_column(default=None)

    extraction_status: Mapped[ExtractionStatus] = mapped_column(
        pg_enum(ExtractionStatus, "extraction_status"),
        default=ExtractionStatus.PENDING,
        server_default=ExtractionStatus.PENDING.value,
    )
