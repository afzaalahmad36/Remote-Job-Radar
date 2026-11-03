import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum


class RemotePolicy(enum.StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class Seniority(enum.StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class SalaryPeriod(enum.StrEnum):
    """What the salary amounts of a post are per, exactly as the post states it."""

    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"


class ExtractionStatus(enum.StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    EXTRACTION_FAILED = "extraction_failed"
    NOT_A_JOB = "not_a_job"  # the model said the text is not a job opening


class ExtractionMethod(enum.StrEnum):
    """How the structured fields were produced. NULL on a job means the rules have not run yet."""

    RULES = "rules"
    RULES_LLM = "rules_llm"


class LlmEnrichment(enum.StrEnum):
    """Whether the optional model pass is needed, waiting, finished or given up."""

    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"


class Job(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "jobs"
    # The same post must never be stored twice for one source. This also creates an index
    # starting with source_id, which serves the foreign key, so no separate index is needed.
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_jobs_source_id_external_id"),
        # Listing the newest jobs first is the main query.
        Index("ix_jobs_created_at", "created_at"),
        # The same post text is found again by its hash, to reuse an extraction.
        Index("ix_jobs_raw_text_hash", "raw_text_hash"),
        # Sweep 1: jobs whose rules phase never ran (rows from before extraction existed, and
        # tasks lost with their Redis message). Tiny, because the rules phase empties it.
        Index(
            "ix_jobs_rules_not_run",
            "created_at",
            postgresql_where=text("extraction_method IS NULL"),
        ),
        # Sweep 2: jobs waiting for the model, oldest first (also used to give up on old ones).
        Index(
            "ix_jobs_enrichment_pending",
            "created_at",
            postgresql_where=text("llm_enrichment = 'pending'"),
        ),
        # A job can only be "extracted" if it has a title: "usable" means "has a title".
        CheckConstraint(
            "extraction_status <> 'extracted' OR title IS NOT NULL",
            name="extracted_has_title",
        ),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(String(255))
    raw_text: Mapped[str] = mapped_column(Text)

    # Filled in by LLM extraction, so everything below starts empty (NULL).
    title: Mapped[str | None] = mapped_column(String(500), default=None)
    company: Mapped[str | None] = mapped_column(String(255), default=None)
    stack: Mapped[list[str]] = mapped_column(ARRAY(String(100)), default=list, server_default="{}")
    seniority: Mapped[Seniority | None] = mapped_column(
        pg_enum(Seniority, "seniority"), default=None
    )
    salary_min: Mapped[int | None] = mapped_column(default=None)
    salary_max: Mapped[int | None] = mapped_column(default=None)
    salary_currency: Mapped[str | None] = mapped_column(String(3), default=None)
    # What the amounts are per. NULL when the post gives amounts without a period.
    salary_period: Mapped[SalaryPeriod | None] = mapped_column(
        pg_enum(SalaryPeriod, "salary_period"), default=None
    )
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

    # --- extraction bookkeeping (the service that writes these is not built yet) ---------------
    # SHA-256 of the normalized post text (app/extraction/text_hash.py); NULL until first run.
    raw_text_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    # NULL = the rules phase has not run. "rules" = rules only, "rules_llm" = merged with a model.
    extraction_method: Mapped[ExtractionMethod | None] = mapped_column(
        pg_enum(ExtractionMethod, "extraction_method"), default=None
    )
    llm_enrichment: Mapped[LlmEnrichment] = mapped_column(
        pg_enum(LlmEnrichment, "llm_enrichment"),
        default=LlmEnrichment.NOT_NEEDED,
        server_default=LlmEnrichment.NOT_NEEDED.value,
    )
    extraction_rules_version: Mapped[str | None] = mapped_column(String(50), default=None)
    extraction_prompt_version: Mapped[str | None] = mapped_column(String(50), default=None)
    extraction_model: Mapped[str | None] = mapped_column(String(100), default=None)  # NULL: rules
    extracted_at: Mapped[datetime | None] = mapped_column(default=None)
    # Short, sanitized reason for a failure or a skip. Never post text, never a provider body.
    extraction_error: Mapped[str | None] = mapped_column(String(500), default=None)
    # extraction_attempts counts every service run (sweeps and Celery redeliveries included);
    # extraction_retries counts only user retries. Separate on purpose: the retry limit must not
    # be used up by the system's own re-runs.
    extraction_attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    extraction_retries: Mapped[int] = mapped_column(default=0, server_default="0")
