import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class LlmUsageEntry(UUIDPrimaryKeyMixin, Base):
    """One row per request made to an LLM provider, so budgets are counted, not guessed.

    Holds no post text, no response text and no key: only who, what kind and how big.
    """

    __tablename__ = "llm_usage"
    __table_args__ = (
        # Daily and per-minute caps count a user's rows in a time window. Starting with user_id,
        # this index also serves the user_id foreign key, so no separate one is needed.
        Index("ix_llm_usage_user_id_created_at", "user_id", "created_at"),
        # The global daily cap counts all rows in a window.
        Index("ix_llm_usage_created_at", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # SET NULL: deleting a job keeps the usage row, because the request was still made and counts.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), index=True, default=None
    )
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(50))
    # A short code such as "ok" or "rate_limited". Plain text, not an enum: the set of codes is
    # decided by the service that writes it and should not need a migration to grow.
    outcome: Mapped[str] = mapped_column(String(50))
    # NULL when the provider did not report a count.
    input_tokens: Mapped[int | None] = mapped_column(default=None)
    output_tokens: Mapped[int | None] = mapped_column(default=None)
    thought_tokens: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
