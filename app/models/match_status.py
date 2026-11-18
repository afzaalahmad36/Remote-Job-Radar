import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, SmallInteger, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class MatchStatus(Base):
    """Where one user's matches stand: is a (re)score requested, running, finished or failing.

    One row per user, holding counters, timestamps and fixed codes only: no CV text, no skills, no
    notes. A change that makes the matches stale (a new CV, edited skills or preferences, a job
    that was embedded or re-extracted) raises `requested_seq` in the same transaction as the change
    itself, so a lost Celery message or a Redis outage cannot lose the request. A run reads
    `requested_seq` when it starts and, when it finishes, sets `finished_seq` to that number. A
    change that arrives during the run raised `requested_seq` past it, so the user stays *pending*
    after the run ends. Counters are used instead of timestamps because two statements of one
    transaction (or of a test) can carry the same clock reading.

    pending = `requested_seq > finished_seq`. Redis only coalesces and delays messages; this row is
    the truth (the same stance as the extraction lock).
    """

    __tablename__ = "match_status"
    __table_args__ = (
        CheckConstraint("failure_count >= 0", name="failures_not_negative"),
        CheckConstraint("requested_seq >= 0 AND finished_seq >= 0", name="seq_not_negative"),
        # The sweep looks for users who are pending; this keeps that look cheap.
        Index(
            "ix_match_status_pending",
            "requested_at",
            postgresql_where="requested_seq > finished_seq",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    requested_seq: Mapped[int] = mapped_column(BigInteger, default=1, server_default="1")
    finished_seq: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    requested_at: Mapped[datetime] = mapped_column(server_default=func.clock_timestamp())
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    # Set when a run starts, cleared when it ends. A value older than the stale window is a run
    # that died, not a run in progress.
    running_since: Mapped[datetime | None] = mapped_column(default=None)
    # When the sweep last queued this user (at most once per 5 minutes per user).
    queued_at: Mapped[datetime | None] = mapped_column(default=None)
    # Scorer version and settings the last finished run used; a different one means a re-score.
    fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    # Failed runs in a row (never a retryable database or Redis error). 3 stops the sweep.
    failure_count: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(String(30), default=None)  # a fixed code
    last_candidates: Mapped[int | None] = mapped_column(default=None)
    last_written: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return (
            f"<MatchStatus user={self.user_id} requested={self.requested_seq} "
            f"finished={self.finished_seq}>"
        )
