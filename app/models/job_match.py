import uuid
from typing import Any

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

# Bounds of the stored skill names (the scorer applies them, a check constraint backs the size).
MAX_SKILL_NAME_CHARS = 100  # the width of the column, equal to the width of `jobs.stack` entries
MAX_MISSING_SKILLS = 30


class JobMatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """How well one job fits one user's CV and preferences: a 0-100 score and why.

    One row per (user, job). The score is computed without any language model from the CV's stored
    skills and vector, the user's job preferences and the job's extracted fields (docs/MATCHING.md).
    `inputs_hash` identifies exactly what the score read, so a row is rewritten only when one of
    those inputs, the weights or the scorer version changed.

    `reasons` holds the explanation (matched and missing skills, each filter with the values it
    compared, the components and the weights). It never holds CV or preference text: the skill names
    in it are the job's own stack entries.
    """

    __tablename__ = "job_matches"
    __table_args__ = (
        UniqueConstraint("user_id", "job_id", name="uq_job_matches_user_job"),
        CheckConstraint("score BETWEEN 0 AND 100", name="score_range"),
        CheckConstraint(f"cardinality(missing_skills) <= {MAX_MISSING_SKILLS}", name="missing_cap"),
        # The unique constraint already serves "this user's matches"; this one serves the ranking
        # (ORDER BY score DESC is a backward scan of it).
        Index("ix_job_matches_user_score", "user_id", "score"),
        Index("ix_job_matches_job_id", "job_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    score: Mapped[int] = mapped_column(SmallInteger)  # 0-100
    # True when a hard filter failed (the score is then capped); lists hide these by default.
    filters_failed: Mapped[bool] = mapped_column(default=False, server_default="false")
    missing_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(MAX_SKILL_NAME_CHARS)), default=list, server_default="{}"
    )
    reasons: Mapped[dict[str, Any]]
    scorer_version: Mapped[str] = mapped_column(String(30))
    inputs_hash: Mapped[str] = mapped_column(String(64))

    def __repr__(self) -> str:
        # Say only what is safe to print: ids and the number, never the explanation.
        return f"<JobMatch user={self.user_id} job={self.job_id} score={self.score}>"
