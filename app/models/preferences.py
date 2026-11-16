import uuid

from sqlalchemy import ARRAY, CheckConstraint, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum
from app.models.job import RemotePolicy, SalaryPeriod, Seniority


class JobPreferences(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """What a user wants from a job. One row per user, written only when they save it.

    An empty list means "no preference" (any level, any remote policy). The notes are free text
    the user wrote about themselves, so like the CV they are personal data: never logged, never
    sent to a provider, hidden from `repr`.
    """

    __tablename__ = "job_preferences"
    __table_args__ = (
        CheckConstraint("min_salary IS NULL OR min_salary > 0", name="min_salary_positive"),
        # An amount without its currency and period means nothing ("90000 of what, per what?").
        CheckConstraint(
            "(min_salary IS NULL) = (salary_currency IS NULL) "
            "AND (min_salary IS NULL) = (salary_period IS NULL)",
            name="salary_all_or_nothing",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    seniority_levels: Mapped[list[Seniority]] = mapped_column(
        ARRAY(pg_enum(Seniority, "seniority", create_type=False)),
        default=list,
        server_default="{}",
    )
    remote_policies: Mapped[list[RemotePolicy]] = mapped_column(
        ARRAY(pg_enum(RemotePolicy, "remote_policy", create_type=False)),
        default=list,
        server_default="{}",
    )
    min_salary: Mapped[int | None] = mapped_column(default=None)
    salary_currency: Mapped[str | None] = mapped_column(String(3), default=None)
    salary_period: Mapped[SalaryPeriod | None] = mapped_column(
        pg_enum(SalaryPeriod, "salary_period", create_type=False), default=None
    )
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    def __repr__(self) -> str:
        # The notes are the user's own words: say only what is safe to print.
        return f"<JobPreferences id={self.id} user={self.user_id}>"
