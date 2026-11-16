"""add cv skills and job preferences

Revision ID: 11878e9cb8aa
Revises: f29a32b4831a
Create Date: 2026-11-13 17:25:37.818902

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "11878e9cb8aa"
down_revision: str | Sequence[str] | None = "f29a32b4831a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_enum(*values: str, name: str) -> postgresql.ENUM:
    """A reference to an enum type that `jobs` already created: create_type=False, so this
    migration never tries to CREATE it again (it would fail with "type already exists")."""
    return postgresql.ENUM(*values, name=name, create_type=False)


def upgrade() -> None:
    """Upgrade schema."""
    # Skills of a CV: a list of names, empty for CVs stored before this migration (the user can
    # save their own list, or upload the CV again to have it read).
    op.add_column(
        "cvs",
        sa.Column("skills", sa.ARRAY(sa.String(length=50)), server_default="{}", nullable=False),
    )
    op.add_column(
        "cvs",
        sa.Column("skills_edited", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_check_constraint(op.f("ck_cvs_skills_max_100"), "cvs", "cardinality(skills) <= 100")

    op.create_table(
        "job_preferences",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "seniority_levels",
            sa.ARRAY(
                _existing_enum(
                    "intern", "junior", "mid", "senior", "lead", "principal", name="seniority"
                )
            ),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "remote_policies",
            sa.ARRAY(_existing_enum("remote", "hybrid", "onsite", name="remote_policy")),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("min_salary", sa.Integer(), nullable=True),
        sa.Column("salary_currency", sa.String(length=3), nullable=True),
        sa.Column(
            "salary_period",
            _existing_enum("year", "month", "week", "day", "hour", name="salary_period"),
            nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(min_salary IS NULL) = (salary_currency IS NULL) "
            "AND (min_salary IS NULL) = (salary_period IS NULL)",
            name=op.f("ck_job_preferences_salary_all_or_nothing"),
        ),
        sa.CheckConstraint(
            "min_salary IS NULL OR min_salary > 0",
            name=op.f("ck_job_preferences_min_salary_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_job_preferences_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_preferences")),
        sa.UniqueConstraint("user_id", name=op.f("uq_job_preferences_user_id")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The enum types belong to `jobs`, so they are not dropped here.
    op.drop_table("job_preferences")
    op.drop_constraint(op.f("ck_cvs_skills_max_100"), "cvs", type_="check")
    op.drop_column("cvs", "skills_edited")
    op.drop_column("cvs", "skills")
