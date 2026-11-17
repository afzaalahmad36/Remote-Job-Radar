"""create job_matches

Revision ID: 08d8e3481558
Revises: c3d1a7e95b20
Create Date: 2026-11-17 11:35:10.475397

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "08d8e3481558"
down_revision: str | Sequence[str] | None = "c3d1a7e95b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "job_matches",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("score", sa.SmallInteger(), nullable=False),
        sa.Column("filters_failed", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "missing_skills",
            sa.ARRAY(sa.String(length=100)),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("scorer_version", sa.String(length=30), nullable=False),
        sa.Column("inputs_hash", sa.String(length=64), nullable=False),
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
            "cardinality(missing_skills) <= 30", name=op.f("ck_job_matches_missing_cap")
        ),
        sa.CheckConstraint("score BETWEEN 0 AND 100", name=op.f("ck_job_matches_score_range")),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_job_matches_job_id_jobs"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_job_matches_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_matches")),
        sa.UniqueConstraint("user_id", "job_id", name="uq_job_matches_user_job"),
    )
    op.create_index("ix_job_matches_job_id", "job_matches", ["job_id"], unique=False)
    op.create_index("ix_job_matches_user_score", "job_matches", ["user_id", "score"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_job_matches_user_score", table_name="job_matches")
    op.drop_index("ix_job_matches_job_id", table_name="job_matches")
    op.drop_table("job_matches")
