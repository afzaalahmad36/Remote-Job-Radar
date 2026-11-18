"""create match_status

Revision ID: 5f2b8c41d7a3
Revises: 08d8e3481558
Create Date: 2026-11-18 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5f2b8c41d7a3"
down_revision: str | Sequence[str] | None = "08d8e3481558"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "match_status",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("requested_seq", sa.BigInteger(), server_default="1", nullable=False),
        sa.Column("finished_seq", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("running_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fingerprint", sa.String(length=64), nullable=True),
        sa.Column("failure_count", sa.SmallInteger(), server_default="0", nullable=False),
        sa.Column("last_error", sa.String(length=30), nullable=True),
        sa.Column("last_candidates", sa.Integer(), nullable=True),
        sa.Column("last_written", sa.Integer(), nullable=True),
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
            "failure_count >= 0", name=op.f("ck_match_status_failures_not_negative")
        ),
        sa.CheckConstraint(
            "requested_seq >= 0 AND finished_seq >= 0",
            name=op.f("ck_match_status_seq_not_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_match_status_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_match_status")),
    )
    op.create_index(
        "ix_match_status_pending",
        "match_status",
        ["requested_at"],
        unique=False,
        postgresql_where="requested_seq > finished_seq",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_match_status_pending",
        table_name="match_status",
        postgresql_where="requested_seq > finished_seq",
    )
    op.drop_table("match_status")
