"""create cvs

Revision ID: f29a32b4831a
Revises: 3088b4fb6733
Create Date: 2026-11-13 11:28:36.693859

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f29a32b4831a"
down_revision: str | Sequence[str] | None = "3088b4fb6733"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "cvs",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_format", sa.Enum("pdf", "text", name="cv_format"), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_cvs_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cvs")),
        sa.UniqueConstraint("user_id", name=op.f("uq_cvs_user_id")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("cvs")
    # drop_table leaves the enum type behind (Postgres keeps types separately).
    op.execute("DROP TYPE cv_format")
