"""add embeddings (pgvector) to jobs and cvs

Revision ID: c3d1a7e95b20
Revises: 11878e9cb8aa
Create Date: 2026-11-16 09:12:41.220518

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "c3d1a7e95b20"
down_revision: str | Sequence[str] | None = "11878e9cb8aa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = 384  # bge-small; a model with another length needs a new migration

# A vector comes with the model that made it, the hash of what was embedded and the time.
COMPLETE = (
    "(embedding IS NULL) = (embedding_model IS NULL) "
    "AND (embedding IS NULL) = (embedding_text_hash IS NULL) "
    "AND (embedding IS NULL) = (embedded_at IS NULL)"
)


def upgrade() -> None:
    """Upgrade schema."""
    # Creating the extension needs a superuser (pgvector is not a "trusted" extension): see
    # docs/DEPLOYMENT.md. IF NOT EXISTS: a database that already has it is fine.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for table in ("jobs", "cvs"):
        op.add_column(table, sa.Column("embedding", Vector(DIM), nullable=True))
        op.add_column(table, sa.Column("embedding_model", sa.String(length=100), nullable=True))
        op.add_column(table, sa.Column("embedding_text_hash", sa.String(length=64), nullable=True))
        op.add_column(table, sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))
        op.create_check_constraint(op.f(f"ck_{table}_embedding_complete"), table, COMPLETE)

    # The embedding sweep: extracted jobs that have no vector yet, oldest first. No vector index
    # (HNSW) yet: with a few hundred jobs an exact scan is fast, and the choice belongs to the
    # PR that adds the similarity query.
    op.create_index(
        "ix_jobs_embedding_pending",
        "jobs",
        ["created_at"],
        postgresql_where=sa.text("embedding IS NULL AND extraction_status = 'extracted'"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_jobs_embedding_pending",
        table_name="jobs",
        postgresql_where=sa.text("embedding IS NULL AND extraction_status = 'extracted'"),
    )
    for table in ("cvs", "jobs"):
        op.drop_constraint(op.f(f"ck_{table}_embedding_complete"), table, type_="check")
        op.drop_column(table, "embedded_at")
        op.drop_column(table, "embedding_text_hash")
        op.drop_column(table, "embedding_model")
        op.drop_column(table, "embedding")
    # This migration created the extension, nothing else here uses it, and a downgrade to "base"
    # should leave the database as it found it.
    op.execute("DROP EXTENSION IF EXISTS vector")
