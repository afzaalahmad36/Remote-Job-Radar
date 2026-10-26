import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Parent of every model. Its `metadata` collects all tables, which Alembic reads.

    SQLAlchemy 2.0 style: columns are declared as `name: Mapped[type] = mapped_column(...)`.
    `Mapped[...]` is a normal type hint that mypy understands *and* tells SQLAlchemy the column
    type (`Mapped[str | None]` means a nullable column).
    """

    # Predictable constraint names, so migrations can drop or alter them by name later.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    # Python type -> column type, so `Mapped[datetime]` is always timezone-aware.
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
    }


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key (generated in Python, so no database extension is needed)."""

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """Adds created_at / updated_at, set by the database clock (timestamptz, stored as UTC)."""

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


def pg_enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    """A native Postgres enum that stores the enum *values* ('rss'), not the names ('RSS')."""
    return Enum(enum_cls, name=name, values_callable=lambda e: [member.value for member in e])
