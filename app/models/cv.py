import enum
import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, pg_enum


class CvFormat(enum.StrEnum):
    """What the user uploaded. Only the extracted text is kept, never the file itself."""

    PDF = "pdf"
    TEXT = "text"


class Cv(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A user's CV as text. One per user; uploading again replaces it.

    Personal data. The text is stored after emails, phone numbers and links were masked, the
    original file is discarded, and nothing in this table ever goes to a hosted provider.
    """

    __tablename__ = "cvs"

    # unique=True: at most one CV per user, enforced by the database (so two simultaneous first
    # uploads cannot create two rows). Deleting the user deletes the CV.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    text: Mapped[str] = mapped_column(Text)
    source_format: Mapped[CvFormat] = mapped_column(pg_enum(CvFormat, "cv_format"))
    page_count: Mapped[int | None] = mapped_column(default=None)  # PDFs only
    char_count: Mapped[int]
    # SHA-256 of `text`: lets later steps tell "same CV again" from "changed CV" without
    # comparing the text.
    content_hash: Mapped[str] = mapped_column(String(64))

    def __repr__(self) -> str:
        # The default repr would print every column, and a debugger, a log line or a failed
        # assertion that shows an object would then show the CV. Say only what is safe.
        return f"<Cv id={self.id} format={self.source_format} chars={self.char_count}>"
