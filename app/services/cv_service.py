"""Storing, reading and deleting a user's CV.

A CV is personal data, so this module (and everything under app/cv/) never imports an LLM client
or an HTTP library: a test checks it. Nothing here logs the text, and every error is a fixed
phrase, never a message taken from the file or from a parser.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    CvBusyError,
    CvUnreadableError,
    CvUnsupportedTypeError,
    NotFoundError,
)
from app.cv.pdf_text import PdfReadFailure, read_pdf_text
from app.cv.skills import find_skills
from app.cv.suggestions import Suggestions, suggest
from app.cv.text import clean_cv_text, mask_cv_text
from app.models import Cv, CvFormat

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
ALLOWED_CONTENT_TYPES = frozenset({"application/pdf", "text/plain", "application/octet-stream"})


@dataclass(frozen=True)
class CvLimits:
    """The caps on one CV (from settings). The byte cap is enforced while the body is read."""

    max_pages: int
    max_chars: int
    pdf_timeout_seconds: float
    pdf_memory_limit_mb: int


# What the user is told when a PDF cannot be read: fixed phrases only.
_PDF_FAILURES = {
    "encrypted": "The PDF is password protected.",
    "too_many_pages": "The PDF has too many pages.",
    "too_long": "The CV text is too long.",
    "timeout": "Reading the PDF took too long.",
    "unreadable": "The PDF could not be read.",
}


async def save_cv(
    session: AsyncSession,
    user_id: uuid.UUID,
    data: bytes,
    content_type: str,
    limits: CvLimits,
) -> Cv:
    """Store `data` (a PDF or UTF-8 text) as the user's CV, replacing any earlier one.

    Raises CvUnsupportedTypeError, CvUnreadableError or CvBusyError. Only the cleaned, masked
    text is stored; the uploaded bytes are never written anywhere.
    """
    started = time.monotonic()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise CvUnsupportedTypeError("Upload a PDF or plain text (UTF-8).")
    pages: int | None = None
    if data.startswith(PDF_MAGIC):  # the content decides, not the header or a file name
        source_format = CvFormat.PDF
        pages, raw = await _pdf_text(data, limits)
    else:
        if content_type == "application/pdf":
            raise CvUnsupportedTypeError("The file is not a PDF.")
        source_format = CvFormat.TEXT
        raw = _decode_text(data)

    text = clean_cv_text(raw)
    if not text:
        raise CvUnreadableError(
            "No text was found in the CV. Scanned PDFs (images of text) are not supported."
        )
    if len(text) > limits.max_chars:
        raise CvUnreadableError(f"The CV text is too long (limit {limits.max_chars} characters).")
    text = mask_cv_text(text)

    cv = await _upsert(session, user_id, text, source_format, pages)
    # Counts and timing only: never the text.
    logger.info(
        "CV stored: user=%s format=%s pages=%s chars=%d ms=%d",
        user_id,
        source_format.value,
        pages,
        len(text),
        (time.monotonic() - started) * 1000,
    )
    return cv


async def get_cv(session: AsyncSession, user_id: uuid.UUID) -> Cv:
    """The user's CV, or NotFoundError. There is no CV id in the API: a user only ever reaches
    their own."""
    cv = await session.scalar(select(Cv).where(Cv.user_id == user_id))
    if cv is None:
        raise NotFoundError("No CV uploaded")
    return cv


async def delete_cv(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Delete the user's CV for good (the row, and so the text). NotFoundError if there is none."""
    result = await session.execute(delete(Cv).where(Cv.user_id == user_id).returning(Cv.id))
    if result.first() is None:
        raise NotFoundError("No CV uploaded")
    await session.commit()
    logger.info("CV deleted: user=%s", user_id)


@dataclass(frozen=True)
class SkillsView:
    """The stored skills with the suggestions read from the CV text (not stored: cheap to redo)."""

    skills: list[str]
    edited: bool
    suggestions: Suggestions
    updated_at: datetime


def _skills_view(cv: Cv) -> SkillsView:
    return SkillsView(list(cv.skills), cv.skills_edited, suggest(cv.text), cv.updated_at)


async def get_skills(session: AsyncSession, user_id: uuid.UUID) -> SkillsView:
    """The user's skills and suggestions, or NotFoundError when there is no CV."""
    return _skills_view(await get_cv(session, user_id))


async def save_skills(session: AsyncSession, user_id: uuid.UUID, skills: list[str]) -> SkillsView:
    """Replace the user's skills with their own list (already normalised by the schema).

    One UPDATE, no read first, so it cannot race with an upload. NotFoundError if there is no CV.
    """
    statement = (
        update(Cv)
        .where(Cv.user_id == user_id)
        .values(skills=skills, skills_edited=True, updated_at=func.now())
        .returning(Cv)
        .execution_options(populate_existing=True)
    )
    cv = await session.scalar(statement)
    if cv is None:
        raise NotFoundError("No CV uploaded")
    await session.commit()
    logger.info("CV skills saved: user=%s count=%d", user_id, len(skills))  # never the names
    return _skills_view(cv)


def _decode_text(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")  # tolerates the BOM some editors add
    except UnicodeDecodeError:
        raise CvUnsupportedTypeError("Upload a PDF or plain text (UTF-8).") from None
    if "\x00" in text:  # a binary file that happens to decode
        raise CvUnsupportedTypeError("Upload a PDF or plain text (UTF-8).")
    return text


async def _pdf_text(data: bytes, limits: CvLimits) -> tuple[int, str]:
    try:
        result = await read_pdf_text(
            data,
            max_pages=limits.max_pages,
            max_chars=limits.max_chars,
            timeout_seconds=limits.pdf_timeout_seconds,
            memory_limit_mb=limits.pdf_memory_limit_mb,
        )
    except PdfReadFailure as failure:
        if failure.code == "busy":
            raise CvBusyError("Too many CVs are being read right now. Try again shortly.") from None
        phrase = _PDF_FAILURES.get(failure.code, _PDF_FAILURES["unreadable"])
        if failure.code == "too_many_pages":
            phrase = f"The PDF has too many pages (limit {limits.max_pages})."
        raise CvUnreadableError(phrase) from None
    return result.pages, result.text


async def _upsert(
    session: AsyncSession,
    user_id: uuid.UUID,
    text: str,
    source_format: CvFormat,
    pages: int | None,
) -> Cv:
    """Insert the CV, or replace the user's existing one, in a single statement.

    `INSERT ... ON CONFLICT (user_id) DO UPDATE` is atomic, so two uploads at the same moment
    cannot both insert (the unique constraint on user_id would make one of them fail).
    """
    values = {
        "text": text,
        "source_format": source_format,
        "page_count": pages,
        "char_count": len(text),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        # A new CV is a new document: its skills are read from it again, and any list the user
        # edited for the previous one is replaced (the edited flag goes back to False).
        "skills": find_skills(text),
        "skills_edited": False,
    }
    statement = (
        insert(Cv)
        .values(user_id=user_id, **values)
        .on_conflict_do_update(
            index_elements=[Cv.user_id],
            # onupdate=... on the column does not apply to an upsert, so set updated_at here.
            set_={**values, "updated_at": func.now()},
        )
        .returning(Cv)
        .execution_options(populate_existing=True)
    )
    cv = await session.scalar(statement)
    await session.commit()
    assert cv is not None  # RETURNING always yields the row that was inserted or updated
    return cv
