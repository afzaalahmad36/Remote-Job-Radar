"""A user's job preferences: read, and save when the user asks.

Nothing is created behind the user's back: reading before the first save returns the empty
defaults without writing a row. The notes are personal data, so they are cleaned and masked like a
CV (emails, phones and links removed), never logged, and this module imports no LLM client or HTTP
library (a test checks it).
"""

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.cv.text import clean_cv_text, mask_cv_text
from app.models import JobPreferences
from app.schemas.preferences import PreferencesWrite

logger = logging.getLogger(__name__)


async def get_preferences(session: AsyncSession, user_id: uuid.UUID) -> JobPreferences | None:
    """The saved preferences, or None when the user never saved any."""
    result = await session.execute(select(JobPreferences).where(JobPreferences.user_id == user_id))
    return result.scalar_one_or_none()


def _clean_notes(notes: str | None) -> str | None:
    if notes is None:
        return None
    return mask_cv_text(clean_cv_text(notes)) or None


async def save_preferences(
    session: AsyncSession, user_id: uuid.UUID, data: PreferencesWrite
) -> JobPreferences:
    """Store the user's preferences, replacing earlier ones, in one INSERT ... ON CONFLICT."""
    values = {
        "seniority_levels": list(data.seniority_levels),
        "remote_policies": list(data.remote_policies),
        "min_salary": data.min_salary,
        "salary_currency": data.salary_currency,
        "salary_period": data.salary_period,
        "notes": _clean_notes(data.notes),
    }
    statement = (
        insert(JobPreferences)
        .values(user_id=user_id, **values)
        .on_conflict_do_update(
            index_elements=[JobPreferences.user_id],
            set_={**values, "updated_at": func.now()},
        )
        .returning(JobPreferences)
        .execution_options(populate_existing=True)
    )
    saved = await session.scalar(statement)
    await session.commit()
    assert saved is not None  # RETURNING always yields the inserted or updated row
    logger.info("Preferences saved: user=%s", user_id)  # never the notes
    return saved
