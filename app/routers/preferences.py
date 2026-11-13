from fastapi import APIRouter

from app.api_deps import CurrentUserDep, SessionDep
from app.schemas.preferences import PreferencesRead, PreferencesWrite
from app.services import preferences_service

router = APIRouter(prefix="/preferences", tags=["preferences"])


def _empty() -> PreferencesRead:
    return PreferencesRead(
        seniority_levels=[],
        remote_policies=[],
        min_salary=None,
        salary_currency=None,
        salary_period=None,
        notes=None,
    )


@router.get("")
async def get_preferences(user: CurrentUserDep, session: SessionDep) -> PreferencesRead:
    """Your saved preferences, or empty ones (`updated_at` is null) if you never saved any."""
    saved = await preferences_service.get_preferences(session, user.id)
    return PreferencesRead.model_validate(saved) if saved else _empty()


@router.put("")
async def put_preferences(
    body: PreferencesWrite, user: CurrentUserDep, session: SessionDep
) -> PreferencesRead:
    """Save your preferences. The body replaces what was stored: a field left out is cleared."""
    return PreferencesRead.model_validate(
        await preferences_service.save_preferences(session, user.id, body)
    )
