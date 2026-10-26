from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.models import RefreshToken, User
from app.services import auth_service

EMAIL = "svc@example.com"
PASSWORD = "service-password"


async def test_refresh_token_expires_after_the_configured_days(
    db_session: AsyncSession, settings: Settings
) -> None:
    now = datetime.now(UTC)
    await auth_service.register_user(db_session, EMAIL, PASSWORD)
    pair = await auth_service.login(db_session, settings, EMAIL, PASSWORD, now=now)
    just_before = now + timedelta(days=settings.refresh_token_expire_days) - timedelta(minutes=1)
    just_after = now + timedelta(days=settings.refresh_token_expire_days, minutes=1)

    with pytest.raises(AuthenticationError):
        await auth_service.refresh_tokens(db_session, settings, pair.refresh_token, now=just_after)
    # Being expired is not "reuse", so the token is untouched and still valid before the deadline.
    await auth_service.refresh_tokens(db_session, settings, pair.refresh_token, now=just_before)


async def test_register_stores_a_lowercased_email(db_session: AsyncSession) -> None:
    user = await auth_service.register_user(db_session, "  Svc@Example.COM ", PASSWORD)

    assert user.email == "svc@example.com"


async def test_register_duplicate_raises_conflict_and_keeps_the_session_usable(
    db_session: AsyncSession,
) -> None:
    await auth_service.register_user(db_session, EMAIL, PASSWORD)

    with pytest.raises(ConflictError):
        await auth_service.register_user(db_session, EMAIL.upper(), PASSWORD)

    assert len((await db_session.scalars(select(User))).all()) == 1


async def test_each_login_starts_a_new_family(db_session: AsyncSession, settings: Settings) -> None:
    await auth_service.register_user(db_session, EMAIL, PASSWORD)

    await auth_service.login(db_session, settings, EMAIL, PASSWORD)
    await auth_service.login(db_session, settings, EMAIL, PASSWORD)

    families = {t.family_id for t in (await db_session.scalars(select(RefreshToken))).all()}
    assert len(families) == 2


async def test_rotation_keeps_the_family_and_revokes_the_used_token(
    db_session: AsyncSession, settings: Settings
) -> None:
    await auth_service.register_user(db_session, EMAIL, PASSWORD)
    pair = await auth_service.login(db_session, settings, EMAIL, PASSWORD)

    await auth_service.refresh_tokens(db_session, settings, pair.refresh_token)

    tokens = (
        await db_session.scalars(select(RefreshToken).order_by(RefreshToken.created_at))
    ).all()
    assert len(tokens) == 2
    assert tokens[0].family_id == tokens[1].family_id
    assert sum(t.revoked_at is None for t in tokens) == 1  # only the newest is still usable
