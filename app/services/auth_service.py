import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_against_dummy,
    verify_password,
)
from app.models import RefreshToken, User

INVALID_CREDENTIALS = "Invalid email or password"
INVALID_REFRESH_TOKEN = "Invalid refresh token"


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int


def _normalize_email(email: str) -> str:
    return email.strip().lower()


async def register_user(session: AsyncSession, email: str, password: str) -> User:
    """Create a user. Raises ConflictError if the email is already registered."""
    user = User(email=_normalize_email(email), hashed_password=await hash_password(password))
    try:
        # A savepoint, so a duplicate email only undoes this insert, not the caller's whole
        # transaction. The unique constraint (not a prior SELECT) is what decides, which also
        # makes two simultaneous registrations of one email safe.
        async with session.begin_nested():
            session.add(user)
    except IntegrityError as error:
        raise ConflictError("Email already registered") from error
    await session.commit()
    return user


async def login(
    session: AsyncSession,
    settings: Settings,
    email: str,
    password: str,
    now: datetime | None = None,
) -> TokenPair:
    """Check credentials and start a new refresh-token family."""
    now = now or datetime.now(UTC)
    user = await session.scalar(select(User).where(User.email == _normalize_email(email)))
    if user is None:
        await verify_against_dummy(password)  # same cost as a real check; see its docstring
        raise AuthenticationError(INVALID_CREDENTIALS)
    if not await verify_password(password, user.hashed_password):
        raise AuthenticationError(INVALID_CREDENTIALS)  # same message as above, on purpose

    refresh_token = await _issue_refresh_token(session, settings, user.id, uuid.uuid4(), now)
    await session.commit()
    return _token_pair(settings, user.id, refresh_token, now)


async def refresh_tokens(
    session: AsyncSession, settings: Settings, raw_token: str, now: datetime | None = None
) -> TokenPair:
    """Exchange a refresh token for a new access + refresh token (rotation).

    The used token is revoked. Presenting a token that was already used or revoked revokes its
    whole family, i.e. every token descended from the same login, and raises AuthenticationError.
    """
    now = now or datetime.now(UTC)
    # FOR UPDATE locks the row, so two simultaneous requests with the same token cannot both win.
    stored = await session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(raw_token))
        .with_for_update()
    )
    if stored is None:
        raise AuthenticationError(INVALID_REFRESH_TOKEN)
    if stored.revoked_at is not None:
        await _revoke_family(session, stored.family_id, now)
        await session.commit()  # commit *before* raising, or the revocation would be rolled back
        raise AuthenticationError(INVALID_REFRESH_TOKEN)
    if stored.expires_at <= now:
        raise AuthenticationError(INVALID_REFRESH_TOKEN)

    stored.revoked_at = now
    new_token = await _issue_refresh_token(session, settings, stored.user_id, stored.family_id, now)
    await session.commit()
    return _token_pair(settings, stored.user_id, new_token, now)


async def logout(session: AsyncSession, raw_token: str, now: datetime | None = None) -> None:
    """Revoke the token's whole family (this login's session). Unknown tokens are ignored."""
    now = now or datetime.now(UTC)
    stored = await session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw_token))
    )
    if stored is not None:
        await _revoke_family(session, stored.family_id, now)
        await session.commit()


REFRESH_TOKEN_PURGE_GRACE = timedelta(days=7)
PURGE_BATCH_SIZE = 5_000


async def purge_expired_refresh_tokens(
    session: AsyncSession,
    now: datetime | None = None,
    grace: timedelta = REFRESH_TOKEN_PURGE_GRACE,
    batch_size: int = PURGE_BATCH_SIZE,
) -> int:
    """Delete refresh tokens that expired more than `grace` ago; return how many were deleted.

    The rule is about `expires_at` only, never about `revoked_at`:

      * A token is deleted only once it is expired AND `grace` (7 days) has passed since.
      * A revoked token whose original expiry has not passed is NEVER deleted. Every rotation
        revokes the old token, and keeping it is what lets reuse detection work: replaying it
        must still be found so the whole token family can be revoked. If it were purged, a
        replay would just look like an unknown token.
      * Once a token is past its expiry nobody can use it (it would be rejected as expired), so
        what is lost by deleting it is only the "this was a replay" signal for a dead token.
        The grace period keeps even that for a week, which also covers clock differences and
        gives time to investigate recently expired sessions.

    Deletes run in batches, committing after each, so a large backlog never holds a long lock.
    """
    cutoff = (now or datetime.now(UTC)) - grace
    deleted = 0
    while True:
        batch = select(RefreshToken.id).where(RefreshToken.expires_at < cutoff).limit(batch_size)
        result = await session.execute(delete(RefreshToken).where(RefreshToken.id.in_(batch)))
        await session.commit()
        removed = int(result.rowcount)  # type: ignore[attr-defined]
        deleted += removed
        if removed < batch_size:
            return deleted


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.get(User, user_id)


async def _issue_refresh_token(
    session: AsyncSession,
    settings: Settings,
    user_id: uuid.UUID,
    family_id: uuid.UUID,
    now: datetime,
) -> str:
    raw_token = generate_refresh_token()
    session.add(
        RefreshToken(
            user_id=user_id,
            family_id=family_id,
            token_hash=hash_refresh_token(raw_token),
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
        )
    )
    return raw_token


async def _revoke_family(session: AsyncSession, family_id: uuid.UUID, now: datetime) -> None:
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )


def _token_pair(
    settings: Settings, user_id: uuid.UUID, refresh_token: str, now: datetime
) -> TokenPair:
    access_token = create_access_token(
        user_id, settings.secret_key.get_secret_value(), settings.access_token_expire_minutes, now
    )
    return TokenPair(access_token, refresh_token, settings.access_token_expire_minutes * 60)
