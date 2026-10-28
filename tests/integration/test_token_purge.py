import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_refresh_token
from app.models import RefreshToken, User
from app.services import auth_service
from app.services.auth_service import REFRESH_TOKEN_PURGE_GRACE, purge_expired_refresh_tokens
from tests.auth_helpers import refresh, register_and_login

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)


async def make_token(
    session: AsyncSession,
    user: User,
    expires_at: datetime,
    revoked_at: datetime | None = None,
) -> uuid.UUID:
    token = RefreshToken(
        user_id=user.id,
        family_id=uuid.uuid4(),
        token_hash=uuid.uuid4().hex * 2,
        expires_at=expires_at,
        revoked_at=revoked_at,
    )
    session.add(token)
    await session.flush()
    return token.id


async def remaining(session: AsyncSession) -> set[uuid.UUID]:
    return set((await session.scalars(select(RefreshToken.id))).all())


@pytest.fixture
async def user(db_session: AsyncSession) -> User:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    return user


def test_the_grace_period_is_a_week() -> None:
    assert REFRESH_TOKEN_PURGE_GRACE == timedelta(days=7)


async def test_only_tokens_expired_for_longer_than_the_grace_period_are_deleted(
    db_session: AsyncSession, user: User
) -> None:
    valid = await make_token(db_session, user, NOW + 3 * DAY)
    just_expired = await make_token(db_session, user, NOW - DAY)
    at_the_boundary = await make_token(db_session, user, NOW - REFRESH_TOKEN_PURGE_GRACE)
    just_past = await make_token(
        db_session, user, NOW - REFRESH_TOKEN_PURGE_GRACE - timedelta(seconds=1)
    )
    long_expired = await make_token(db_session, user, NOW - 30 * DAY)

    deleted = await purge_expired_refresh_tokens(db_session, now=NOW)

    assert deleted == 2
    assert await remaining(db_session) == {valid, just_expired, at_the_boundary}
    assert {just_past, long_expired}.isdisjoint(await remaining(db_session))


async def test_a_revoked_token_is_kept_until_its_original_expiry_and_the_grace_have_passed(
    db_session: AsyncSession, user: User
) -> None:
    # Rotation revokes the old token but it keeps its original expiry: reuse detection needs it.
    rotated_unexpired = await make_token(db_session, user, NOW + 5 * DAY, revoked_at=NOW - DAY)
    rotated_recently_expired = await make_token(
        db_session, user, NOW - 2 * DAY, revoked_at=NOW - 9 * DAY
    )
    rotated_long_expired = await make_token(
        db_session, user, NOW - 20 * DAY, revoked_at=NOW - 27 * DAY
    )

    deleted = await purge_expired_refresh_tokens(db_session, now=NOW)

    assert deleted == 1
    assert await remaining(db_session) == {rotated_unexpired, rotated_recently_expired}
    assert rotated_long_expired not in await remaining(db_session)


async def test_revoked_status_never_makes_a_token_deletable_by_itself(
    db_session: AsyncSession, user: User
) -> None:
    token = await make_token(db_session, user, NOW + 30 * DAY, revoked_at=NOW - 100 * DAY)

    assert await purge_expired_refresh_tokens(db_session, now=NOW) == 0
    assert await remaining(db_session) == {token}


async def test_a_custom_grace_period_is_respected(db_session: AsyncSession, user: User) -> None:
    old = await make_token(db_session, user, NOW - 2 * DAY)

    assert await purge_expired_refresh_tokens(db_session, now=NOW, grace=timedelta(days=3)) == 0
    assert await purge_expired_refresh_tokens(db_session, now=NOW, grace=timedelta(days=1)) == 1
    assert old not in await remaining(db_session)


@pytest.mark.parametrize(
    ("expired", "batch_size"), [(0, 2), (1, 2), (4, 2), (5, 2), (7, 3), (3, 100)]
)
async def test_a_backlog_is_deleted_in_batches_and_all_of_it_goes(
    db_session: AsyncSession, user: User, expired: int, batch_size: int
) -> None:
    keep = await make_token(db_session, user, NOW + DAY)
    for _ in range(expired):
        await make_token(db_session, user, NOW - 30 * DAY)

    deleted = await purge_expired_refresh_tokens(db_session, now=NOW, batch_size=batch_size)

    assert deleted == expired
    assert await remaining(db_session) == {keep}


async def test_purging_with_nothing_to_do_returns_zero(db_session: AsyncSession) -> None:
    assert await purge_expired_refresh_tokens(db_session, now=NOW) == 0


# --- what the purge rule means for reuse detection -----------------------------------------------


async def test_reuse_detection_still_works_after_a_purge_for_a_rotated_unexpired_token(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    first = await register_and_login(db_client)
    second = (await refresh(db_client, first["refresh_token"])).json()  # rotates: first is revoked

    await purge_expired_refresh_tokens(db_session)  # the real "now": nothing here has expired
    replay = await refresh(db_client, first["refresh_token"])  # a stolen copy of the old token
    newest_after_replay = await refresh(db_client, second["refresh_token"])

    assert replay.status_code == 401
    assert newest_after_replay.status_code == 401  # the replay was detected: the family is revoked


async def test_a_token_purged_after_its_grace_period_is_just_unknown_afterwards(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    tokens = await register_and_login(db_client)
    stored = (await db_session.scalars(select(RefreshToken))).one()
    assert stored.token_hash == hash_refresh_token(tokens["refresh_token"])
    stored.expires_at = datetime.now(UTC) - 30 * DAY
    stored.revoked_at = datetime.now(UTC) - 31 * DAY
    await db_session.flush()

    assert await purge_expired_refresh_tokens(db_session) == 1
    replay = await refresh(db_client, tokens["refresh_token"])

    # Not found, so no family revocation: acceptable for a token that expired over a week ago.
    assert replay.status_code == 401
    assert replay.json()["detail"] == auth_service.INVALID_REFRESH_TOKEN
