from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RefreshToken
from tests.auth_helpers import bearer, login, refresh, register_and_login


async def test_refresh_rotates_the_token_pair(db_client: AsyncClient) -> None:
    old = await register_and_login(db_client)

    response = await refresh(db_client, old["refresh_token"])

    assert response.status_code == 200
    new = response.json()
    assert new["refresh_token"] != old["refresh_token"]
    me = await db_client.get("/api/v1/users/me", headers=bearer(new["access_token"]))
    assert me.status_code == 200


async def test_replaying_a_used_refresh_token_revokes_the_whole_family(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    first = await register_and_login(db_client)
    second = (await refresh(db_client, first["refresh_token"])).json()

    replay = await refresh(db_client, first["refresh_token"])  # an attacker, or a stale copy
    legit_after_replay = await refresh(db_client, second["refresh_token"])

    assert replay.status_code == 401
    assert legit_after_replay.status_code == 401  # the newest token died with its family
    tokens = (await db_session.scalars(select(RefreshToken))).all()
    assert len(tokens) == 2
    assert all(token.revoked_at is not None for token in tokens)


async def test_reuse_detection_leaves_other_logins_alone(db_client: AsyncClient) -> None:
    laptop = await register_and_login(db_client)
    phone = (await login(db_client)).json()  # a second login = a second family
    await refresh(db_client, laptop["refresh_token"])

    await refresh(db_client, laptop["refresh_token"])  # replay: kills the laptop family only

    assert (await refresh(db_client, phone["refresh_token"])).status_code == 200


async def test_logout_revokes_the_session(db_client: AsyncClient) -> None:
    tokens = await register_and_login(db_client)

    logout = await db_client.post(
        "/api/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]}
    )

    assert logout.status_code == 204
    assert (await refresh(db_client, tokens["refresh_token"])).status_code == 401


async def test_logout_revokes_tokens_issued_by_earlier_refreshes(db_client: AsyncClient) -> None:
    first = await register_and_login(db_client)
    second = (await refresh(db_client, first["refresh_token"])).json()

    await db_client.post("/api/v1/auth/logout", json={"refresh_token": second["refresh_token"]})

    assert (await refresh(db_client, second["refresh_token"])).status_code == 401


async def test_logout_with_an_unknown_token_is_still_204(db_client: AsyncClient) -> None:
    response = await db_client.post("/api/v1/auth/logout", json={"refresh_token": "nope"})

    assert response.status_code == 204  # no hint about whether a token exists


async def test_logout_answers_identically_for_unknown_valid_and_revoked_tokens(
    db_client: AsyncClient,
) -> None:
    tokens = await register_and_login(db_client)
    used = (await refresh(db_client, tokens["refresh_token"])).json()  # rotates: the first is used
    logout = "/api/v1/auth/logout"

    unknown = await db_client.post(logout, json={"refresh_token": "never-issued"})
    valid = await db_client.post(logout, json={"refresh_token": used["refresh_token"]})
    again = await db_client.post(logout, json={"refresh_token": used["refresh_token"]})
    rotated_away = await db_client.post(logout, json={"refresh_token": tokens["refresh_token"]})

    for response in (unknown, valid, again, rotated_away):
        assert response.status_code == 204
        assert response.content == b""
    assert valid.headers.keys() == unknown.headers.keys() == again.headers.keys()


async def test_unknown_refresh_token_is_rejected(db_client: AsyncClient) -> None:
    response = await refresh(db_client, "not-a-real-token")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid refresh token", "code": "authentication_failed"}


async def test_expired_refresh_token_is_rejected(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    tokens = await register_and_login(db_client)
    await db_session.execute(
        update(RefreshToken).values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    )

    response = await refresh(db_client, tokens["refresh_token"])

    assert response.status_code == 401


async def test_refresh_requires_a_body(db_client: AsyncClient) -> None:
    response = await db_client.post("/api/v1/auth/refresh", json={})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
