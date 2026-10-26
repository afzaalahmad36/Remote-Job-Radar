import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.rate_limit import RateLimiter
from app.core.security import create_access_token, hash_refresh_token
from app.models import RefreshToken, User
from tests.auth_helpers import EMAIL, PASSWORD, bearer, login, register, register_and_login


async def test_register_creates_a_user_without_exposing_secrets(db_client: AsyncClient) -> None:
    response = await register(db_client)

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == EMAIL
    assert set(body) == {"id", "email", "created_at"}


async def test_password_is_stored_as_an_argon2_hash(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    await register(db_client)

    stored = await db_session.scalar(select(User.hashed_password))

    assert stored is not None
    assert stored.startswith("$argon2id$")
    assert PASSWORD not in stored


async def test_register_normalizes_email_and_blocks_case_variants(db_client: AsyncClient) -> None:
    first = await register(db_client, email="  Ann@Example.COM ")
    duplicate = await register(db_client, email="ANN@example.com")

    assert first.json()["email"] == "ann@example.com"
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Email already registered", "code": "conflict"}


async def test_duplicate_registration_does_not_break_the_session(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    await register(db_client)
    await register(db_client)  # 409

    assert len((await db_session.scalars(select(User))).all()) == 1  # and the session still works


@pytest.mark.parametrize(
    ("email", "password"),
    [("not-an-email", PASSWORD), (EMAIL, "short"), (EMAIL, "x" * 129)],
)
async def test_register_rejects_invalid_input(
    db_client: AsyncClient, email: str, password: str
) -> None:
    response = await register(db_client, email=email, password=password)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_login_returns_tokens(db_client: AsyncClient) -> None:
    await register(db_client)

    response = await login(db_client)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60
    assert body["access_token"] and body["refresh_token"]


async def test_login_email_is_case_insensitive(db_client: AsyncClient) -> None:
    await register(db_client)

    assert (await login(db_client, email="ANN@Example.com")).status_code == 200


async def test_wrong_password_and_unknown_email_are_indistinguishable(
    db_client: AsyncClient,
) -> None:
    await register(db_client)

    wrong_password = await login(db_client, password="not-the-password")
    unknown_email = await login(db_client, email="nobody@example.com")

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json()["code"] == "authentication_failed"


async def test_me_requires_a_token(db_client: AsyncClient) -> None:
    response = await db_client.get("/api/v1/users/me")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated", "code": "authentication_failed"}
    assert response.headers["www-authenticate"] == "Bearer"


async def test_me_returns_the_current_user(db_client: AsyncClient) -> None:
    tokens = await register_and_login(db_client)

    response = await db_client.get("/api/v1/users/me", headers=bearer(tokens["access_token"]))

    assert response.status_code == 200
    assert response.json()["email"] == EMAIL


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer not.a.token"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer"},
    ],
)
async def test_me_rejects_invalid_credentials(
    db_client: AsyncClient, headers: dict[str, str]
) -> None:
    response = await db_client.get("/api/v1/users/me", headers=headers)

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"


async def test_me_rejects_an_expired_token(db_client: AsyncClient, settings: Settings) -> None:
    user_id = uuid.UUID((await register(db_client)).json()["id"])
    expired = create_access_token(
        user_id,
        settings.secret_key.get_secret_value(),
        expires_minutes=5,
        now=datetime.now(UTC) - timedelta(hours=1),
    )

    response = await db_client.get("/api/v1/users/me", headers=bearer(expired))

    assert response.status_code == 401


async def test_me_rejects_a_valid_token_whose_user_was_deleted(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    tokens = await register_and_login(db_client)
    await db_session.execute(delete(User))

    response = await db_client.get("/api/v1/users/me", headers=bearer(tokens["access_token"]))

    assert response.status_code == 401


async def test_refresh_token_is_stored_hashed(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    tokens = await register_and_login(db_client)

    stored = (await db_session.scalars(select(RefreshToken))).one()

    assert stored.token_hash == hash_refresh_token(tokens["refresh_token"])
    assert tokens["refresh_token"] not in (stored.token_hash, str(stored.id))
    assert stored.revoked_at is None


async def test_auth_endpoints_are_rate_limited(app: FastAPI, db_client: AsyncClient) -> None:
    app.state.auth_rate_limiter = RateLimiter(max_attempts=3, window_seconds=60)

    statuses = [(await login(db_client)).status_code for _ in range(4)]
    other_endpoint = await register(db_client)

    assert statuses == [401, 401, 401, 429]  # the 4th attempt is blocked before any check
    assert other_endpoint.status_code == 201  # each endpoint has its own budget


async def test_rate_limit_ignores_x_forwarded_for(app: FastAPI, db_client: AsyncClient) -> None:
    app.state.auth_rate_limiter = RateLimiter(max_attempts=3, window_seconds=60)
    body = {"email": EMAIL, "password": "wrong-password"}

    statuses = []
    for i in range(4):  # a client pretending to be a different address on every attempt
        response = await db_client.post(
            "/api/v1/auth/login", json=body, headers={"X-Forwarded-For": f"203.0.113.{i}"}
        )
        statuses.append(response.status_code)

    assert statuses == [401, 401, 401, 429]


async def test_a_refresh_token_is_not_accepted_as_an_access_token(db_client: AsyncClient) -> None:
    tokens = await register_and_login(db_client)

    response = await db_client.get("/api/v1/users/me", headers=bearer(tokens["refresh_token"]))

    assert response.status_code == 401


async def test_an_access_token_is_not_accepted_as_a_refresh_token(db_client: AsyncClient) -> None:
    tokens = await register_and_login(db_client)

    response = await db_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]}
    )

    assert response.status_code == 401


async def test_every_401_carries_www_authenticate_bearer(db_client: AsyncClient) -> None:
    await register(db_client)
    responses = [
        await db_client.get("/api/v1/users/me"),  # no token
        await db_client.get("/api/v1/users/me", headers=bearer("garbage")),  # bad token
        await login(db_client, password="wrong-password"),  # bad credentials
        await db_client.post("/api/v1/auth/refresh", json={"refresh_token": "unknown"}),
    ]

    assert [r.status_code for r in responses] == [401, 401, 401, 401]
    assert [r.headers["www-authenticate"] for r in responses] == ["Bearer"] * 4


async def test_validation_errors_do_not_echo_the_submitted_password(
    db_client: AsyncClient,
) -> None:
    response = await register(db_client, password="pw-1234")  # too short, so 422

    assert response.status_code == 422
    assert "pw-1234" not in response.text


async def test_rate_limited_response_has_retry_after_and_error_shape(
    app: FastAPI, db_client: AsyncClient
) -> None:
    app.state.auth_rate_limiter = RateLimiter(max_attempts=1, window_seconds=60)
    await login(db_client)

    response = await login(db_client)

    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"
    assert 1 <= int(response.headers["retry-after"]) <= 60
