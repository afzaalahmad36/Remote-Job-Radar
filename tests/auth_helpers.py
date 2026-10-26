from typing import Any

from httpx import AsyncClient, Response

EMAIL = "ann@example.com"
PASSWORD = "correct-horse-battery"


async def register(client: AsyncClient, email: str = EMAIL, password: str = PASSWORD) -> Response:
    return await client.post("/api/v1/auth/register", json={"email": email, "password": password})


async def login(client: AsyncClient, email: str = EMAIL, password: str = PASSWORD) -> Response:
    return await client.post("/api/v1/auth/login", json={"email": email, "password": password})


async def register_and_login(client: AsyncClient) -> dict[str, Any]:
    """Register the default user (once) and return a fresh token pair for a new login."""
    await register(client)
    response = await login(client)
    assert response.status_code == 200
    tokens: dict[str, Any] = response.json()
    return tokens


async def auth_headers(client: AsyncClient, email: str = EMAIL) -> dict[str, str]:
    """Register `email` (if new), log in, and return the Authorization header for it."""
    await register(client, email=email)
    response = await login(client, email=email)
    assert response.status_code == 200
    return bearer(response.json()["access_token"])


async def refresh(client: AsyncClient, refresh_token: str) -> Response:
    return await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})


def bearer(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}
