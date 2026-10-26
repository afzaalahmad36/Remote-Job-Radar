import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.exceptions import DomainError


class TeapotError(DomainError):
    status_code = 418
    code = "teapot"


@pytest.fixture
def app_with_test_routes(app: FastAPI) -> FastAPI:
    async def boom() -> None:
        raise TeapotError("I'm a teapot")

    async def needs_int(n: int) -> None:
        return None

    app.add_api_route("/boom", boom)
    app.add_api_route("/needs-int", needs_int)
    return app


async def test_unknown_route_returns_error_shape(client: AsyncClient) -> None:
    response = await client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found", "code": "http_404"}


@pytest.mark.usefixtures("app_with_test_routes")
async def test_domain_error_returns_error_shape(client: AsyncClient) -> None:
    response = await client.get("/boom")

    assert response.status_code == 418
    assert response.json() == {"detail": "I'm a teapot", "code": "teapot"}


@pytest.mark.usefixtures("app_with_test_routes")
async def test_validation_error_returns_error_shape(client: AsyncClient) -> None:
    response = await client.get("/needs-int", params={"n": "abc"})

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "validation_error"
    assert body["detail"][0]["loc"] == ["query", "n"]
