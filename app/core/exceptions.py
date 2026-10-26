from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class DomainError(Exception):
    """Base class for business-rule errors raised by services.

    Services never import FastAPI, so they raise these instead of HTTP errors. Subclasses set
    `status_code` and `code`; the handler below turns them into the API error shape.
    """

    status_code: int = 400
    code: str = "domain_error"

    def __init__(self, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = headers


class AuthenticationError(DomainError):
    """The caller is not who they claim to be (bad credentials, missing or invalid token)."""

    status_code = 401
    code = "authentication_failed"

    def __init__(self, detail: str) -> None:
        super().__init__(detail, headers={"WWW-Authenticate": "Bearer"})


class NotFoundError(DomainError):
    """Also used for rows that exist but belong to someone else: 404, never 403, so the API
    does not reveal which ids exist."""

    status_code = 404
    code = "not_found"


class InvalidConfigError(DomainError):
    status_code = 422
    code = "invalid_config"


class ConflictError(DomainError):
    """The request clashes with existing data (e.g. an email that is already registered)."""

    status_code = 409
    code = "conflict"


class DuplicateSourceError(ConflictError):
    code = "duplicate_source"


class RateLimitError(DomainError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            "Too many requests. Try again later.",
            headers={"Retry-After": str(retry_after_seconds)},
        )


def error_response(
    status_code: int, detail: Any, code: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Build the one error shape the whole API uses: {"detail": ..., "code": ...}."""
    return JSONResponse(
        status_code=status_code, content={"detail": detail, "code": code}, headers=headers
    )


async def handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DomainError)  # narrows the type; Starlette passes a plain Exception
    return error_response(exc.status_code, exc.detail, exc.code, exc.headers)


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    # Covers unknown routes (404), wrong methods (405) and any HTTPException raised in routers.
    assert isinstance(exc, StarletteHTTPException)
    return error_response(
        exc.status_code, exc.detail, f"http_{exc.status_code}", dict(exc.headers or {})
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(422, jsonable_errors(exc), "validation_error")


def jsonable_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Keep only the JSON-safe parts of each validation error (type, location, message)."""
    return [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, handle_domain_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
