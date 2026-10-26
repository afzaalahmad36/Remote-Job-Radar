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

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def error_response(status_code: int, detail: Any, code: str) -> JSONResponse:
    """Build the one error shape the whole API uses: {"detail": ..., "code": ...}."""
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code})


async def handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DomainError)  # narrows the type; Starlette passes a plain Exception
    return error_response(exc.status_code, exc.detail, exc.code)


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    # Covers unknown routes (404), wrong methods (405) and any HTTPException raised in routers.
    assert isinstance(exc, StarletteHTTPException)
    return error_response(exc.status_code, exc.detail, f"http_{exc.status_code}")


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
