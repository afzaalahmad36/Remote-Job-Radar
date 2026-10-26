from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.routers import health

API_V1_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app (the "app factory" pattern).

    Importing this module builds nothing; Uvicorn calls the factory itself:
    `uvicorn app.main:create_app --factory`. Tests call it too, so each gets a fresh app.

    `settings` is only for construction-time wiring (CORS, logging); request handlers get
    theirs through `Depends(get_settings)`, which tests replace via `app.dependency_overrides`.
    """
    settings = settings or get_settings()
    setup_logging(settings.env)

    app = FastAPI(title="Remote Job Radar")
    # allow_credentials stays False (the default): auth will use an Authorization header, not
    # cookies, and browsers reject credentials combined with a wildcard origin anyway.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health.router, prefix=API_V1_PREFIX)
    return app
