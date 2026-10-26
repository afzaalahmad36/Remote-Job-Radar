from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.rate_limit import RateLimiter
from app.routers import auth, health, sources, users

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
    # One limiter per app instance, kept on app.state so each test app starts with a clean slate.
    app.state.auth_rate_limiter = RateLimiter(
        settings.auth_rate_limit_attempts, settings.auth_rate_limit_window_seconds
    )
    app.include_router(health.router, prefix=API_V1_PREFIX)
    app.include_router(auth.router, prefix=API_V1_PREFIX)
    app.include_router(users.router, prefix=API_V1_PREFIX)
    app.include_router(sources.router, prefix=API_V1_PREFIX)
    return app
