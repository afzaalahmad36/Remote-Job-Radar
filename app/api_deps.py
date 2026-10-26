from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.rate_limit import RateLimiter
from app.core.security import decode_access_token
from app.core.url_safety import Resolver, system_resolver
from app.db.session import get_session
from app.models import User
from app.schemas.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, MAX_PAGE_OFFSET, PageParams
from app.services import auth_service

# `Depends` is FastAPI's dependency injection: FastAPI calls get_settings() for each request
# and passes the result in. Tests can swap it via `app.dependency_overrides`.
# `Annotated[...]` bundles the type and the Depends so endpoints just write `settings: SettingsDep`.
SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# Reads "Authorization: Bearer <token>" and shows the Authorize button in the API docs.
# auto_error=False so a missing header reaches our code and gets our {detail, code} error shape.
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    """The logged-in user, from the access token. Raises 401 if absent, invalid or expired."""
    if credentials is None:
        raise AuthenticationError("Not authenticated")
    user_id = decode_access_token(credentials.credentials, settings.secret_key.get_secret_value())
    user = await auth_service.get_user(session, user_id)
    if user is None:  # a valid token whose user has since been deleted
        raise AuthenticationError("Not authenticated")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_resolver() -> Resolver:
    """The DNS resolver used to check user-supplied URLs; tests override it so DNS is never hit."""
    return system_resolver


ResolverDep = Annotated[Resolver, Depends(get_resolver)]


def get_page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
) -> PageParams:
    """Reads and bounds `?limit=&offset=`; out-of-range values get a 422."""
    return PageParams(limit=limit, offset=offset)


PageDep = Annotated[PageParams, Depends(get_page_params)]


async def rate_limit_auth(request: Request) -> None:
    """Limit each client IP per auth endpoint.

    The key is the socket peer address. Uvicorn must run with `--no-proxy-headers` (as the
    Dockerfile does), or it would rewrite this address from a client-supplied X-Forwarded-For
    header and let anyone dodge the limit. Behind a reverse proxy, enabling forwarded headers
    for that proxy's address only is a deploy-time task (see the deferred items in the PR).
    """
    limiter: RateLimiter = request.app.state.auth_rate_limiter
    client_ip = request.client.host if request.client else "unknown"
    limiter.check(f"{request.url.path}:{client_ip}")
