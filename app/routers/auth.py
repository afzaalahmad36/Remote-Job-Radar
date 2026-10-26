from fastapi import APIRouter, Depends, Response, status

from app.api_deps import SessionDep, SettingsDep, rate_limit_auth
from app.schemas.auth import LoginCreate, RefreshCreate, TokenOut, UserCreate, UserRead
from app.services import auth_service
from app.services.auth_service import TokenPair

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(rate_limit_auth)])


def _token_out(pair: TokenPair) -> TokenOut:
    return TokenOut(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, session: SessionDep) -> UserRead:
    user = await auth_service.register_user(session, body.email, body.password)
    return UserRead.model_validate(user)


@router.post("/login")
async def login(body: LoginCreate, session: SessionDep, settings: SettingsDep) -> TokenOut:
    return _token_out(await auth_service.login(session, settings, body.email, body.password))


@router.post("/refresh")
async def refresh(body: RefreshCreate, session: SessionDep, settings: SettingsDep) -> TokenOut:
    return _token_out(await auth_service.refresh_tokens(session, settings, body.refresh_token))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshCreate, session: SessionDep) -> Response:
    await auth_service.logout(session, body.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
