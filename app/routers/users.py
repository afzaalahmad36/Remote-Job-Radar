from fastapi import APIRouter

from app.api_deps import CurrentUserDep
from app.schemas.auth import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def read_me(user: CurrentUserDep) -> UserRead:
    return UserRead.model_validate(user)
