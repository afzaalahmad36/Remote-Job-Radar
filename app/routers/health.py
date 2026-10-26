from fastapi import APIRouter

from app.api_deps import SettingsDep
from app.schemas.health import HealthOut

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(settings: SettingsDep) -> HealthOut:
    """Liveness check: returns 200 if the app is up. Does not touch the database yet."""
    return HealthOut(status="ok", env=settings.env)
