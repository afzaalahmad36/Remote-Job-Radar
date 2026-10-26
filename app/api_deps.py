from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings

# `Depends` is FastAPI's dependency injection: FastAPI calls get_settings() for each request
# and passes the result in. Tests can swap it via `app.dependency_overrides`.
# `Annotated[...]` bundles the type and the Depends so endpoints just write `settings: SettingsDep`.
SettingsDep = Annotated[Settings, Depends(get_settings)]
