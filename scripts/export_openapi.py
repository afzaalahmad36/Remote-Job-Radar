"""Write the API's OpenAPI schema to frontend/openapi.json, the source of the web app's types.

    uv run python -m scripts.export_openapi

It builds the app in memory only: no server, no database, no network, and its settings do not
come from `.env` (the placeholders below do not change the schema). A test fails if the file is
stale.
"""

import json
from pathlib import Path

from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app

TARGET = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def render_schema() -> str:
    """The schema as stable JSON text (sorted keys, trailing newline)."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]  # pydantic-settings: do not read .env
        database_url="postgresql+asyncpg://unused@localhost/unused",
        redis_url="redis://localhost:6379/0",
        secret_key=SecretStr("schema-export-only"),
    )
    return json.dumps(create_app(settings).openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    TARGET.write_text(render_schema(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
