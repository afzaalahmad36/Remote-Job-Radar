"""The committed OpenAPI file, from which the web app's types are generated, is current."""

from scripts.export_openapi import TARGET, render_schema


def test_the_committed_schema_matches_the_api() -> None:
    assert TARGET.read_text(encoding="utf-8") == render_schema(), (
        "run `uv run python -m scripts.export_openapi`, then `npm run gen:api` in frontend/"
    )
