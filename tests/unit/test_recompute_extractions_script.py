"""The parts of `scripts/recompute_extractions.py` that need no database."""

import ast
import re
from pathlib import Path

import pytest

from scripts import recompute_extractions as script
from scripts.db_guard import ScriptError

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL = "postgresql+asyncpg://u:p@localhost:5434/jobradar"


@pytest.mark.parametrize(("version", "number"), [("rules_v1", 1), ("rules_v12", 12)])
def test_version_number_reads_the_digits(version: str, number: int) -> None:
    assert script.version_number(version) == number


@pytest.mark.parametrize("version", ["", "rules_v", "rules_vX", "v2", "rules_v2b", "rules_v-1"])
def test_version_number_rejects_other_spellings(version: str) -> None:
    with pytest.raises(ValueError, match="not a rules version"):
        script.version_number(version)


def test_a_real_run_without_the_flag_is_refused_and_the_reminder_has_the_dump_and_the_size_check(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "get_settings", lambda: pytest.fail("must refuse before settings"))

    code = script.main([])

    err = capsys.readouterr().err
    assert code == 2
    assert script.CONFIRM_FLAG in err
    assert "pg_dump" in err and "-Fc" in err and "jobradar" in err
    assert "Check the file size" in err


def test_the_help_text_shows_the_same_reminder(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        script.main(["--help"])

    out = capsys.readouterr().out
    assert "pg_dump" in out and "Check the file size" in out


def test_a_batch_size_outside_the_range_is_refused() -> None:
    import asyncio

    for size in (0, script.MAX_BATCH_SIZE + 1):
        with pytest.raises(ScriptError, match="--batch-size"):
            asyncio.run(script.recompute(None, batch_size=size))  # type: ignore[arg-type]


def test_only_the_local_jobradar_database_is_allowed() -> None:
    assert script.check_local_database(LOCAL) == "localhost:5434/jobradar"
    assert script.check_local_database(LOCAL.replace("localhost", "127.0.0.1")).startswith(
        "127.0.0.1"
    )


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        (LOCAL.replace("/jobradar", "/jobradar_test"), "named 'jobradar'"),
        (LOCAL.replace("/jobradar", "/other"), "named 'jobradar'"),
        (LOCAL.replace("localhost", "db.example.com"), "this machine"),
        ("not a url", "valid database URL"),
    ],
)
def test_a_wrong_database_name_or_a_remote_host_is_refused(url: str, reason: str) -> None:
    with pytest.raises(ScriptError, match=reason):
        script.check_local_database(url)


def test_no_app_code_imports_the_script() -> None:
    for path in (ROOT / "app").rglob("*.py"):
        assert "recompute_extractions" not in path.read_text(encoding="utf-8"), path


def test_the_script_makes_no_provider_request() -> None:
    tree = ast.parse((ROOT / "scripts" / "recompute_extractions.py").read_text(encoding="utf-8"))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(re.match(r"app\.(llm|services|tasks)", name) for name in imported)
    assert not any(name.startswith(("httpx", "requests", "openai", "google")) for name in imported)
