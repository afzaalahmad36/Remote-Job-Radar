"""The per-test time limit really stops a stuck test, whatever kind of test it is.

Each case runs a throwaway test file in its own pytest process with a 1 s limit, so the stuck test
cannot hold up this suite. Covers the cases that matter here: an async test, a plain test that
blocks, and a plain test that starts its own event loop with asyncio.run (like the Celery task
tests do).
"""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

STUCK_TESTS = {
    "async-test": """
        import asyncio

        async def test_stuck():
            await asyncio.sleep(60)
    """,
    "sync-test-that-blocks": """
        import time

        def test_stuck():
            time.sleep(60)
    """,
    "sync-test-with-its-own-event-loop": """
        import asyncio

        def test_stuck():
            asyncio.run(asyncio.sleep(60))  # what the Celery task tests do
    """,
    "sync-test-blocked-on-a-socket": """
        import socket

        def test_stuck():
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            client = socket.create_connection(server.getsockname())
            client.recv(1)  # waits for data that never comes
    """,
}


def pytest_config() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        options: dict[str, object] = tomllib.load(handle)["tool"]["pytest"]["ini_options"]
    return options


def test_the_project_sets_a_60_second_limit_and_a_stack_dump_before_it() -> None:
    options = pytest_config()

    assert options["timeout"] == 60
    assert 0 < int(options["faulthandler_timeout"]) < 60  # type: ignore[call-overload]


@pytest.mark.parametrize("source", STUCK_TESTS.values(), ids=STUCK_TESTS.keys())
def test_a_stuck_test_is_stopped_by_the_time_limit(source: str, tmp_path: Path) -> None:
    (tmp_path / "test_stuck.py").write_text(textwrap.dedent(source), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "timeout=1",
            "-o",
            "asyncio_mode=auto",
            "--rootdir",
            str(tmp_path),
            str(tmp_path / "test_stuck.py"),
        ],  # fmt: skip
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=40,  # the stuck test sleeps for 60 s; getting here at all means it was stopped
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Timeout" in output  # pytest-timeout's own message, with the stack of the stuck code
