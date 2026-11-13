"""Reading PDFs in a separate process. Most tests start the real worker (about 0.3 s each)."""

import asyncio
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pypdf
import pytest

from app.cv import pdf_text, pdf_worker
from app.cv.pdf_text import PdfReadFailure, read_pdf_text
from tests.cv_helpers import make_pdf
from tests.settings_helpers import settings_with


async def read(data: bytes, **overrides: float) -> pdf_text.PdfText:
    options: dict[str, float] = {
        "max_pages": 10,
        "max_chars": 30_000,
        "timeout_seconds": 20.0,
        "memory_limit_mb": 512,
    }
    return await read_pdf_text(data, **(options | overrides))  # type: ignore[arg-type]


async def failure_of(data: bytes, **overrides: float) -> str:
    with pytest.raises(PdfReadFailure) as caught:
        await read(data, **overrides)
    return caught.value.code


# --- the real worker ----------------------------------------------------------------------------


async def test_the_text_and_page_count_of_a_pdf_come_back() -> None:
    pdf = make_pdf([["Alex Example", "Python and FastAPI"], ["Page two"]])

    result = await read(pdf)

    assert result.pages == 2
    assert "Python and FastAPI" in result.text
    assert "Page two" in result.text


async def test_a_pdf_without_a_text_layer_gives_empty_text_not_an_error() -> None:
    result = await read(make_pdf([[]]))  # the service turns this into "no text found"

    assert result.text.strip() == ""


async def test_too_many_pages_is_refused() -> None:
    assert await failure_of(make_pdf([["x"]] * 4), max_pages=3) == "too_many_pages"


async def test_too_much_text_is_refused() -> None:
    long_page = ["word " * 40] * 30  # about 6,000 characters

    assert await failure_of(make_pdf([long_page]), max_chars=500) == "too_long"


@pytest.mark.parametrize("data", [b"%PDF-1.4 this is not really a pdf", b"%PDF-", b""])
async def test_damaged_files_are_unreadable(data: bytes) -> None:
    assert await failure_of(data) == "unreadable"


async def test_the_failure_carries_a_fixed_code_and_nothing_from_the_file() -> None:
    secret = "SECRET-MARKER-12345"

    code = await failure_of(b"%PDF-1.4\n" + secret.encode() + b"\n garbage")

    assert code == "unreadable"
    assert secret not in code


# --- the page count is checked before any text is extracted -------------------------------------


def test_text_is_not_extracted_from_a_pdf_over_the_page_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def spy(self: pypdf.PageObject, *args: object, **kwargs: object) -> str:
        calls.append(1)
        return "text"

    monkeypatch.setattr(pypdf.PageObject, "extract_text", spy)

    verdict = pdf_worker.extract(make_pdf([["x"]] * 5), max_pages=2, max_chars=1000)

    assert verdict == {"ok": False, "error": "too_many_pages"}
    assert calls == []


def test_an_encrypted_pdf_is_refused_without_a_password_attempt() -> None:
    writer = pypdf.PdfWriter()
    writer.append(pypdf.PdfReader(io.BytesIO(make_pdf([["x"]]))))
    writer.encrypt(
        "a-password", algorithm="RC4-128"
    )  # the one algorithm that needs no extra package
    buffer = io.BytesIO()
    writer.write(buffer)

    verdict = pdf_worker.extract(buffer.getvalue(), max_pages=10, max_chars=1000)

    assert verdict == {"ok": False, "error": "encrypted"}


# --- the time limit really kills the process ----------------------------------------------------


async def test_a_reader_that_takes_too_long_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "survived"
    slow = tmp_path / "slow_worker.py"
    slow.write_text(
        "import sys, time\n"
        "sys.stdin.buffer.read()\n"
        "time.sleep(2.5)\n"
        f"open({str(marker)!r}, 'w').close()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pdf_text, "WORKER_SCRIPT", slow)
    started = time.monotonic()

    code = await failure_of(b"%PDF-1.4", timeout_seconds=0.7)

    assert code == "timeout"
    assert time.monotonic() - started < 2.0  # we did not wait for the sleep to end
    await asyncio.sleep(3.0)  # long enough for a surviving process to finish what it was doing
    assert not marker.exists()  # it was killed, not just abandoned


async def test_when_all_slots_are_taken_the_next_reader_gets_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(pdf_text, "_slots", slots)
    slots.acquire()
    try:
        assert await failure_of(make_pdf([["x"]]), timeout_seconds=0.3) == "busy"
    finally:
        slots.release()


# --- what the parent accepts back from the worker -----------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        pytest.param(1, b'{"ok": true, "pages": 1, "text": "x"}', id="nonzero-exit"),
        pytest.param(0, b"not json", id="not-json"),
        pytest.param(0, b"[1, 2]", id="not-an-object"),
        pytest.param(0, b'{"ok": true, "pages": "1", "text": "x"}', id="pages-not-a-number"),
        pytest.param(0, b'{"ok": true, "pages": 1}', id="no-text"),
        pytest.param(0, b'{"ok": false, "error": "something made up"}', id="unknown-code"),
        pytest.param(0, b'{"ok": false}', id="no-code"),
        # explicit id: a 2 MB value as the test id would overflow Windows' environment limit
        pytest.param(0, b"x" * (pdf_text.MAX_WORKER_OUTPUT_BYTES + 1), id="oversized-output"),
    ],
)
def test_anything_unexpected_from_the_worker_is_unreadable(returncode: int, stdout: bytes) -> None:
    with pytest.raises(PdfReadFailure) as caught:
        pdf_text._parse_output(returncode, stdout)

    assert caught.value.code == "unreadable"


def test_a_known_failure_code_from_the_worker_is_passed_on() -> None:
    with pytest.raises(PdfReadFailure) as caught:
        pdf_text._parse_output(0, json.dumps({"ok": False, "error": "encrypted"}).encode())

    assert caught.value.code == "encrypted"


# --- the memory limit of the reader process -----------------------------------------------------

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux only (Docker); see the docs"
)


def test_the_default_memory_limit_is_512_mb_and_has_a_floor() -> None:
    assert settings_with().cv_pdf_memory_limit_mb == 512
    with pytest.raises(ValueError):
        settings_with(cv_pdf_memory_limit_mb=64)  # too small for Python plus pypdf to run


@linux_only
def test_the_worker_sets_the_address_space_limit_to_the_configured_size() -> None:
    probe = (
        "import resource, sys; sys.path.insert(0, sys.argv[1]); import pdf_worker; "
        "pdf_worker._limit_memory(300); print(resource.getrlimit(resource.RLIMIT_AS))"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(pdf_text.WORKER_SCRIPT.parent)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == f"({300 * 1024 * 1024}, {300 * 1024 * 1024})"


@linux_only
def test_an_allocation_past_the_limit_raises_memory_error_in_the_worker_process() -> None:
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import pdf_worker; "
        "pdf_worker._limit_memory(200); "
        "exec(\"try: bytearray(400 * 1024 * 1024); print('allocated')\\n"
        "except MemoryError: print('MemoryError')\")"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(pdf_text.WORKER_SCRIPT.parent)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "MemoryError"


@linux_only
async def test_a_reader_with_too_little_memory_fails_and_the_default_is_enough() -> None:
    small = make_pdf([["Alex Example"]])
    dense_page = ["Python FastAPI PostgreSQL Docker Celery Redis " * 2] * 40
    dense = make_pdf([dense_page] * 10)  # the page limit's worth of dense text

    # Measured in the built image: a small PDF needs 30-40 MB, so 16 MB cannot work.
    assert await failure_of(small, memory_limit_mb=16) == "unreadable"
    assert (await read(small)).pages == 1  # the default 512 MB
    assert (await read(dense)).pages == 10


def test_a_worker_started_without_its_arguments_fails_cleanly() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(pdf_text.WORKER_SCRIPT), "10"],
        input=b"%PDF-",
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == b""
