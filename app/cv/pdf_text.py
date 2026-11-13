"""Reading a PDF's text safely: in a separate process with a hard timeout.

PDFs come from users, so they are untrusted. Reading one happens in `pdf_worker.py`, a script run
in its own Python process. If it takes longer than the timeout, we kill it. (A thread could not be
killed, so a hostile PDF could keep it busy forever.) The bytes go in through stdin and the text
comes back through stdout: no file is written.

Python note: `subprocess.run` blocks, so it runs in a worker thread via `asyncio.to_thread`, which
keeps the event loop free. The thread itself is bounded, because `run` kills the child when the
timeout ends.
"""

import asyncio
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

WORKER_SCRIPT = Path(__file__).with_name("pdf_worker.py")
# At most this many PDFs are read at the same moment in this API process (each one is a Python
# process). The rate limit bounds one user; this bounds everyone together. Like the rate limiter,
# it is per API process: N processes allow N times as many.
MAX_CONCURRENT_PARSES = 2
# Longest JSON line we accept back: the text limit is far below it; it only stops a runaway child.
MAX_WORKER_OUTPUT_BYTES = 2_000_000

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_PARSES)

# The only reasons a read can fail. Fixed codes: nothing from the file is ever kept.
FAILURE_CODES = frozenset(
    {"encrypted", "too_many_pages", "too_long", "unreadable", "timeout", "busy"}
)


@dataclass(frozen=True)
class PdfText:
    text: str
    pages: int


class PdfReadFailure(Exception):
    """The PDF could not be read. `code` is one of FAILURE_CODES."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def read_pdf_text(
    data: bytes, *, max_pages: int, max_chars: int, timeout_seconds: float, memory_limit_mb: int
) -> PdfText:
    """The text of a PDF, or PdfReadFailure. Never raises anything that carries file content."""
    return await asyncio.to_thread(
        _run_worker, data, max_pages, max_chars, timeout_seconds, memory_limit_mb
    )


def _run_worker(
    data: bytes, max_pages: int, max_chars: int, timeout_seconds: float, memory_limit_mb: int
) -> PdfText:
    # Waiting for a free slot is part of the same time budget.
    if not _slots.acquire(timeout=timeout_seconds):
        raise PdfReadFailure("busy")
    try:
        # "-I": isolated mode (ignores PYTHON* variables and the current directory), so nothing
        # in the environment can change what the child imports. sys.executable is this
        # interpreter, so the child sees the same virtualenv (and pypdf).
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(WORKER_SCRIPT),
                str(max_pages),
                str(max_chars),
                str(memory_limit_mb),
            ],
            input=data,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # `from None`: the exception holds the child's partial output; do not chain it.
        raise PdfReadFailure("timeout") from None
    except OSError:
        raise PdfReadFailure("unreadable") from None
    finally:
        _slots.release()
    return _parse_output(completed.returncode, completed.stdout)


def _parse_output(returncode: int, stdout: bytes) -> PdfText:
    if returncode != 0 or len(stdout) > MAX_WORKER_OUTPUT_BYTES:
        raise PdfReadFailure("unreadable")
    try:
        verdict = json.loads(stdout)
    except ValueError:
        raise PdfReadFailure("unreadable") from None
    if not isinstance(verdict, dict):
        raise PdfReadFailure("unreadable")
    if verdict.get("ok") is True:
        text, pages = verdict.get("text"), verdict.get("pages")
        if isinstance(text, str) and isinstance(pages, int):
            return PdfText(text=text, pages=pages)
        raise PdfReadFailure("unreadable")
    code = verdict.get("error")
    raise PdfReadFailure(code if isinstance(code, str) and code in FAILURE_CODES else "unreadable")
