"""Reads ONE untrusted PDF in a process of its own. Run as a script, never imported by the app.

    python -I app/cv/pdf_worker.py MAX_PAGES MAX_CHARS MEMORY_LIMIT_MB < file.pdf

Why a separate process: a PDF can be built to make a parser spin or eat memory, and a thread
cannot be killed. A process can: `pdf_text.py` starts this script with a hard timeout and kills it
when the time is up. The PDF bytes arrive on stdin and one JSON line leaves on stdout, so nothing
touches the disk. stderr is not used, so no parser message can leak.

This file imports only the standard library and pypdf, never the `app` package.

Result: {"ok": true, "pages": N, "text": "..."} or {"ok": false, "error": "<fixed code>"}.
Codes: encrypted, too_many_pages, too_long, unreadable. Nothing from the PDF is ever put in an
error: pypdf's messages can quote the file's content.
"""

import importlib
import io
import json
import sys
from typing import Any

# The raw text may shrink when the app cleans it (runs of spaces, blank lines), so stop reading
# only when it is far past the limit; the app then enforces the exact limit on the cleaned text.
RAW_TEXT_SLACK = 4


def _limit_memory(limit_mb: int) -> None:
    """Cap this process's address space (RLIMIT_AS). On Linux the kernel then refuses any
    allocation past the cap, so a PDF that tries to use more gets a MemoryError instead of
    taking the machine's memory. Windows has no `resource` module (no cap there), and macOS
    accepts the call but does not enforce it."""
    try:
        resource: Any = importlib.import_module("resource")
    except ImportError:
        return
    limit = limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        return


def _error(code: str) -> dict[str, Any]:
    return {"ok": False, "error": code}


def extract(data: bytes, max_pages: int, max_chars: int) -> dict[str, Any]:
    """The verdict for one PDF. The page count is checked before any text is extracted."""
    from pypdf import PdfReader

    # `except Exception` is deliberate: pypdf raises many unrelated types for damaged files, and
    # every one of them means "cannot read this". The message is dropped on purpose.
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            return _error("encrypted")
        pages = len(reader.pages)
        if pages > max_pages:
            return _error("too_many_pages")
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            piece = page.extract_text() or ""
            total += len(piece) + 1
            if total > max_chars * RAW_TEXT_SLACK:
                return _error("too_long")
            parts.append(piece)
        return {"ok": True, "pages": pages, "text": "\n".join(parts)}
    except Exception:
        return _error("unreadable")


def main(argv: list[str]) -> int:
    try:
        max_pages, max_chars, memory_mb = int(argv[1]), int(argv[2]), int(argv[3])
    except (IndexError, ValueError):
        return 2
    _limit_memory(memory_mb)
    data = sys.stdin.buffer.read()
    sys.stdout.write(json.dumps(extract(data, max_pages, max_chars)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
