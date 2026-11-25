"""Text transcripts in docs/demo must hold nothing private (the PNGs need the human checklist)."""

import re
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent.parent / "docs" / "demo"
FORBIDDEN = {
    "a JWT": re.compile(r"eyJ[\w-]{10,}"),
    "a Google API key": re.compile(r"AIza[\w-]{20,}"),
    "a Windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE),
    "a personal email": re.compile(r"@(gmail|outlook|hotmail|yahoo)\.", re.IGNORECASE),
    "an email not on example.com": re.compile(r"[\w.+-]+@(?!example\.com)[\w-]+\.[a-z]{2,}"),
    "the transcript header": re.compile(r"(Username|Machine|RunAs User)\s*:", re.IGNORECASE),
}
TRANSCRIPTS = sorted(DEMO.glob("*.txt"))


@pytest.mark.parametrize("path", TRANSCRIPTS, ids=lambda p: p.name)
def test_a_transcript_holds_nothing_private(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")

    found = [name for name, pattern in FORBIDDEN.items() if pattern.search(text)]

    assert found == []


def test_the_checks_themselves_catch_what_they_should() -> None:
    samples = {
        "a JWT": "Bearer eyJhbGciOiJIUzI1NiJ9.abc",
        "a Windows user path": r"PS C:\Users\someone> ls",
        "a personal email": "mail a@gmail.com",
        "an email not on example.com": "write to jo@acme.io",
        "the transcript header": "Username: PC\someone",
    }
    for name, text in samples.items():
        assert FORBIDDEN[name].search(text), name
    assert not FORBIDDEN["an email not on example.com"].search("jobs@example.com")
