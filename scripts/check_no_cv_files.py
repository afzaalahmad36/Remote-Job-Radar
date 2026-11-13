"""Fail if a CV-like file is tracked by git outside tests/fixtures/cv.

A real CV must never end up in this repository. Two things are rejected:
  * any word-processor or PDF file (.pdf, .doc, .docx, .odt, .rtf, .pages), and
  * any file named like a CV (cv, resume, curriculum vitae) that is not source code.
Source files are exempt from the name rule (app/models/cv.py is fine), because a .py file named
cv holds code. The only allowed place for CV-like data is tests/fixtures/cv, which holds invented
CVs only (see its README).

Run by pre-commit on the files being committed, and by a unit test over every tracked file, so
a CI run catches what a skipped hook missed. It prints paths only, never file content.

    uv run python -m scripts.check_no_cv_files [path ...]    # no paths: every tracked file
"""

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import PurePosixPath

ALLOWED_DIR = "tests/fixtures/cv/"
DOCUMENT_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".odt", ".rtf", ".pages"})
CODE_SUFFIXES = frozenset(
    {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".css", ".sql", ".toml", ".yml", ".yaml"}
)
# "cv", "cvs", "resume", "résumé", "curriculum", "curriculum vitae" as a whole word of the file
# name: my-cv.txt and Resume_2026.md match, but "cvs" inside "recvs" or "resumed" does not.
_CV_NAME = re.compile(r"(?<![a-z0-9])(?:cvs?|r[eé]sum[eé]s?|curriculum)(?![a-z0-9])")


def is_cv_like(path: str) -> bool:
    """True if `path` (relative, any slash style) must not be committed."""
    posix = path.replace(chr(92), "/").removeprefix("./")
    if posix.startswith(ALLOWED_DIR):
        return False
    file = PurePosixPath(posix)
    suffix = file.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        return True
    return suffix not in CODE_SUFFIXES and _CV_NAME.search(file.name.lower()) is not None


def offending(paths: Iterable[str]) -> list[str]:
    return sorted(path for path in paths if is_cv_like(path))


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True, text=True, encoding="utf-8"
    )
    return [name for name in result.stdout.split("\0") if name]


def main(argv: list[str]) -> int:
    bad = offending(argv[1:] or tracked_files())
    for path in bad:
        sys.stderr.write(f"CV-like file outside {ALLOWED_DIR}: {path}\n")
    if bad:
        sys.stderr.write("A real CV must never be committed. Move it outside the repository.\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
