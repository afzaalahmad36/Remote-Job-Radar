import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.check_no_cv_files import is_cv_like, main, offending, tracked_files

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize(
    "path",
    [
        "cv.pdf",
        "docs/My CV.PDF",
        "resume.docx",
        "notes/Resume_2026.md",
        "my-cv.txt",
        "CV.json",
        "curriculum_vitae.txt",
        "résumé.md",
        "backup/cvs.csv",
        "scan.pdf",
        "app/data/portfolio.rtf",
        "tests\\fixtures\\feeds\\cv.txt",
        "tests/fixtures/cv_extra/resume.txt",  # a look-alike folder is not the allowed one
        "tests/fixtures/cv.txt",
    ],
)
def test_cv_like_files_are_rejected(path: str) -> None:
    assert is_cv_like(path)


@pytest.mark.parametrize(
    "path",
    [
        "tests/fixtures/cv/alex_example.txt",
        "tests/fixtures/cv/sample.pdf",  # the one place a (synthetic) PDF may live
        "app/models/cv.py",
        "app/routers/cv.py",
        "app/cv/text.py",
        "app/services/cv_service.py",
        "tests/unit/test_cv_text.py",
        "migrations/versions/f29a32b4831a_create_cvs.py",
        "README.md",
        "docs/ACCURACY.md",
        "app/resumed_tasks.py",  # "resumed" is not "resume"
        "scripts/recvs.txt",
    ],
)
def test_ordinary_files_and_the_fixtures_folder_are_allowed(path: str) -> None:
    assert not is_cv_like(path)


def test_offending_lists_only_the_bad_paths_sorted() -> None:
    assert offending(["b/cv.pdf", "app/cv.py", "a/resume.txt"]) == ["a/resume.txt", "b/cv.pdf"]


def test_the_script_fails_with_a_message_naming_the_path_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["prog", "notes/my-cv.txt", "app/main.py"]) == 1

    assert "notes/my-cv.txt" in capsys.readouterr().err
    assert main(["prog", "app/main.py"]) == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_no_tracked_file_in_this_repository_is_cv_like() -> None:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, check=False
    )
    if inside.returncode != 0:
        pytest.skip("not a git checkout")

    assert offending(tracked_files()) == []


def test_the_pre_commit_hook_runs_the_check() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "id: no-cv-files" in config
    assert "scripts.check_no_cv_files" in config
