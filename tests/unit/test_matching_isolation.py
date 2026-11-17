"""What the scoring path may reach, and what it may read.

Scoring works from a user's CV skills, vector and preferences. It must not be able to reach a
language model or the network, must not read the CV text or the preference notes, and must not read
the display-only suggestions (see test_suggestions_are_display_only.py for those).
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from app.matching.scoring import CvFacts, JobFacts, PreferenceFacts
from tests.unit.test_embedding_isolation import forbidden_imports

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "app"

SCORING_PATH = [
    APP / "matching" / "scoring.py",
    APP / "matching" / "reasons.py",
    APP / "matching" / "inputs_hash.py",
    APP / "services" / "match_service.py",
    APP / "repositories" / "match_repository.py",
    APP / "models" / "job_match.py",
]


@pytest.mark.parametrize("path", SCORING_PATH, ids=lambda p: str(p.relative_to(APP)))
def test_the_scoring_path_reaches_no_llm_and_no_http_client_even_indirectly(path: Path) -> None:
    assert forbidden_imports(APP, path) == []


@pytest.mark.parametrize("path", SCORING_PATH, ids=lambda p: str(p.relative_to(APP)))
def test_the_scoring_path_imports_no_celery_task_and_no_cv_text_code(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not {name for name in imported if name.startswith(("app.tasks", "app.cv", "app.llm"))}


def _attributes(path: Path) -> set[tuple[str, str]]:
    """(object name, attribute) pairs such as ("Cv", "text") found anywhere in a file."""
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            pairs.add((node.value.id, node.attr))
    return pairs


@pytest.mark.parametrize("path", SCORING_PATH, ids=lambda p: str(p.relative_to(APP)))
def test_the_scoring_path_never_names_the_cv_text_or_the_preference_notes(path: Path) -> None:
    attributes = _attributes(path)

    assert ("Cv", "text") not in attributes
    assert ("JobPreferences", "notes") not in attributes
    assert ("cv", "text") not in attributes
    assert ("preferences", "notes") not in attributes


def test_the_scoring_input_records_have_no_field_that_could_hold_cv_or_note_text() -> None:
    assert {f.name for f in dataclasses.fields(CvFacts)} == {"skills", "vector_id"}
    assert "notes" not in {f.name for f in dataclasses.fields(PreferenceFacts)}
    # A job's post text enters only as the start used by the language check (public text).
    assert "text_start" in {f.name for f in dataclasses.fields(JobFacts)}
    assert "raw_text" not in {f.name for f in dataclasses.fields(JobFacts)}


def test_the_cv_repository_row_has_no_text_field() -> None:
    from app.repositories.match_repository import CvRow, PreferenceRow

    assert {f.name for f in dataclasses.fields(CvRow)} == {
        "skills",
        "vector",
        "embedding_model",
        "embedding_text_hash",
    }
    assert "notes" not in {f.name for f in dataclasses.fields(PreferenceRow)}


def test_the_scan_catches_a_file_that_reads_the_cv_text(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(cv):\n    return Cv.text\n", encoding="utf-8")

    assert ("Cv", "text") in _attributes(bad)
