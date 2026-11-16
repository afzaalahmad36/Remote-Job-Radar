"""The years and seniority suggestions are shown to the user and nothing else.

A guess read from a CV's wording must not silently decide which jobs a person sees, so no scoring
or matching code may read it, and it is not part of what the user saves as preferences.
"""

import ast
from pathlib import Path

import pytest

from app.models import Cv, JobPreferences
from app.schemas.preferences import PreferencesRead, PreferencesWrite

APP = Path(__file__).resolve().parent.parent.parent / "app"

# Modules and names that carry a suggestion.
SUGGESTION_MODULES = ("app.cv.suggestions",)
SUGGESTION_NAMES = frozenset(
    {
        "Suggestions",
        "YearsSuggestion",
        "SenioritySuggestion",
        "suggest",
        "suggest_years",
        "suggest_seniority",
        "SkillsView",  # the service object that holds them
        "get_skills",  # the service call that returns them
        "CvSuggestionsOut",
        "CvSkillsRead",
    }
)
SUGGESTION_WORDS = ("suggestion", "years_of_experience")

# The only modules that may touch them: the code that computes and serves them.
ALLOWED = {
    Path("cv") / "suggestions.py",
    Path("services") / "cv_service.py",
    Path("routers") / "cv.py",
    Path("schemas") / "cv.py",
}


def _imported_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def reads_suggestions(source: str, *, by_word: bool = True) -> bool:
    """True if the code imports a suggestion module or name; with `by_word` (the strict check for
    scoring code) also if it so much as mentions one, even in a comment or a string."""
    imported = _imported_names(source)
    if imported & set(SUGGESTION_MODULES) or imported & SUGGESTION_NAMES:
        return True
    lowered = source.lower()
    return by_word and any(word in lowered for word in SUGGESTION_WORDS)


def scoring_files(root: Path) -> list[Path]:
    """Everything under app/matching, and any module named like scoring, wherever it is."""
    files = set((root / "matching").rglob("*.py")) if (root / "matching").is_dir() else set()
    files |= {path for path in root.rglob("*.py") if "scor" in path.stem.lower()}
    files |= {path for path in root.rglob("*.py") if "matching" in path.relative_to(root).parts}
    return sorted(files)


def test_no_matching_or_scoring_module_reads_the_suggestions() -> None:
    offenders = [
        str(path.relative_to(APP))
        for path in scoring_files(APP)
        if reads_suggestions(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_only_the_cv_code_that_serves_them_touches_the_suggestions() -> None:
    # A wider net than "scoring": any module outside the CV code that imports the suggestions
    # is a new reader that needs a deliberate decision (and an edit of ALLOWED here).
    readers = {
        path.relative_to(APP)
        for path in APP.rglob("*.py")
        if reads_suggestions(path.read_text(encoding="utf-8"), by_word=False)
    }

    assert readers <= ALLOWED, sorted(str(path) for path in readers - ALLOWED)


def test_the_scan_catches_a_scoring_module_that_reads_them(tmp_path: Path) -> None:
    root = tmp_path / "app"
    (root / "matching").mkdir(parents=True)
    (root / "matching" / "scoring.py").write_text(
        "from app.cv.suggestions import suggest_years\n", encoding="utf-8"
    )
    (root / "matching" / "by_name.py").write_text(
        "def f(view):\n    return view.suggestions.seniority\n", encoding="utf-8"
    )
    (root / "matching" / "embeddings.py").write_text("import hashlib\n", encoding="utf-8")
    (root / "scoring_helpers.py").write_text("x = 'years_of_experience'\n", encoding="utf-8")
    (root / "other.py").write_text("x = 1\n", encoding="utf-8")

    found = {
        path.name
        for path in scoring_files(root)
        if reads_suggestions(path.read_text(encoding="utf-8"))
    }

    assert found == {"scoring.py", "by_name.py", "scoring_helpers.py"}


def test_the_scan_covers_the_files_it_is_meant_to_even_before_matching_exists() -> None:
    # When Phase 4 adds app/matching/, its files join the scan with no change to this test.
    assert all(path.suffix == ".py" for path in scoring_files(APP))


# --- not part of what is stored --------------------------------------------------------------


def _column_names(model: type) -> set[str]:
    return {column.name for column in model.__table__.columns}  # type: ignore[attr-defined]


@pytest.mark.parametrize("model", [JobPreferences, Cv], ids=["job_preferences", "cvs"])
def test_no_column_stores_a_suggestion(model: type) -> None:
    for name in _column_names(model):
        assert not any(word in name for word in ("suggest", "years", "experience")), name


def test_the_preferences_documents_have_no_field_for_a_suggestion() -> None:
    for schema in (PreferencesWrite, PreferencesRead):
        for name in schema.model_fields:
            assert not any(word in name for word in ("suggest", "years", "experience")), name


def test_the_preferences_columns_are_exactly_what_the_user_chooses() -> None:
    assert _column_names(JobPreferences) == {
        "id",
        "user_id",
        "seniority_levels",
        "remote_policies",
        "min_salary",
        "salary_currency",
        "salary_period",
        "notes",
        "created_at",
        "updated_at",
    }
