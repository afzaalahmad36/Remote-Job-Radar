"""The CV code cannot reach a model or the network, and a CV object never prints its text.

Layer 3 of 4 (see test_cv_llm_guard.py for layers 1-2 and tests/integration/test_cv_canary.py for
layer 4).
"""

import ast
import uuid
from pathlib import Path

import pytest

from app.models import Cv, CvFormat

APP = Path(__file__).resolve().parent.parent.parent / "app"

# --- 3: the CV code cannot reach a model or the network -----------------------------------------

# Anything under these names could carry a CV off the machine. `socket`/`http.client`/`urllib` are
# the standard-library routes; the rest are the HTTP clients and LLM SDKs a project might add.
FORBIDDEN = (
    "app.llm",
    "app.tasks",
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
    "http.client",
    "socket",
    "smtplib",
    "anthropic",
    "openai",
    "google",
)


def _module_file(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] != "app":
        return None
    base = root.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def forbidden_imports(root: Path, start: Path, *, follow: bool) -> list[str]:
    """Forbidden modules imported by `start` (with `follow`: also by the app modules it imports)."""
    found: list[str] = []
    seen: set[Path] = set()
    pending = [start]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN):
                found.append(f"{path.relative_to(root.parent)} imports {name}")
            elif follow and (target := _module_file(root, name)) is not None:
                pending.append(target)
    return sorted(set(found))


def cv_files() -> list[Path]:
    return sorted((APP / "cv").glob("*.py")) + [
        APP / "services" / "cv_service.py",
        APP / "models" / "cv.py",
        APP / "schemas" / "cv.py",
    ]


@pytest.mark.parametrize("path", cv_files(), ids=lambda p: str(p.relative_to(APP)))
def test_cv_code_does_not_reach_an_llm_or_the_network_even_indirectly(path: Path) -> None:
    assert forbidden_imports(APP, path, follow=True) == []


def test_the_cv_router_imports_no_llm_or_network_module_directly() -> None:
    # The router imports api_deps, which legitimately reaches the task queue, so only its own
    # imports are checked; everything it calls for CV work is covered by the test above.
    assert forbidden_imports(APP, APP / "routers" / "cv.py", follow=False) == []


def test_the_scan_finds_a_direct_and_an_indirect_violation(tmp_path: Path) -> None:
    root = tmp_path / "app"
    (root / "cv").mkdir(parents=True)
    (root / "helper.py").write_text("import httpx\n", encoding="utf-8")
    (root / "cv" / "direct.py").write_text("from app.llm.client import build_client\n", "utf-8")
    (root / "cv" / "indirect.py").write_text("from app import helper\n", "utf-8")
    (root / "cv" / "clean.py").write_text("import hashlib\n", "utf-8")
    (root / "__init__.py").write_text("", "utf-8")

    direct = forbidden_imports(root, root / "cv" / "direct.py", follow=True)
    assert direct and all("direct.py" in line and "app.llm" in line for line in direct)
    assert len(forbidden_imports(root, root / "cv" / "indirect.py", follow=True)) == 1
    assert forbidden_imports(root, root / "cv" / "clean.py", follow=True) == []
    assert forbidden_imports(root, root / "cv" / "indirect.py", follow=False) == []


# --- the model never prints the text ------------------------------------------------------------


def test_a_cv_object_never_shows_its_text_when_printed() -> None:
    cv = Cv(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        text="CANARY-REPR-91 Alex Example, Python developer",
        source_format=CvFormat.TEXT,
        char_count=44,
        content_hash="0" * 64,
    )

    for rendered in (repr(cv), str(cv), f"{cv}", f"{[cv]}"):
        assert "CANARY-REPR-91" not in rendered
        assert "Alex" not in rendered
