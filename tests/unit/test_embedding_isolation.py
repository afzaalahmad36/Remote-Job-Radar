"""Where the embedding model may be loaded, and what the embedding path may reach.

The CV text goes through the embedding code, so (like the CV code, see test_cv_privacy.py) it must
not be able to reach a language model or the network. And the model is heavy (onnxruntime, numpy,
Pillow, ...): only the embed worker may load it, never the API or the main worker.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "app"

# The libraries that make up the model runtime.
MODEL_LIBRARIES = ("fastembed", "onnxruntime", "tokenizers", "PIL", "huggingface_hub")

# Anything here could carry text off the machine, or hand it to a model. (`app.tasks` is not
# listed: the embedding task module lives there. It must not import `app.tasks.jobs`, which
# reaches the LLM client; the app.llm entry makes this scan fail if it ever does.)
FORBIDDEN = (
    "app.llm",
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
    "http.client",
    "socket",
    "smtplib",
    "anthropic",
    "openai",
    "google.generativeai",
)

# The files the CV text (or a job's text) passes through on its way to a vector.
EMBEDDING_PATH = [
    APP / "matching" / "embeddings.py",
    APP / "matching" / "fake_embedder.py",
    APP / "matching" / "factory.py",
    APP / "matching" / "fastembed_embedder.py",
    APP / "services" / "embedding_service.py",
    APP / "tasks" / "embedding.py",
    APP / "tasks" / "common.py",
]


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _module_file(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] != "app":
        return None
    base = root.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def forbidden_imports(root: Path, start: Path) -> list[str]:
    """Forbidden modules imported by `start` or by any app module it imports, directly or not."""
    found: set[str] = set()
    seen: set[Path] = set()
    pending = [start]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN):
                found.add(f"{path.relative_to(root.parent)} imports {name}")
            elif (target := _module_file(root, name)) is not None:
                pending.append(target)
    return sorted(found)


# --- the embedding path cannot reach a model or the network ------------------------------------


@pytest.mark.parametrize("path", EMBEDDING_PATH, ids=lambda p: str(p.relative_to(APP)))
def test_the_embedding_path_reaches_no_llm_and_no_http_client_even_indirectly(path: Path) -> None:
    assert forbidden_imports(APP, path) == []


def test_the_scan_finds_a_violation_it_is_meant_to_find(tmp_path: Path) -> None:
    root = tmp_path / "app"
    root.mkdir()
    (root / "__init__.py").write_text("", "utf-8")
    (root / "helper.py").write_text("import requests\n", "utf-8")
    (root / "indirect.py").write_text("from app import helper\n", "utf-8")
    (root / "direct.py").write_text("from app.llm.client import build_client\n", "utf-8")
    (root / "clean.py").write_text("import hashlib\n", "utf-8")

    indirect = forbidden_imports(root, root / "indirect.py")
    direct = forbidden_imports(root, root / "direct.py")
    assert indirect and all("helper.py imports requests" in line for line in indirect)
    assert direct and all("direct.py imports app.llm" in line for line in direct)
    assert forbidden_imports(root, root / "clean.py") == []


def test_the_embedding_task_module_does_not_import_the_extraction_tasks() -> None:
    # tasks/jobs.py imports the LLM client; the embedding tasks must stay clear of it.
    assert "app.tasks.jobs" not in _imports(APP / "tasks" / "embedding.py")


# --- only one module loads the model -----------------------------------------------------------


def test_only_the_fastembed_adapter_imports_the_model_libraries() -> None:
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if path.name == "fastembed_embedder.py":
            continue
        for name in _imports(path):
            if name.split(".")[0] in MODEL_LIBRARIES:
                offenders.append(f"{path.relative_to(APP)} imports {name}")

    assert offenders == []


def test_the_factory_imports_the_real_embedder_only_when_asked_for_it() -> None:
    tree = ast.parse((APP / "matching" / "factory.py").read_text(encoding="utf-8"))
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "app.matching.fastembed_embedder" not in top_level  # imported inside the function


def test_the_api_and_the_main_worker_load_no_model_library() -> None:
    """In a clean interpreter, importing everything the API, the beat scheduler and the main
    worker import leaves the model runtime unloaded. (Only the embed worker, which calls
    `build_embedder`, ever loads it.)"""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import app.main, app.tasks.celery_app, app.tasks.jobs\n"
        "import app.tasks.embedding, app.tasks.queue, app.tasks.matching\n"
        "from app.matching import factory, embeddings, fake_embedder\n"
        f"libs = {MODEL_LIBRARIES!r}\n"
        "print(json.dumps(sorted(m for m in libs if m in sys.modules)))\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
    )

    assert result.returncode == 0, result.stderr[-500:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_the_embed_worker_does_load_the_model_libraries_when_it_builds_the_real_embedder() -> None:
    """The control for the test above: the same check does see the libraries when they load."""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import fastembed\n"
        f"libs = {MODEL_LIBRARIES!r}\n"
        "print(json.dumps(sorted(m for m in libs if m in sys.modules)))\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
    )

    assert {"fastembed", "onnxruntime"} <= set(json.loads(result.stdout.strip().splitlines()[-1]))
