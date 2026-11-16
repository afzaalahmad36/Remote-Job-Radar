"""The REAL embedding model (fastembed + onnxruntime + the pinned weights).

Marked `model`: excluded from the normal run and from CI (CI's image has no weights and must never
contact Hugging Face). They run in Docker, against the weights baked into the image, with the
network switched off:

    docker run --rm --network none -v <repo>:/src:ro -v pytestlib:/pytestlib:ro \\
        -e PYTHONPATH=/src:/pytestlib --entrypoint python remote-job-radar:dev \\
        -m pytest --noconftest -c /dev/null -m model -p no:cacheprovider \\
        /src/tests/integration/test_fastembed_model.py

They are self-contained on purpose (no conftest, no database): they only need the image.
"""

import math
import os
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.matching.embeddings import EMBEDDING_DIM, EmbeddingError
from app.matching.factory import build_embedder
from tests.cv_helpers import synthetic_cv_text

pytestmark = pytest.mark.model

JOB = (
    "Title: Senior Python Developer\nCompany: Acme\nSeniority: senior\nRemote: remote\n"
    "Stack: Python, Django, PostgreSQL\nWe build backend services for logistics customers."
)
BAKER = "Title: Pastry Chef\nStack: \nWe bake croissants, sourdough bread and cakes every morning."


def real_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://u:p@localhost/x",
        redis_url="redis://localhost:1/0",
        secret_key=SecretStr("test-secret-key-at-least-32-characters-long"),
        embedding_provider="fastembed",
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    built = build_embedder(real_settings())
    assert built is not None
    return built


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_the_pinned_weights_are_in_the_image_and_load_offline() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    snapshot = Path("/models/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots")

    revisions = [p.name for p in snapshot.iterdir()]

    assert revisions == [real_settings().embedding_model_revision]  # exactly the pinned commit


def test_a_vector_has_384_dimensions_and_length_one(embedder) -> None:  # type: ignore[no-untyped-def]
    vector = embedder.embed([JOB])[0]

    assert len(vector) == EMBEDDING_DIM
    assert math.isclose(sum(x * x for x in vector), 1.0, abs_tol=1e-3)
    assert embedder.model_id == real_settings().embedding_model_id


def test_the_same_text_gives_the_same_vector(embedder) -> None:  # type: ignore[no-untyped-def]
    first, second = embedder.embed([JOB])[0], embedder.embed([JOB])[0]

    assert cosine(first, second) > 0.999999


def test_a_cv_is_closer_to_a_matching_job_than_to_an_unrelated_one(embedder) -> None:  # type: ignore[no-untyped-def]
    cv, job, baker = (embedder.embed([t])[0] for t in (synthetic_cv_text(), JOB, BAKER))

    assert cosine(cv, job) > cosine(cv, baker) + 0.1


def test_texts_are_embedded_one_at_a_time_in_order(embedder) -> None:  # type: ignore[no-untyped-def]
    batch = embedder.embed([JOB, BAKER])

    assert len(batch) == 2
    assert cosine(batch[0], embedder.embed([JOB])[0]) > 0.999
    assert cosine(batch[1], embedder.embed([BAKER])[0]) > 0.999


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_embedding_a_full_length_text_stays_under_the_memory_budget(embedder) -> None:  # type: ignore[no-untyped-def]
    long_text = ("python fastapi postgresql docker kubernetes backend engineer " * 100)[:2500]
    for _ in range(5):
        embedder.embed([long_text])

    peak_kb = next(
        int(line.split()[1]) for line in open("/proc/self/status") if line.startswith("VmHWM")
    )

    assert peak_kb / 1024 < 400  # the worker-embed budget (MiB), measured with batch size 1


def test_a_missing_revision_folder_is_a_clear_error_not_a_download() -> None:
    with pytest.raises(EmbeddingError, match="model files are missing"):
        build_embedder(real_settings(embedding_model_revision="0" * 40))
