"""The CV and its embedding: queued by the upload, and kept away from every model and log.

The CV text is personal data. Embedding it runs on this machine (the embed worker has no route to
the internet), and the text must not appear in a log line or reach a language model.
"""

import logging

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import QueueUnavailableError
from app.cv.text import clean_cv_text, mask_cv_text
from app.llm.client import GuardedClient
from app.llm.providers.fake import FakeLLMClient
from app.llm.providers.gemini import GeminiClient
from app.matching.fake_embedder import FakeEmbedder
from app.models import User
from app.services.embedding_service import embed_cv
from app.services.extraction_types import SessionFactory
from tests.auth_helpers import auth_headers
from tests.extraction_helpers import add_user
from tests.fakes import FakeTaskQueue
from tests.integration.test_embedding_service import add_cv

CV = "/api/v1/cv"
TEXT = {"content-type": "text/plain"}
MARKER = "CANARY-EMBED-MARKER-7d20"
EMAIL = "embed.person@example.com"
PHONE = "+1 555 010 7788"
LINK = "https://github.com/embed-person"
CV_TEXT = f"""{MARKER} Person
{EMAIL} | {PHONE} | {LINK}
Backend engineer. Python, FastAPI, PostgreSQL, Docker.
""" + ("Built services and pipelines for customers. " * 400)  # long enough to be cut into chunks


# --- the upload queues the embedding -----------------------------------------------------------


async def test_uploading_a_cv_queues_its_embedding(
    db_client: AsyncClient, db_session: AsyncSession, task_queue: FakeTaskQueue
) -> None:
    headers = await auth_headers(db_client, "embed@example.com")

    response = await db_client.put(CV, content="Python developer", headers=headers | TEXT)

    user_id = await db_session.scalar(select(User.id))
    assert response.status_code == 200
    assert task_queue.cv_embedding_enqueued == [user_id]


async def test_a_refused_upload_queues_nothing(
    db_client: AsyncClient, task_queue: FakeTaskQueue
) -> None:
    headers = await auth_headers(db_client, "embed@example.com")

    response = await db_client.put(CV, content=b"\xff\xfe", headers=headers | TEXT)

    assert response.status_code == 415
    assert task_queue.cv_embedding_enqueued == []


async def test_a_down_queue_never_fails_the_upload(
    db_client: AsyncClient, task_queue: FakeTaskQueue, caplog: pytest.LogCaptureFixture
) -> None:
    headers = await auth_headers(db_client, "embed@example.com")
    task_queue.fail_cv_embedding_enqueue_with = QueueUnavailableError("down")
    caplog.set_level(logging.WARNING)

    response = await db_client.put(CV, content="Python developer", headers=headers | TEXT)

    assert response.status_code == 200 and response.json()["text"] == "Python developer"
    assert any("not queued for embedding" in r.getMessage() for r in caplog.records)


# --- the canary --------------------------------------------------------------------------------


def _leaks(value: str) -> list[str]:
    return [secret for secret in (MARKER, EMAIL, PHONE, LINK) if secret in value]


def _log_text(record: logging.LogRecord) -> str:
    parts = [record.getMessage(), str(record.args), record.exc_text or ""]
    parts.extend(str(v) for v in vars(record).values())
    return "\n".join(parts)


async def test_the_cv_text_never_reaches_an_llm_or_a_log_line_on_its_way_to_a_vector(
    factory: SessionFactory,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_calls: list[object] = []

    async def recorder(self: object, request: object) -> object:
        llm_calls.append(request)
        raise AssertionError("embedding a CV must never call an LLM client")

    for client_class in (GuardedClient, FakeLLMClient, GeminiClient):
        monkeypatch.setattr(client_class, "complete", recorder)
    user_id = await add_user(factory)
    stored = mask_cv_text(clean_cv_text(CV_TEXT))  # exactly what an upload stores
    await add_cv(factory, user_id, stored)
    embedder = FakeEmbedder()
    caplog.set_level(logging.DEBUG)

    await embed_cv(factory, user_id, embedder)

    sent = "\n".join(text for call in embedder.calls for text in call)
    assert MARKER in sent  # it is the user's own text, embedded on this machine
    assert [secret for secret in (EMAIL, PHONE, LINK) if secret in sent] == []  # masked
    assert llm_calls == []
    assert [_log_text(r) for r in caplog.records if _leaks(_log_text(r))] == []
    assert any("CV embedded" in r.getMessage() for r in caplog.records)  # we did log, only counts
    assert any("truncated" in r.getMessage() for r in caplog.records)


async def test_a_wrong_sized_vector_error_never_quotes_the_text(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    from collections.abc import Sequence

    from app.matching.embeddings import EmbeddingError

    class Broken(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.1] * 7 for _ in texts]

    user_id = await add_user(factory)
    await add_cv(factory, user_id, f"{MARKER} Python developer.")

    with pytest.raises(EmbeddingError) as error:
        await embed_cv(factory, user_id, Broken())

    assert MARKER not in str(error.value) and "Python developer" not in str(error.value)
