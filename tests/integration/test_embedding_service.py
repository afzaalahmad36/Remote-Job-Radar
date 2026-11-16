"""Embedding jobs and CVs, with the fake embedder and a real Postgres (pgvector included).

No model is involved: `FakeEmbedder` gives deterministic 384-dimension vectors, so these tests
check the plan the service follows (skip unchanged text, copy identical text, write with a
compare-and-set), not the quality of any embedding.
"""

import hashlib
import logging
import math
import threading
import uuid
from collections.abc import Sequence
from datetime import timedelta

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError

from app.matching.embeddings import CV_MAX_CHUNKS, EMBEDDING_DIM, EmbeddingError
from app.matching.fake_embedder import FakeEmbedder
from app.models import Cv, CvFormat, ExtractionStatus, Job, Seniority
from app.services import embedding_service
from app.services.embedding_service import EmbedOutcome, collect_work, embed_cv, embed_job
from app.services.extraction_types import SessionFactory
from tests.extraction_helpers import NOW, add_job, add_user, checked_out_connections, row

EXTRACTED = {
    "extraction_status": ExtractionStatus.EXTRACTED,
    "title": "Senior Python Developer",
    "company": "Acme",
    "stack": ["Python", "Django"],
    "seniority": Seniority.SENIOR,
    "extracted_at": NOW,
    "extraction_attempts": 1,
}
POST = "We are hiring a Senior Python Developer for our remote team. Stack: Python, Django."


async def extracted_job(factory: SessionFactory, post: str = POST, **fields: object) -> uuid.UUID:
    return await add_job(factory, post, **{**EXTRACTED, **fields})


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


# --- a job -------------------------------------------------------------------------------------


async def test_an_extracted_job_gets_a_vector_with_its_identity(factory: SessionFactory) -> None:
    job_id = await extracted_job(factory)
    fake = FakeEmbedder("fake@1")

    outcome = await embed_job(factory, job_id, fake)

    job = await row(factory, job_id)
    assert outcome is EmbedOutcome.EMBEDDED
    assert len(job.embedding) == EMBEDDING_DIM and math.isclose(
        norm(job.embedding), 1.0, abs_tol=1e-5
    )
    assert job.embedding_model == "fake@1"
    assert job.embedding_text_hash and len(job.embedding_text_hash) == 64
    assert job.embedded_at is not None
    assert "Title: Senior Python Developer" in fake.calls[0][0]


async def test_a_second_run_on_unchanged_text_does_not_ask_the_model(
    factory: SessionFactory,
) -> None:
    job_id = await extracted_job(factory)
    fake = FakeEmbedder()
    await embed_job(factory, job_id, fake)
    first = await row(factory, job_id)

    outcome = await embed_job(factory, job_id, fake)

    again = await row(factory, job_id)
    assert outcome is EmbedOutcome.UNCHANGED
    assert len(fake.calls) == 1  # the model ran once
    assert list(again.embedding) == list(first.embedding)
    assert again.embedded_at >= first.embedded_at  # checked again: the sweep will leave it be


async def test_a_job_that_is_not_extracted_is_not_embedded(factory: SessionFactory) -> None:
    job_id = await add_job(factory, POST)  # status pending
    fake = FakeEmbedder()

    outcome = await embed_job(factory, job_id, fake)

    assert outcome is EmbedOutcome.NOT_READY and fake.calls == []
    assert (await row(factory, job_id)).embedding is None


async def test_a_job_that_no_longer_exists_is_not_found(factory: SessionFactory) -> None:
    assert await embed_job(factory, uuid.uuid4(), FakeEmbedder()) is EmbedOutcome.NOT_FOUND


async def test_changed_text_gets_a_new_vector(factory: SessionFactory) -> None:
    job_id = await extracted_job(factory)
    fake = FakeEmbedder()
    await embed_job(factory, job_id, fake)
    before = await row(factory, job_id)
    async with factory() as session, session.begin():
        await session.execute(
            update(Job).where(Job.id == job_id).values(stack=["Rust", "Go"], title="Rust Engineer")
        )

    outcome = await embed_job(factory, job_id, fake)

    after = await row(factory, job_id)
    assert outcome is EmbedOutcome.EMBEDDED and len(fake.calls) == 2
    assert after.embedding_text_hash != before.embedding_text_hash
    assert list(after.embedding) != list(before.embedding)


async def test_a_different_model_re_embeds_the_same_text_and_replaces_the_identity(
    factory: SessionFactory,
) -> None:
    job_id = await extracted_job(factory)
    await embed_job(factory, job_id, FakeEmbedder("fake@1"))

    outcome = await embed_job(factory, job_id, FakeEmbedder("fake@2"))

    job = await row(factory, job_id)
    assert outcome is EmbedOutcome.EMBEDDED and job.embedding_model == "fake@2"


async def test_identical_text_in_another_job_is_copied_without_asking_the_model(
    factory: SessionFactory,
) -> None:
    first = await extracted_job(factory)
    second = await extracted_job(factory)  # the same post under another source
    fake = FakeEmbedder()
    await embed_job(factory, first, fake)

    outcome = await embed_job(factory, second, fake)

    a, b = await row(factory, first), await row(factory, second)
    assert outcome is EmbedOutcome.COPIED and len(fake.calls) == 1
    assert list(a.embedding) == list(b.embedding)
    assert a.embedding_text_hash == b.embedding_text_hash


async def test_a_vector_of_another_model_is_never_copied(factory: SessionFactory) -> None:
    first = await extracted_job(factory)
    second = await extracted_job(factory)
    await embed_job(factory, first, FakeEmbedder("fake@1"))
    other = FakeEmbedder("fake@2")

    outcome = await embed_job(factory, second, other)

    assert outcome is EmbedOutcome.EMBEDDED and len(other.calls) == 1


async def test_the_model_runs_with_no_database_connection_checked_out(
    factory: SessionFactory,
) -> None:
    job_id = await extracted_job(factory)

    class Watching(FakeEmbedder):
        seen: list[int] = []

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.seen.append(checked_out_connections(factory))
            return super().embed(texts)

    await embed_job(factory, job_id, Watching())

    assert Watching.seen == [0]  # nothing is held open while the model runs


async def test_a_job_that_changed_while_the_model_ran_keeps_no_stale_vector(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = await extracted_job(factory)
    real = embedding_service._embed_one

    async def re_extracted_meanwhile(embedder: FakeEmbedder, value: str) -> list[float]:
        async with factory() as session, session.begin():  # another worker re-extracts the job
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(extraction_attempts=2, extracted_at=NOW + timedelta(minutes=1))
            )
        return await real(embedder, value)

    monkeypatch.setattr(embedding_service, "_embed_one", re_extracted_meanwhile)

    outcome = await embed_job(factory, job_id, FakeEmbedder())

    assert outcome is EmbedOutcome.LOST_RACE
    assert (await row(factory, job_id)).embedding is None  # the sweep (or the new run) redoes it


async def test_two_workers_embedding_the_same_job_at_once_leave_one_consistent_vector(
    factory: SessionFactory,
) -> None:
    """Both run the model (neither has a vector yet) and both write the same answer. A barrier
    inside the embedder makes sure the two really overlap instead of running one after the other."""
    job_id = await extracted_job(factory)
    barrier = threading.Barrier(2, timeout=10)

    class Meeting(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            barrier.wait()
            return super().embed(texts)

    import asyncio

    outcomes = await asyncio.gather(
        embed_job(factory, job_id, Meeting()), embed_job(factory, job_id, Meeting())
    )

    job = await row(factory, job_id)
    assert outcomes == [EmbedOutcome.EMBEDDED, EmbedOutcome.EMBEDDED]
    assert len(job.embedding) == EMBEDDING_DIM and job.embedding_model == "fake-hash@v1"


async def test_a_vector_of_the_wrong_size_is_refused_and_nothing_is_stored(
    factory: SessionFactory,
) -> None:
    job_id = await extracted_job(factory)

    class WrongSize(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.5] * 128 for _ in texts]

    with pytest.raises(EmbeddingError, match="128 dimensions"):
        await embed_job(factory, job_id, WrongSize())

    assert (await row(factory, job_id)).embedding is None


async def test_postgres_itself_refuses_a_vector_of_the_wrong_size(factory: SessionFactory) -> None:
    job_id = await extracted_job(factory)

    with pytest.raises(DBAPIError, match="384 dimensions"):
        async with factory() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE jobs SET embedding = '[1,2,3]', embedding_model = 'm', "
                    "embedding_text_hash = 'h', embedded_at = now() WHERE id = :id"
                ),
                {"id": job_id},
            )


async def test_the_database_requires_the_vector_the_model_the_hash_and_the_time_together(
    factory: SessionFactory,
) -> None:
    job_id = await extracted_job(factory)

    with pytest.raises(DBAPIError, match="embedding_complete"):
        async with factory() as session, session.begin():
            await session.execute(
                update(Job).where(Job.id == job_id).values(embedding_model="only-the-model")
            )


# --- the CV ------------------------------------------------------------------------------------


async def add_cv(factory: SessionFactory, user_id: uuid.UUID, body: str) -> None:
    async with factory() as session, session.begin():
        session.add(
            Cv(
                user_id=user_id,
                text=body,
                source_format=CvFormat.TEXT,
                char_count=len(body),
                content_hash=hashlib.sha256(body.encode()).hexdigest(),
            )
        )


async def cv_row(factory: SessionFactory, user_id: uuid.UUID) -> Cv:
    async with factory() as session:
        return (await session.scalars(select(Cv).where(Cv.user_id == user_id))).one()


LONG_CV = ("Built backend services in Python and PostgreSQL. " * 12 + "\n") * 40  # ~24,000 chars


async def test_a_cv_gets_one_normalised_vector_from_its_chunks(factory: SessionFactory) -> None:
    user_id = await add_user(factory)
    await add_cv(factory, user_id, "Backend engineer. Python, FastAPI, PostgreSQL.")
    fake = FakeEmbedder("fake@1")

    outcome = await embed_cv(factory, user_id, fake)

    cv = await cv_row(factory, user_id)
    assert outcome is EmbedOutcome.EMBEDDED
    assert len(cv.embedding) == EMBEDDING_DIM and math.isclose(
        norm(cv.embedding), 1.0, abs_tol=1e-5
    )
    assert cv.embedding_model == "fake@1" and cv.embedded_at is not None
    assert len(fake.calls) == 1 and fake.calls[0] == [
        "Backend engineer. Python, FastAPI, PostgreSQL."
    ]


async def test_a_long_cv_is_cut_at_the_chunk_cap_one_chunk_per_model_call(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    user_id = await add_user(factory)
    marker = "CANARY-CV-EMBED-5521"
    await add_cv(factory, user_id, marker + " " + LONG_CV)
    fake = FakeEmbedder()
    caplog.set_level(logging.DEBUG)

    await embed_cv(factory, user_id, fake)

    assert len(fake.calls) == CV_MAX_CHUNKS and all(
        len(call) == 1 for call in fake.calls
    )  # batch 1
    truncated = [r.getMessage() for r in caplog.records if "truncated" in r.getMessage()]
    assert len(truncated) == 1 and f"kept={CV_MAX_CHUNKS}" in truncated[0]  # counts only
    assert str(user_id) in truncated[0]
    assert all(marker not in record.getMessage() for record in caplog.records)


async def test_a_cv_that_fits_logs_no_truncation(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    user_id = await add_user(factory)
    await add_cv(factory, user_id, "Short CV. Python.")
    caplog.set_level(logging.DEBUG)

    await embed_cv(factory, user_id, FakeEmbedder())

    assert not [r for r in caplog.records if "truncated" in r.getMessage()]


async def test_an_unchanged_cv_is_not_embedded_again(factory: SessionFactory) -> None:
    user_id = await add_user(factory)
    await add_cv(factory, user_id, "Python developer.")
    fake = FakeEmbedder()
    await embed_cv(factory, user_id, fake)

    outcome = await embed_cv(factory, user_id, fake)

    assert outcome is EmbedOutcome.UNCHANGED and len(fake.calls) == 1


async def test_a_new_model_re_embeds_the_cv(factory: SessionFactory) -> None:
    user_id = await add_user(factory)
    await add_cv(factory, user_id, "Python developer.")
    await embed_cv(factory, user_id, FakeEmbedder("fake@1"))

    outcome = await embed_cv(factory, user_id, FakeEmbedder("fake@2"))

    assert outcome is EmbedOutcome.EMBEDDED
    assert (await cv_row(factory, user_id)).embedding_model == "fake@2"


async def test_a_cv_replaced_while_the_model_ran_is_not_overwritten(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await add_user(factory)
    await add_cv(factory, user_id, "Old CV. Python.")
    real = embedding_service._embed_one

    async def uploaded_again(embedder: FakeEmbedder, value: str) -> list[float]:
        async with factory() as session, session.begin():
            await session.execute(
                update(Cv).where(Cv.user_id == user_id).values(content_hash="1" * 64)
            )
        return await real(embedder, value)

    monkeypatch.setattr(embedding_service, "_embed_one", uploaded_again)

    outcome = await embed_cv(factory, user_id, FakeEmbedder())

    assert outcome is EmbedOutcome.LOST_RACE
    assert (await cv_row(factory, user_id)).embedding is None


async def test_no_cv_means_nothing_to_embed(factory: SessionFactory) -> None:
    user_id = await add_user(factory)
    fake = FakeEmbedder()

    assert await embed_cv(factory, user_id, fake) is EmbedOutcome.NOT_FOUND
    assert fake.calls == []


# --- the sweep ---------------------------------------------------------------------------------


async def test_the_sweep_finds_extracted_jobs_without_a_vector_and_nothing_else(
    factory: SessionFactory,
) -> None:
    missing = await extracted_job(factory)
    pending = await add_job(factory, POST)  # not extracted
    done = await extracted_job(factory)
    fake = FakeEmbedder()
    await embed_job(factory, done, fake)

    work = await collect_work(factory, model_id=fake.model_id)

    assert missing in work.job_ids
    assert pending not in work.job_ids and done not in work.job_ids


async def test_the_sweep_finds_vectors_of_another_model_and_vectors_older_than_the_extraction(
    factory: SessionFactory,
) -> None:
    other_model = await extracted_job(factory)
    outdated = await extracted_job(factory)
    await embed_job(factory, other_model, FakeEmbedder("fake@old"))
    await embed_job(factory, outdated, FakeEmbedder("fake@1"))
    async with factory() as session, session.begin():  # re-extracted after it was embedded
        await session.execute(
            update(Job).where(Job.id == outdated).values(extracted_at=NOW + timedelta(days=3650))
        )

    work = await collect_work(factory, model_id="fake@1")

    assert other_model in work.job_ids and outdated in work.job_ids


async def test_the_sweep_takes_the_oldest_first_and_stops_at_the_limit(
    factory: SessionFactory,
) -> None:
    ids = [await extracted_job(factory) for _ in range(4)]

    work = await collect_work(factory, model_id="fake@1", limit=2)

    assert len(work.job_ids) == 2
    ordered = [i for i in work.job_ids if i in ids]
    assert ordered == [i for i in ids if i in ordered]  # creation order is kept


async def test_the_sweep_finds_cvs_without_a_vector_or_with_another_models_vector(
    factory: SessionFactory,
) -> None:
    fresh, current, old_model = [await add_user(factory) for _ in range(3)]
    for user_id in (fresh, current, old_model):
        await add_cv(factory, user_id, f"CV of {user_id}. Python.")
    await embed_cv(factory, current, FakeEmbedder("fake@1"))
    await embed_cv(factory, old_model, FakeEmbedder("fake@old"))

    work = await collect_work(factory, model_id="fake@1")

    assert fresh in work.cv_user_ids and old_model in work.cv_user_ids
    assert current not in work.cv_user_ids
