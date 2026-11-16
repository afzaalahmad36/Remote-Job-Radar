"""The embedding tasks, the hook after extraction and the sweep, with fakes only.

Synchronous on purpose: a task runs its own event loop with `asyncio.run`, which cannot start
inside an async test. No broker and no model: publishing is replaced by a recorder.
"""

import logging
import uuid
from datetime import UTC, datetime

import pytest
from kombu.exceptions import OperationalError as KombuOperationalError

from app.core.config import Settings
from app.matching.fake_embedder import FakeEmbedder
from app.models import ExtractionStatus
from app.tasks import embedding, jobs
from app.tasks.embedding import (
    DISABLED,
    enqueue_embed_cv,
    enqueue_embed_job,
    get_embedder,
    run_embed_cv,
    run_embed_job,
    run_sweep_embeddings,
)
from app.tasks.jobs import NO_EMBEDDING_AFTER, queue_embedding_after
from tests.extraction_helpers import RICH
from tests.task_helpers import Seeder

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
EXTRACTED = {
    "extraction_status": ExtractionStatus.EXTRACTED,
    "title": "Senior Python Developer",
    "stack": ["Python"],
    "extracted_at": NOW,
    "extraction_attempts": 1,
}


@pytest.fixture
def fake_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"embedding_provider": "fake"})


class Recorder:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[uuid.UUID] = []
        self.fail_with = fail_with

    def __call__(self, item: uuid.UUID) -> None:
        self.calls.append(item)
        if self.fail_with:
            raise self.fail_with


@pytest.fixture
def owner_source(seed: Seeder) -> tuple[uuid.UUID, uuid.UUID]:
    user_id = seed.user()
    return user_id, seed.source(user_id)


# --- the tasks ---------------------------------------------------------------------------------


def test_with_embeddings_off_the_tasks_do_nothing(
    seed: Seeder, settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, source_id = owner_source
    job_id = seed.job(source_id, RICH, **EXTRACTED)
    seed.cv(user_id, "Python developer.")

    assert settings.embedding_provider == "none"
    assert run_embed_job(job_id, settings=settings) == DISABLED
    assert run_embed_cv(user_id, settings=settings) == DISABLED
    assert seed.job_row(job_id).embedding is None and seed.cv_row(user_id).embedding is None


def test_the_job_task_embeds_a_job_and_a_second_run_changes_nothing(
    seed: Seeder, fake_settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    _, source_id = owner_source
    job_id = seed.job(source_id, RICH, **EXTRACTED)
    fake = FakeEmbedder()

    first = run_embed_job(job_id, settings=fake_settings, embedder=fake)
    second = run_embed_job(job_id, settings=fake_settings, embedder=fake)

    job = seed.job_row(job_id)
    assert (first, second) == ("embedded", "unchanged") and len(fake.calls) == 1
    assert len(job.embedding) == 384 and job.embedding_model == "fake-hash@v1"


def test_the_cv_task_embeds_the_cv(
    seed: Seeder, fake_settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, _ = owner_source
    seed.cv(user_id, "Backend engineer. Python and PostgreSQL.")

    result = run_embed_cv(user_id, settings=fake_settings, embedder=FakeEmbedder())

    assert result == "embedded" and len(seed.cv_row(user_id).embedding) == 384


def test_the_task_builds_one_embedder_per_process_and_none_when_off(
    settings: Settings, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embedding, "_cached", {})

    assert get_embedder(settings) is None
    first, second = get_embedder(fake_settings), get_embedder(fake_settings)

    assert first is second and first is not None  # built once, kept for the next task


# --- publishing --------------------------------------------------------------------------------


def test_nothing_is_published_while_embeddings_are_off(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a task was published although embeddings are off")

    monkeypatch.setattr(embedding.embed_job, "apply_async", forbidden)
    monkeypatch.setattr(embedding.embed_cv, "apply_async", forbidden)

    enqueue_embed_job(uuid.uuid4())
    enqueue_embed_cv(uuid.uuid4())


def test_a_task_is_published_with_its_id_when_embeddings_are_on(
    celery_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    from app.core.config import get_settings

    get_settings.cache_clear()
    sent: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(embedding.embed_job, "apply_async", lambda **kw: sent.append(("job", kw)))
    monkeypatch.setattr(embedding.embed_cv, "apply_async", lambda **kw: sent.append(("cv", kw)))
    job_id, user_id = uuid.uuid4(), uuid.uuid4()

    enqueue_embed_job(job_id)
    enqueue_embed_cv(user_id)

    assert sent == [
        ("job", {"args": [str(job_id)], "countdown": 0.0}),
        ("cv", {"args": [str(user_id)]}),
    ]


# --- the hook after extraction -----------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["extracted", "reused", "enriched", "already_done", "skipped"])
def test_an_extraction_that_can_leave_a_usable_job_queues_the_embedding(outcome: str) -> None:
    recorder, job_id = Recorder(), uuid.uuid4()

    assert queue_embedding_after(job_id, outcome, recorder) is True
    assert recorder.calls == [job_id]


@pytest.mark.parametrize("outcome", sorted(NO_EMBEDDING_AFTER))
def test_an_extraction_with_nothing_to_embed_queues_nothing(outcome: str) -> None:
    recorder = Recorder()

    assert queue_embedding_after(uuid.uuid4(), outcome, recorder) is False
    assert recorder.calls == []


def test_the_no_embedding_outcomes_are_the_ones_that_leave_no_usable_job() -> None:
    assert NO_EMBEDDING_AFTER == {"not_found", "failed", "not_a_job", "lost_race", "skipped_locked"}


def test_a_failed_publish_is_only_logged(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder(fail_with=KombuOperationalError("broker down"))
    caplog.set_level(logging.WARNING)

    assert queue_embedding_after(uuid.uuid4(), "extracted", recorder) is False
    assert any("not queued for embedding" in r.getMessage() for r in caplog.records)


def test_the_extract_task_queues_the_embedding_after_it_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, job_id = Recorder(), uuid.uuid4()
    monkeypatch.setattr(jobs, "run_extract_job", lambda _id: "extracted")
    monkeypatch.setattr(jobs, "enqueue_embed_job", recorder)

    result = jobs.extract_job.run(str(job_id))

    assert result == "extracted" and recorder.calls == [job_id]


def test_the_extract_task_survives_a_broker_that_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "run_extract_job", lambda _id: "extracted")
    monkeypatch.setattr(jobs, "enqueue_embed_job", Recorder(fail_with=OSError("no route")))

    assert jobs.extract_job.run(str(uuid.uuid4())) == "extracted"  # the extraction still stands


# --- the sweep ---------------------------------------------------------------------------------


def test_the_sweep_queues_what_has_no_vector(
    seed: Seeder, fake_settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, source_id = owner_source
    missing = seed.job(source_id, RICH, **EXTRACTED)
    seed.cv(user_id, "Python developer.")
    jobs_queued, cvs_queued = Recorder(), Recorder()

    result = run_sweep_embeddings(
        settings=fake_settings, enqueue_job=jobs_queued, enqueue_cv=cvs_queued
    )

    assert missing in jobs_queued.calls and user_id in cvs_queued.calls
    assert result["queued"] == len(jobs_queued.calls) + len(cvs_queued.calls)


def test_the_sweep_leaves_alone_what_is_already_embedded(
    seed: Seeder, fake_settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user_id, source_id = owner_source
    job_id = seed.job(source_id, RICH, **EXTRACTED)
    seed.cv(user_id, "Python developer.")
    fake = FakeEmbedder(fake_settings.embedding_model_id)
    run_embed_job(job_id, settings=fake_settings, embedder=fake)
    run_embed_cv(user_id, settings=fake_settings, embedder=fake)
    jobs_queued, cvs_queued = Recorder(), Recorder()

    run_sweep_embeddings(settings=fake_settings, enqueue_job=jobs_queued, enqueue_cv=cvs_queued)

    assert job_id not in jobs_queued.calls and user_id not in cvs_queued.calls


def test_a_new_model_makes_the_sweep_queue_the_old_vectors_again(
    seed: Seeder, fake_settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    _, source_id = owner_source
    job_id = seed.job(source_id, RICH, **EXTRACTED)
    run_embed_job(job_id, settings=fake_settings, embedder=FakeEmbedder("fake@old"))
    queued = Recorder()

    run_sweep_embeddings(settings=fake_settings, enqueue_job=queued, enqueue_cv=Recorder())

    assert job_id in queued.calls  # its vector is from another model than the configured one


def test_the_sweep_does_nothing_while_embeddings_are_off(
    seed: Seeder, settings: Settings, owner_source: tuple[uuid.UUID, uuid.UUID]
) -> None:
    _, source_id = owner_source
    seed.job(source_id, RICH, **EXTRACTED)
    queued = Recorder()

    result = run_sweep_embeddings(settings=settings, enqueue_job=queued, enqueue_cv=queued)

    assert result == {"jobs_due": 0, "cvs_due": 0, "queued": 0} and queued.calls == []


def test_a_down_queue_ends_the_sweep_with_one_warning_and_no_retry(
    seed: Seeder,
    fake_settings: Settings,
    owner_source: tuple[uuid.UUID, uuid.UUID],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, source_id = owner_source
    seed.job(source_id, RICH, **EXTRACTED)
    seed.job(source_id, RICH + " two", **EXTRACTED)
    down = Recorder(fail_with=KombuOperationalError("broker down"))
    caplog.set_level(logging.WARNING)

    result = run_sweep_embeddings(settings=fake_settings, enqueue_job=down, enqueue_cv=down)

    assert len(down.calls) == 1 and result["queued"] == 0  # it stopped at the first failure
    warnings = [r for r in caplog.records if "queue is unavailable" in r.getMessage()]
    assert len(warnings) == 1
